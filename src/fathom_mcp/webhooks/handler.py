"""Fathom webhook handler: signature validation and meeting ingestion."""

# Standard library
import base64
import hashlib
import hmac
import logging
import time
from typing import Any

# Local application
from fathom_mcp.core.config import settings
from fathom_mcp.vector.database import (
    get_db_connection,
    insert_meeting,
    insert_transcript_chunk,
    meeting_exists,
)
from fathom_mcp.vector.embedder import Embedder

logger = logging.getLogger(__name__)

_TIMESTAMP_TOLERANCE_SECONDS = 300  # 5 minutes — matches Fathom's recommendation


def validate_webhook_signature(
    raw_body: bytes,
    webhook_id: str,
    webhook_timestamp: str,
    webhook_signature_header: str,
    secret: str,
) -> bool:
    """
    Validate a Fathom webhook signature using the Svix signing scheme.

    Signed content is ``{webhook-id}.{webhook-timestamp}.{raw_body}`` HMAC'd
    with SHA-256 using the base64-decoded portion of the ``whsec_`` secret.

    Args:
        raw_body: Raw request body bytes (must not be parsed before this call).
        webhook_id: Value of the ``webhook-id`` request header.
        webhook_timestamp: Value of the ``webhook-timestamp`` request header
            (Unix epoch seconds as a string).
        webhook_signature_header: Value of the ``webhook-signature`` header.
            May contain multiple space-delimited ``v1,<base64>`` entries.
        secret: Webhook secret in ``whsec_<base64>`` format.

    Returns:
        True if any provided signature matches and the timestamp is within the
        acceptable tolerance window.
    """
    # --- Replay protection -----------------------------------------------
    try:
        ts = int(webhook_timestamp)
    except ValueError:
        logger.warning("Invalid webhook-timestamp value: %r", webhook_timestamp)
        return False

    if abs(time.time() - ts) > _TIMESTAMP_TOLERANCE_SECONDS:
        logger.warning(
            "Webhook timestamp outside tolerance window: %s", webhook_timestamp
        )
        return False

    # --- Decode the webhook secret ---------------------------------------
    try:
        if not secret.startswith("whsec_"):
            raise ValueError("Secret is missing the 'whsec_' prefix")
        secret_bytes = base64.b64decode(secret[len("whsec_") :])
    except Exception as exc:
        logger.error("Failed to decode webhook secret: %s", exc)
        return False

    # --- Compute expected signature --------------------------------------
    signed_content = f"{webhook_id}.{webhook_timestamp}.{raw_body.decode('utf-8')}"
    expected_sig = base64.b64encode(
        hmac.new(secret_bytes, signed_content.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")

    # --- Extract all provided signatures (strip ``v1,`` prefix) ---------
    provided_sigs: list[str] = []
    for token in webhook_signature_header.strip().split():
        _, _, sig = token.partition(",")
        provided_sigs.append(sig if sig else token)

    # --- Constant-time comparison against each candidate -----------------
    for sig in provided_sigs:
        try:
            if hmac.compare_digest(
                expected_sig.encode("utf-8"),
                sig.encode("utf-8"),
            ):
                return True
        except Exception:
            continue

    logger.warning("No valid webhook signature found for message %s", webhook_id)
    return False


async def process_new_meeting(payload: dict[str, Any]) -> None:
    """
    Persist a new meeting and vectorize its transcript from a ``newMeeting`` payload.

    This function is idempotent: if the meeting already exists in the database it
    returns immediately without making any changes.

    Args:
        payload: Decoded JSON body of the Fathom ``newMeeting`` webhook event.
    """
    recording_id: int = payload["recording_id"]
    conn = get_db_connection()
    try:
        if meeting_exists(recording_id, conn=conn):
            logger.info("Meeting %s already stored; skipping webhook", recording_id)
            return

        insert_meeting(payload, conn=conn)
        logger.info("Stored meeting %s from webhook", recording_id)

        transcript_items: list[dict[str, Any]] = payload.get("transcript") or []
        text_items = [item for item in transcript_items if item.get("text")]

        if text_items:
            texts = [item["text"] for item in text_items]
            embedder = Embedder()
            embeddings = await embedder.get_embeddings(texts)

            for item, embedding in zip(text_items, embeddings):
                speaker = item.get("speaker", {})
                insert_transcript_chunk(
                    meeting_id=recording_id,
                    speaker_display_name=speaker.get("display_name", "Unknown"),
                    speaker_matched_invitee_email=speaker.get(
                        "matched_calendar_invitee_email"
                    ),
                    text=item["text"],
                    timestamp=item.get("timestamp", "00:00:00"),
                    embedding=embedding,
                    embedding_model=settings.embedding_model,
                    conn=conn,
                )

            logger.info(
                "Vectorized %d transcript chunks for meeting %s",
                len(text_items),
                recording_id,
            )

        conn.commit()

    except Exception:
        conn.rollback()
        logger.exception("Failed to process webhook for meeting %s", recording_id)
        raise
    finally:
        conn.close()
