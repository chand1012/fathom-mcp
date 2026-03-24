"""FastAPI router: webhook ingest, manual sync trigger, and tool endpoints."""

# Standard library
import hmac
import json
import logging

# Third-party
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import BaseModel
from typing import Annotated, Optional

# Local application
from fathom_mcp.api.service import fetch_and_store_recent_meetings
from fathom_mcp.core.config import settings
from fathom_mcp.vector.database import (
    get_meeting_by_id,
    get_meeting_transcripts,
    search_meetings_by_date_range,
    search_meetings_by_title_substring,
    search_similar_transcripts,
    search_similar_transcripts_in_meeting,
)
from fathom_mcp.vector.embedder import Embedder
from fathom_mcp.webhooks.handler import process_new_meeting, validate_webhook_signature

logger = logging.getLogger(__name__)

router = APIRouter()


def _extract_service_api_key(
    authorization: str | None,
    x_api_key: str | None,
) -> str | None:
    """Return the candidate service API key from supported auth headers."""
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token.strip()

    if x_api_key:
        return x_api_key.strip() or None

    return None


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def require_local_or_apikey(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """
    Dependency that permits requests originating from localhost OR bearing
    the service API key as a Bearer token or X-API-Key header.

    Raises:
        HTTPException: 401 if neither condition is met.
    """
    client_host = (request.client.host if request.client else "") or ""
    if client_host in {"127.0.0.1", "::1", "localhost"}:
        return

    token = _extract_service_api_key(
        authorization=authorization, x_api_key=x_api_key)
    if token and hmac.compare_digest(token, settings.service_api_key):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=(
            "Valid API key required via Authorization: Bearer <key> or X-API-Key, "
            "or request must originate from localhost."
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/webhook", status_code=200)
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    webhook_id: Annotated[str | None, Header(alias="webhook-id")] = None,
    webhook_timestamp: Annotated[str | None,
                                 Header(alias="webhook-timestamp")] = None,
    webhook_signature: Annotated[str | None,
                                 Header(alias="webhook-signature")] = None,
) -> dict:
    """
    Receive a Fathom ``newMeeting`` webhook event.

    Validates the Svix-style HMAC-SHA256 signature before accepting the payload.
    Processing (DB insert + transcript vectorization) runs in the background so
    Fathom receives a 200 response immediately.
    """
    if not all([webhook_id, webhook_timestamp, webhook_signature]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required webhook headers (webhook-id, webhook-timestamp, webhook-signature).",
        )

    # Read raw bytes first — required for correct signature verification.
    raw_body = await request.body()

    if not validate_webhook_signature(
        raw_body=raw_body,
        webhook_id=webhook_id,  # type: ignore[arg-type]
        webhook_timestamp=webhook_timestamp,  # type: ignore[arg-type]
        webhook_signature_header=webhook_signature,  # type: ignore[arg-type]
        secret=settings.fathom_webhook_secret,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook signature.",
        )

    payload = json.loads(raw_body)
    background_tasks.add_task(process_new_meeting, payload)
    return {"status": "accepted"}


@router.post(
    "/sync",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_local_or_apikey)],
)
async def trigger_sync(background_tasks: BackgroundTasks) -> dict:
    """
    Trigger a full download of all recent meetings from the Fathom API.

    Requires either a valid ``Authorization: Bearer <service_api_key>`` or
    ``X-API-Key: <service_api_key>`` header, or must be called from localhost.
    Returns 202 immediately; sync runs in the background.
    """
    background_tasks.add_task(fetch_and_store_recent_meetings)
    return {"status": "sync started"}


# ---------------------------------------------------------------------------
# Tool endpoints — mirrors the MCP tools for easy testing via /docs
# ---------------------------------------------------------------------------


@router.get(
    "/tools/search_meetings",
    dependencies=[Depends(require_local_or_apikey)],
    tags=["tools"],
    summary="Search meetings by title and/or date range",
)
async def tool_search_meetings(
    title: Annotated[Optional[str], Query(
        description="Substring to match in meeting or calendar title")] = None,
    start_date: Annotated[Optional[str], Query(
        description="ISO 8601 lower bound for created_at, e.g. 2025-01-01T00:00:00Z")] = None,
    end_date: Annotated[Optional[str], Query(
        description="ISO 8601 upper bound for created_at, e.g. 2025-12-31T23:59:59Z")] = None,
    limit: Annotated[int, Query(
        description="Maximum number of results", ge=1, le=100)] = 10,
) -> list[dict]:
    """Search meetings by title substring and/or date range."""
    if not title and not start_date and not end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of 'title', 'start_date', or 'end_date' must be provided.",
        )

    seen: dict[int, dict] = {}
    if title:
        for meeting in search_meetings_by_title_substring(title, limit=limit):
            seen[meeting["id"]] = dict(meeting)
    if start_date or end_date:
        for meeting in search_meetings_by_date_range(start_date, end_date, limit=limit):
            seen[meeting["id"]] = dict(meeting)

    sorted_results = sorted(
        seen.values(), key=lambda m: m["created_at"], reverse=True)
    return sorted_results[:limit]


@router.get(
    "/tools/search_transcripts",
    dependencies=[Depends(require_local_or_apikey)],
    tags=["tools"],
    summary="Semantic search across all meeting transcripts",
)
async def tool_search_transcripts(
    query: Annotated[str, Query(description="Natural language query")],
    limit: Annotated[int, Query(
        description="Maximum number of transcript chunks to return", ge=1, le=100)] = 10,
) -> list[dict]:
    """Search all stored transcripts using semantic (vector) similarity."""
    embedder = Embedder()
    embeddings = await embedder.get_embeddings(query)
    return search_similar_transcripts(embeddings[0], limit=limit)


@router.get(
    "/tools/search_meeting_transcripts",
    dependencies=[Depends(require_local_or_apikey)],
    tags=["tools"],
    summary="Semantic search within a single meeting's transcript",
)
async def tool_search_meeting_transcripts(
    meeting_id: Annotated[int, Query(description="ID of the meeting to search within")],
    query: Annotated[str, Query(description="Natural language query")],
    limit: Annotated[int, Query(
        description="Maximum number of transcript chunks to return", ge=1, le=100)] = 10,
) -> list[dict]:
    """Search transcripts within a specific meeting using semantic similarity."""
    embedder = Embedder()
    embeddings = await embedder.get_embeddings(query)
    return search_similar_transcripts_in_meeting(meeting_id, embeddings[0], limit=limit)


@router.get(
    "/tools/get_meeting",
    dependencies=[Depends(require_local_or_apikey)],
    tags=["tools"],
    summary="Get full details for a meeting by ID",
)
async def tool_get_meeting(
    meeting_id: Annotated[int, Query(description="ID of the meeting to retrieve")],
) -> dict:
    """Return all stored fields for a single meeting."""
    meeting = get_meeting_by_id(meeting_id)
    if meeting is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found.")
    return dict(meeting)


@router.get(
    "/tools/get_meeting_transcript",
    dependencies=[Depends(require_local_or_apikey)],
    tags=["tools"],
    summary="Get all transcript chunks for a meeting",
)
async def tool_get_meeting_transcript(
    meeting_id: Annotated[int, Query(description="ID of the meeting whose transcript to retrieve")],
) -> list[dict]:
    """Return all transcript chunks for a meeting, ordered by timestamp."""
    return get_meeting_transcripts(meeting_id)
