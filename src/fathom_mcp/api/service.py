"""
Service for fetching and storing Fathom meetings.
"""

import logging
from sqlite3 import Connection
from typing import List, Optional, Dict, Any
from datetime import datetime

from fathom_mcp.api.client import FathomClient, FathomAPIError, create_fathom_client
from fathom_mcp.vector.database import (
    get_db_connection,
    insert_meeting,
    insert_transcript_chunk,
    meeting_exists,
)
from fathom_mcp.vector.embedder import Embedder
from fathom_mcp.core.config import settings

logger = logging.getLogger(__name__)


class MeetingService:
    """Service for fetching meetings from Fathom API and storing them in the database."""

    def __init__(
        self,
        api_client: Optional[FathomClient] = None,
        embedder: Optional[Embedder] = None,
        db_connection: Optional[Connection] = None,
    ):
        """
        Initialize the meeting service.

        Args:
            api_client: Optional FathomClient instance. If not provided, one will be created.
            embedder: Optional Embedder instance for generating embeddings. If not provided, one will be created.
            db_connection: Optional SQLite connection. If not provided, one will be created and managed by this service.
        """
        self.api_client = api_client or create_fathom_client()
        self.embedder = embedder or Embedder()
        self._db_connection = db_connection
        self._owns_db_connection = db_connection is None

    def _get_db_connection(self) -> Connection:
        """Get or lazily create a database connection for the service lifecycle."""
        if self._db_connection is None:
            self._db_connection = get_db_connection()
        return self._db_connection

    async def aclose(self) -> None:
        """Close managed resources for the service."""
        if self._owns_db_connection and self._db_connection is not None:
            self._db_connection.close()
            self._db_connection = None

    async def __aenter__(self) -> "MeetingService":
        self._get_db_connection()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def fetch_and_store_meetings(
        self,
        limit: int = 10,
        cursor: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        teams: Optional[List[str]] = None,
        recorded_by: Optional[List[str]] = None,
        calendar_invitees_domains: Optional[List[str]] = None,
        calendar_invitees_domains_type: Optional[str] = None,
        include_action_items: bool = True,
        include_summary: bool = True,
        store_new_only: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch meetings from Fathom API and store new ones in the database.

        Args:
            limit: Maximum number of meetings to fetch per API call
            cursor: Pagination cursor for fetching next page
            created_after: Filter meetings created after this timestamp (ISO 8601)
            created_before: Filter meetings created before this timestamp (ISO 8601)
            teams: Filter by team names
            recorded_by: Filter by recorder email addresses
            calendar_invitees_domains: Filter by company domains of calendar invitees
            calendar_invitees_domains_type: Filter by internal/external participants
            include_action_items: Include action items in response
            include_summary: Include summary in response
            store_new_only: Only store meetings that don't already exist in database

        Returns:
            Dictionary with statistics about the operation
        """
        stats = {
            "fetched": 0,
            "stored": 0,
            "skipped": 0,
            "errors": 0,
            "next_cursor": None
        }
        conn = self._get_db_connection()

        try:
            # Fetch meetings from Fathom API
            logger.info(
                f"Fetching meetings from Fathom API with limit={limit}")
            response = await self.api_client.list_meetings(
                limit=limit,
                cursor=cursor,
                created_after=created_after,
                created_before=created_before,
                teams=teams,
                recorded_by=recorded_by,
                calendar_invitees_domains=calendar_invitees_domains,
                calendar_invitees_domains_type=calendar_invitees_domains_type,
                include_action_items=include_action_items,
                include_summary=include_summary,
            )

            meetings = response.get("items", [])
            stats["fetched"] = len(meetings)
            stats["next_cursor"] = response.get("next_cursor")

            logger.info(f"Fetched {len(meetings)} meetings from Fathom API")

            # Process each meeting
            for meeting_data in meetings:
                try:
                    recording_id = meeting_data["recording_id"]

                    # Check if we should skip existing meetings
                    if store_new_only and meeting_exists(recording_id, conn=conn):
                        stats["skipped"] += 1
                        logger.debug(
                            f"Skipping existing meeting {recording_id}")
                        continue

                    # Store the meeting
                    insert_meeting(meeting_data, conn=conn)
                    stats["stored"] += 1
                    logger.debug(f"Stored meeting {recording_id}")

                    if meeting_data.get("transcript"):
                        await self._vectorize_transcript(
                            recording_id, meeting_data["transcript"], conn=conn
                        )

                except Exception as e:
                    stats["errors"] += 1
                    logger.error(
                        f"Error processing meeting {meeting_data.get('recording_id', 'unknown')}: {e}")

            # Commit meeting/transcript inserts as a single unit for this fetch run.
            conn.commit()

            logger.info(
                f"Meeting fetch and store completed: "
                f"fetched={stats['fetched']}, stored={stats['stored']}, "
                f"skipped={stats['skipped']}, errors={stats['errors']}"
            )

        except FathomAPIError as e:
            conn.rollback()
            logger.error(f"Fathom API error: {e}")
            stats["errors"] += 1
        except Exception as e:
            conn.rollback()
            logger.error(f"Unexpected error in fetch_and_store_meetings: {e}")
            stats["errors"] += 1

        return stats

    async def fetch_recent_meetings(
        self,
        limit: int = 10,
        days_back: int = 30,
        store_new_only: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch recent meetings from the last N days, paging through all results.

        Args:
            limit: Page size per API call (Fathom caps at 10)
            days_back: Number of days back to fetch meetings from
            store_new_only: Only store meetings that don't already exist in database

        Returns:
            Dictionary with cumulative statistics about the operation
        """
        from datetime import timedelta, timezone
        created_after = (datetime.now(timezone.utc) -
                         timedelta(days=days_back)).isoformat()

        logger.info(
            f"Fetching meetings from the last {days_back} days (since {created_after})")

        totals: Dict[str, Any] = {"fetched": 0, "stored": 0,
                                  "skipped": 0, "errors": 0, "next_cursor": None}
        cursor: Optional[str] = None
        page = 0

        while True:
            page += 1
            logger.info(
                f"Fetching page {page} of meetings (cursor={cursor!r})")
            result = await self.fetch_and_store_meetings(
                limit=limit,
                cursor=cursor,
                created_after=created_after,
                store_new_only=store_new_only,
            )
            for key in ("fetched", "stored", "skipped", "errors"):
                totals[key] += result.get(key, 0)
            cursor = result.get("next_cursor")
            totals["next_cursor"] = cursor
            if not cursor:
                break

        return totals

    async def fetch_meetings_by_team(
        self,
        teams: List[str],
        limit: int = 10,
        store_new_only: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch meetings for specific teams, paging through all results.

        Args:
            teams: List of team names to filter by
            limit: Page size per API call (Fathom caps at 10)
            store_new_only: Only store meetings that don't already exist in database

        Returns:
            Dictionary with cumulative statistics about the operation
        """
        logger.info(f"Fetching meetings for teams: {teams}")

        totals: Dict[str, Any] = {"fetched": 0, "stored": 0,
                                  "skipped": 0, "errors": 0, "next_cursor": None}
        cursor: Optional[str] = None
        page = 0

        while True:
            page += 1
            logger.info(
                f"Fetching page {page} for teams {teams} (cursor={cursor!r})")
            result = await self.fetch_and_store_meetings(
                limit=limit,
                cursor=cursor,
                teams=teams,
                store_new_only=store_new_only,
            )
            for key in ("fetched", "stored", "skipped", "errors"):
                totals[key] += result.get(key, 0)
            cursor = result.get("next_cursor")
            totals["next_cursor"] = cursor
            if not cursor:
                break

        return totals

    _CHUNK_TARGET_CHARS: int = 4000

    def _build_transcript_chunks(
        self,
        transcript_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Group transcript items into ~2000-character text blocks.

        Each line inside a block is formatted as:
            ``HH:MM:SS - Speaker Name: utterance text``

        Returns a list of chunk dicts, each with:
            - ``text``: the combined block string
            - ``timestamp``: timestamp of the first item in the block
            - ``speaker_display_name``: name of the first speaker in the block
            - ``speaker_matched_invitee_email``: email of the first speaker (may be None)
        """
        chunks: List[Dict[str, Any]] = []
        current_lines: List[str] = []
        current_chars: int = 0
        first_item: Optional[Dict[str, Any]] = None

        for item in transcript_items:
            if "text" not in item:
                continue

            speaker = item.get("speaker", {})
            speaker_name = speaker.get("display_name", "Unknown")
            timestamp = item.get("timestamp", "00:00:00")
            line = f"{timestamp} - {speaker_name}: {item['text']}"

            if current_lines and current_chars + len(line) + 1 > self._CHUNK_TARGET_CHARS:
                # Flush the current block
                first_speaker = first_item.get(
                    "speaker", {}) if first_item else {}
                chunks.append({
                    "text": "\n".join(current_lines),
                    "timestamp": first_item.get("timestamp", "00:00:00") if first_item else "00:00:00",
                    "speaker_display_name": first_speaker.get("display_name", "Unknown"),
                    "speaker_matched_invitee_email": first_speaker.get("matched_calendar_invitee_email"),
                })
                current_lines = []
                current_chars = 0
                first_item = None

            if not current_lines:
                first_item = item

            current_lines.append(line)
            current_chars += len(line) + 1  # +1 for the newline separator

        # Flush any remaining lines
        if current_lines and first_item is not None:
            first_speaker = first_item.get("speaker", {})
            chunks.append({
                "text": "\n".join(current_lines),
                "timestamp": first_item.get("timestamp", "00:00:00"),
                "speaker_display_name": first_speaker.get("display_name", "Unknown"),
                "speaker_matched_invitee_email": first_speaker.get("matched_calendar_invitee_email"),
            })

        return chunks

    async def _vectorize_transcript(
        self,
        meeting_id: int,
        transcript_items: List[Dict[str, Any]],
        conn: Optional[Connection] = None,
    ) -> Dict[str, int]:
        """
        Chunk transcript items into ~2000-character blocks, embed each block,
        and store them in the database.

        Each chunk is formatted as newline-separated lines of:
            ``HH:MM:SS - Speaker Name: utterance text``
        """
        stats = {
            "total": len(transcript_items),
            "vectorized": 0,
            "stored": 0,
            "errors": 0,
        }

        if not transcript_items:
            return stats

        chunks = self._build_transcript_chunks(transcript_items)

        if not chunks:
            return stats

        try:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = await self.embedder.get_embeddings(texts)

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                try:
                    insert_transcript_chunk(
                        meeting_id=meeting_id,
                        speaker_display_name=chunk["speaker_display_name"],
                        speaker_matched_invitee_email=chunk["speaker_matched_invitee_email"],
                        text=chunk["text"],
                        timestamp=chunk["timestamp"],
                        embedding=embedding,
                        embedding_model=settings.embedding_model,
                        conn=conn,
                    )
                    stats["stored"] += 1
                    logger.debug(
                        "Stored transcript chunk %d/%d for meeting %s",
                        i + 1, len(chunks), meeting_id,
                    )
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(
                        "Error storing transcript chunk %d for meeting %s: %s",
                        i, meeting_id, e,
                    )

            stats["vectorized"] = stats["stored"]
            logger.info(
                "Vectorized transcript for meeting %s: chunks=%d, stored=%d, errors=%d",
                meeting_id, len(chunks), stats["stored"], stats["errors"],
            )

        except Exception as e:
            logger.error(
                "Error vectorizing transcript for meeting %s: %s", meeting_id, e
            )
            stats["errors"] = len(chunks)

        return stats


# Convenience function for quick usage
async def fetch_and_store_recent_meetings(
    limit: int = 50,
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Convenience function to fetch and store recent meetings.

    Args:
        limit: Maximum number of meetings to fetch
        days_back: Number of days back to fetch meetings from

    Returns:
        Dictionary with statistics about the operation
    """
    async with MeetingService() as service:
        return await service.fetch_recent_meetings(limit=limit, days_back=days_back)
