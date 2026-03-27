"""FastMCP server definition and tool registrations."""

# Standard library
import logging
from typing import Annotated

# Third-party
from fastmcp import FastMCP

# Local application
from fathom_mcp.vector.database import (
    get_meeting_by_id,
    get_meeting_transcripts,
    search_meetings_by_date_range,
    search_meetings_by_title_substring,
    search_similar_transcripts,
    search_similar_transcripts_in_meeting,
)
from fathom_mcp.vector.embedder import Embedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP("Fathom MCP")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(annotations={"readOnlyHint": True})
async def search_meetings(
    title: Annotated[
        str | None,
        "Substring to search for in the meeting or calendar title (case-insensitive)",
    ] = None,
    start_date: Annotated[
        str | None,
        "ISO 8601 lower bound for created_at, e.g. '2025-01-01T00:00:00Z'",
    ] = None,
    end_date: Annotated[
        str | None,
        "ISO 8601 upper bound for created_at, e.g. '2025-12-31T23:59:59Z'",
    ] = None,
    limit: Annotated[int, "Maximum number of meetings to return"] = 10,
) -> list[dict]:
    """
    Search meetings by title substring and/or date range.

    At least one of ``title``, ``start_date``, or ``end_date`` must be supplied.
    When multiple filters are provided the results are merged and deduplicated by
    meeting ID, then sorted descending by ``created_at``.
    """
    if not title and not start_date and not end_date:
        raise ValueError(
            "At least one of 'title', 'start_date', or 'end_date' must be provided."
        )

    seen: dict[int, dict] = {}

    if title:
        for meeting in search_meetings_by_title_substring(title, limit=limit):
            seen[meeting["id"]] = meeting

    if start_date or end_date:
        for meeting in search_meetings_by_date_range(start_date, end_date, limit=limit):
            seen[meeting["id"]] = meeting

    sorted_results = sorted(
        seen.values(), key=lambda m: m["created_at"], reverse=True)
    return sorted_results[:limit]


@mcp.tool(annotations={"readOnlyHint": True})
async def search_transcripts(
    query: Annotated[
        str,
        "Natural language query to find semantically similar transcript passages",
    ],
    limit: Annotated[int,
                     "Maximum number of transcript chunks to return"] = 10,
) -> list[dict]:
    """
    Search all meeting transcripts using semantic (vector) similarity.

    Returns the most relevant transcript chunks across every stored meeting,
    along with meeting metadata and a cosine similarity score.
    """
    embedder = Embedder()
    embeddings = await embedder.get_embeddings(query)
    return search_similar_transcripts(embeddings[0], limit=limit)


@mcp.tool(annotations={"readOnlyHint": True})
async def search_meeting_transcripts(
    meeting_id: Annotated[int, "ID of the meeting to search within"],
    query: Annotated[
        str,
        "Natural language query to find semantically similar transcript passages",
    ],
    limit: Annotated[int,
                     "Maximum number of transcript chunks to return"] = 10,
) -> list[dict]:
    """
    Search transcripts within a single meeting using semantic (vector) similarity.

    Useful for drilling into a specific meeting after discovering it via
    ``search_meetings`` or ``search_transcripts``.
    """
    embedder = Embedder()
    embeddings = await embedder.get_embeddings(query)
    return search_similar_transcripts_in_meeting(meeting_id, embeddings[0], limit=limit)


@mcp.tool(annotations={"readOnlyHint": True})
async def get_meeting(
    meeting_id: Annotated[int, "ID of the meeting to retrieve"],
) -> dict | None:
    """
    Get full details for a single meeting by its ID.

    Returns ``null`` if the meeting does not exist in the local database.
    """
    return get_meeting_by_id(meeting_id)


@mcp.tool(annotations={"readOnlyHint": True})
async def get_meeting_transcript(
    meeting_id: Annotated[int, "ID of the meeting whose transcript to retrieve"],
) -> list[dict]:
    """
    Get all transcript chunks for a meeting, ordered by timestamp.

    Each chunk includes the speaker's name, email (if matched), the spoken
    text, and the timestamp within the recording.
    """
    return get_meeting_transcripts(meeting_id)
