"""
SQLite database interface with sqlite-vec support for Fathom MCP server.
Handles meetings and transcript chunks with vector embeddings.
"""

import json
import re
import sqlite3
from typing import List, Optional
from sqlite3 import Connection

import sqlite_vec
from sqlite_vec import serialize_float32

from fathom_mcp.core.config import settings


def _get_vec_table_dimension(conn: Connection) -> Optional[int]:
    """Return the configured embedding dimension for the vec table, if present."""
    result = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec_transcripts'"
    ).fetchone()
    if result is None or result[0] is None:
        return None

    match = re.search(r"float\[(\d+)\]", result[0])
    if match is None:
        return None
    return int(match.group(1))


def _get_connection(conn: Optional[Connection]) -> tuple[Connection, bool]:
    """Return a usable connection and whether this function created it."""
    if conn is not None:
        return conn, False
    return get_db_connection(), True


def get_db_connection() -> Connection:
    """
    Create and configure a SQLite connection with sqlite-vec extension.

    Returns:
        Configured SQLite connection
    """
    # Ensure the data directory exists
    import os
    os.makedirs(os.path.dirname(settings.vector_db_path), exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(settings.vector_db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Return connection with row factory for dict-like access
    conn.row_factory = sqlite3.Row
    return conn


def init_database(conn: Optional[Connection] = None) -> None:
    """
    Initialize the database with required tables if they don't exist.
    Called on application startup.
    """
    conn, should_close = _get_connection(conn)
    try:
        # Create meetings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                meeting_title TEXT,
                url TEXT NOT NULL,
                share_url TEXT NOT NULL,
                created_at TEXT NOT NULL,
                scheduled_start_time TEXT NOT NULL,
                scheduled_end_time TEXT NOT NULL,
                recording_start_time TEXT NOT NULL,
                recording_end_time TEXT NOT NULL,
                calendar_invitees_domains_type TEXT NOT NULL,
                transcript_language TEXT NOT NULL,
                default_summary_markdown TEXT,
                action_items_json TEXT,
                recorded_by_name TEXT NOT NULL,
                recorded_by_email TEXT NOT NULL,
                recorded_by_email_domain TEXT NOT NULL,
                recorded_by_team TEXT
            )
        """)

        # Create transcripts table with vector support
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                speaker_display_name TEXT NOT NULL,
                speaker_matched_invitee_email TEXT,
                text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_model TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (meeting_id) REFERENCES meetings (id) ON DELETE CASCADE
            )
        """)

        # Create index on meeting_id for faster lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transcripts_meeting_id 
            ON transcripts (meeting_id)
        """)

        # Create virtual table for vector search using sqlite-vec
        # This allows us to perform KNN search on embeddings
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_transcripts
            USING vec0(
                embedding float[{settings.embedding_dimension}]
            )
        """)

        vec_dimension = _get_vec_table_dimension(conn)
        if vec_dimension != settings.embedding_dimension:
            raise ValueError(
                "Existing vec_transcripts table dimension does not match the "
                f"configured embedding dimension. Expected {settings.embedding_dimension}, "
                f"found {vec_dimension}. Rebuild the vector database at "
                f"{settings.vector_db_path} for the new embedding model."
            )

        conn.commit()
    finally:
        if should_close:
            conn.close()


def insert_meeting(meeting_data: dict, conn: Optional[Connection] = None) -> None:
    """
    Insert a meeting into the database.

    Args:
        meeting_data: Dictionary containing meeting data from Fathom API
    """
    conn, should_close = _get_connection(conn)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO meetings (
                id, title, meeting_title, url, share_url, created_at,
                scheduled_start_time, scheduled_end_time, recording_start_time,
                recording_end_time, calendar_invitees_domains_type,
                transcript_language, default_summary_markdown,
                action_items_json,
                recorded_by_name, recorded_by_email, recorded_by_email_domain,
                recorded_by_team
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            meeting_data["recording_id"],
            meeting_data["title"],
            meeting_data.get("meeting_title"),
            meeting_data["url"],
            meeting_data["share_url"],
            meeting_data["created_at"],
            meeting_data["scheduled_start_time"],
            meeting_data["scheduled_end_time"],
            meeting_data["recording_start_time"],
            meeting_data["recording_end_time"],
            meeting_data["calendar_invitees_domains_type"],
            meeting_data["transcript_language"],
            meeting_data.get("default_summary", {}).get(
                "markdown_formatted") if meeting_data.get("default_summary") else None,
            json.dumps(meeting_data.get("action_items", [])),
            meeting_data["recorded_by"]["name"],
            meeting_data["recorded_by"]["email"],
            meeting_data["recorded_by"]["email_domain"],
            meeting_data["recorded_by"].get("team")
        ))
        if should_close:
            conn.commit()
    finally:
        if should_close:
            conn.close()


def insert_transcript_chunk(
    meeting_id: int,
    speaker_display_name: str,
    speaker_matched_invitee_email: Optional[str],
    text: str,
    timestamp: str,
    embedding: List[float],
    embedding_model: str = "default",
    conn: Optional[Connection] = None,
) -> None:
    """
    Insert a transcript chunk with its vector embedding.

    Args:
        meeting_id: ID of the meeting this transcript belongs to
        speaker_display_name: Name of the speaker
        speaker_matched_invitee_email: Email of matched calendar invitee (if any)
        text: The transcript text
        timestamp: Timestamp in HH:MM:SS format
        embedding: Vector embedding as list of floats
        embedding_model: Identifier for the embedding model used
    """
    if len(embedding) != settings.embedding_dimension:
        raise ValueError(
            f"Embedding dimension mismatch. Expected {settings.embedding_dimension}, "
            f"got {len(embedding)}"
        )

    conn, should_close = _get_connection(conn)
    try:
        # Serialize the embedding for storage
        embedding_blob = serialize_float32(embedding)

        # Insert into transcripts table
        cursor = conn.execute("""
            INSERT INTO transcripts (
                meeting_id, speaker_display_name, speaker_matched_invitee_email,
                text, timestamp, embedding, embedding_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            meeting_id,
            speaker_display_name,
            speaker_matched_invitee_email,
            text,
            timestamp,
            embedding_blob,
            embedding_model
        ))

        transcript_id = cursor.lastrowid

        # Also insert into the vector table for similarity search
        conn.execute(
            "INSERT INTO vec_transcripts(rowid, embedding) VALUES (?, ?)",
            (transcript_id, embedding_blob)
        )

        if should_close:
            conn.commit()
    finally:
        if should_close:
            conn.close()


def search_similar_transcripts(
    query_embedding: List[float],
    limit: int = 10,
    conn: Optional[Connection] = None,
) -> List[dict]:
    """
    Search for transcript chunks similar to the query embedding across all meetings.

    Args:
        query_embedding: Vector embedding to search for
        limit: Maximum number of results to return

    Returns:
        List of matching transcript chunks with metadata
    """
    if len(query_embedding) != settings.embedding_dimension:
        raise ValueError(
            f"Query embedding dimension mismatch. Expected {settings.embedding_dimension}, "
            f"got {len(query_embedding)}"
        )

    conn, should_close = _get_connection(conn)
    try:
        # Serialize the query embedding
        query_blob = serialize_float32(query_embedding)

        # Perform vector search using sqlite-vec
        # We join with the transcripts table to get the actual data
        results = conn.execute("""
            SELECT 
                t.id,
                t.meeting_id,
                t.speaker_display_name,
                t.speaker_matched_invitee_email,
                t.text,
                t.timestamp,
                t.embedding_model,
                t.created_at,
                m.title as meeting_title,
                m.meeting_title as meeting_calendar_title,
                vec_distance_cosine(t.embedding, ?) as similarity
            FROM transcripts t
            JOIN meetings m ON t.meeting_id = m.id
            WHERE t.embedding IS NOT NULL
            ORDER BY vec_distance_cosine(t.embedding, ?)
            LIMIT ?
        """, (query_blob, query_blob, limit)).fetchall()

        # Convert to list of dictionaries
        return [dict(row) for row in results]
    finally:
        if should_close:
            conn.close()


def search_similar_transcripts_in_meeting(
    meeting_id: int,
    query_embedding: List[float],
    limit: int = 10,
    conn: Optional[Connection] = None,
) -> List[dict]:
    """
    Search for transcript chunks similar to the query embedding within a specific meeting.

    Args:
        meeting_id: ID of the meeting to search within
        query_embedding: Vector embedding to search for
        limit: Maximum number of results to return

    Returns:
        List of matching transcript chunks with metadata
    """
    if len(query_embedding) != settings.embedding_dimension:
        raise ValueError(
            f"Query embedding dimension mismatch. Expected {settings.embedding_dimension}, "
            f"got {len(query_embedding)}"
        )

    # First verify the meeting exists
    conn, should_close = _get_connection(conn)

    if not meeting_exists(meeting_id, conn=conn):
        raise ValueError(f"Meeting with ID {meeting_id} does not exist")

    try:
        # Serialize the query embedding
        query_blob = serialize_float32(query_embedding)

        # Perform vector search using sqlite-vec within the specific meeting
        # We join with the transcripts table to get the actual data
        results = conn.execute("""
            SELECT 
                t.id,
                t.meeting_id,
                t.speaker_display_name,
                t.speaker_matched_invitee_email,
                t.text,
                t.timestamp,
                t.embedding_model,
                t.created_at,
                m.title as meeting_title,
                m.meeting_title as meeting_calendar_title,
                vec_distance_cosine(t.embedding, ?) as similarity
            FROM transcripts t
            JOIN meetings m ON t.meeting_id = m.id
            WHERE t.embedding IS NOT NULL
            AND t.meeting_id = ?
            ORDER BY vec_distance_cosine(t.embedding, ?)
            LIMIT ?
        """, (query_blob, meeting_id, query_blob, limit)).fetchall()

        # Convert to list of dictionaries
        return [dict(row) for row in results]
    finally:
        if should_close:
            conn.close()


def get_meeting_transcripts(
    meeting_id: int, conn: Optional[Connection] = None
) -> List[dict]:
    """
    Get all transcript chunks for a specific meeting.

    Args:
        meeting_id: ID of the meeting

    Returns:
        List of transcript chunks ordered by timestamp
    """
    conn, should_close = _get_connection(conn)
    try:
        results = conn.execute("""
            SELECT 
                t.id,
                t.speaker_display_name,
                t.speaker_matched_invitee_email,
                t.text,
                t.timestamp,
                t.embedding_model,
                t.created_at
            FROM transcripts t
            WHERE t.meeting_id = ?
            ORDER BY t.timestamp
        """, (meeting_id,)).fetchall()

        return [dict(row) for row in results]
    finally:
        if should_close:
            conn.close()


def get_meeting_by_id(
    meeting_id: int, conn: Optional[Connection] = None
) -> Optional[dict]:
    """
    Get a meeting by its ID.

    Args:
        meeting_id: ID of the meeting to retrieve

    Returns:
        Meeting data as dictionary, or None if not found
    """
    conn, should_close = _get_connection(conn)
    try:
        result = conn.execute("""
            SELECT * FROM meetings WHERE id = ?
        """, (meeting_id,)).fetchone()

        # Parse action_items_json back to Python object
        if result:
            meeting_dict = dict(result)
            if meeting_dict.get("action_items_json"):
                try:
                    meeting_dict["action_items"] = json.loads(
                        meeting_dict["action_items_json"])
                except json.JSONDecodeError:
                    meeting_dict["action_items"] = []
            else:
                meeting_dict["action_items"] = []
            # Remove the JSON column from the result
            del meeting_dict["action_items_json"]
            return meeting_dict
        return None
    finally:
        if should_close:
            conn.close()


def meeting_exists(meeting_id: int, conn: Optional[Connection] = None) -> bool:
    """
    Check if a meeting exists in the database.

    Args:
        meeting_id: ID of the meeting to check

    Returns:
        True if meeting exists, False otherwise
    """
    conn, should_close = _get_connection(conn)
    try:
        result = conn.execute(
            "SELECT 1 FROM meetings WHERE id = ? LIMIT 1",
            (meeting_id,)
        ).fetchone()

        return result is not None
    finally:
        if should_close:
            conn.close()


def search_meetings_by_title_substring(
    substring: str, limit: int = 10, conn: Optional[Connection] = None
) -> List[dict]:
    """
    Search for meetings by substring in title or meeting_title.

    Args:
        substring: Substring to search for (case-insensitive)
        limit: Maximum number of results to return

    Returns:
        List of matching meetings
    """
    conn, should_close = _get_connection(conn)
    try:
        # Use LIKE with wildcards for substring search, case-insensitive
        # We search in both title and meeting_title columns
        results = conn.execute("""
            SELECT * FROM meetings
            WHERE title LIKE ? ESCAPE '!' 
               OR meeting_title LIKE ? ESCAPE '!'
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{substring}%", f"%{substring}%", limit)).fetchall()

        # Convert to list of dictionaries and parse action_items_json
        meetings = []
        for row in results:
            meeting_dict = dict(row)
            if meeting_dict.get("action_items_json"):
                try:
                    meeting_dict["action_items"] = json.loads(
                        meeting_dict["action_items_json"])
                except json.JSONDecodeError:
                    meeting_dict["action_items"] = []
            else:
                meeting_dict["action_items"] = []
            # Remove the JSON column from the result
            del meeting_dict["action_items_json"]
            meetings.append(meeting_dict)

        return meetings
    finally:
        if should_close:
            conn.close()


def search_meetings_by_date_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
    conn: Optional[Connection] = None,
) -> List[dict]:
    """
    Search for meetings by date range on the created_at field.

    Args:
        start_date: Start date in ISO 8601 format (inclusive), optional
        end_date: End date in ISO 8601 format (inclusive), optional
        limit: Maximum number of results to return

    Returns:
        List of meetings within the date range, ordered by created_at descending
    """
    conn, should_close = _get_connection(conn)
    try:
        # Build the query dynamically based on provided date parameters
        query = "SELECT * FROM meetings WHERE 1=1"
        params = []

        if start_date is not None:
            query += " AND created_at >= ?"
            params.append(start_date)

        if end_date is not None:
            query += " AND created_at <= ?"
            params.append(end_date)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        results = conn.execute(query, params).fetchall()

        # Convert to list of dictionaries and parse action_items_json
        meetings = []
        for row in results:
            meeting_dict = dict(row)
            if meeting_dict.get("action_items_json"):
                try:
                    meeting_dict["action_items"] = json.loads(
                        meeting_dict["action_items_json"])
                except json.JSONDecodeError:
                    meeting_dict["action_items"] = []
            else:
                meeting_dict["action_items"] = []
            # Remove the JSON column from the result
            del meeting_dict["action_items_json"]
            meetings.append(meeting_dict)

        return meetings
    finally:
        if should_close:
            conn.close()
