# Fathom MCP Server - Agent Guidelines

## Project Overview
MCP server for Fathom video recording service with SQLite vector storage for transcript chunks, webhook processing, and vector search.

Key technologies: FastAPI/FastMCP, SQLite-Vec, HTTPX

## Development Setup
- Python 3.14+
- UV package manager

### Installation
```bash
uv sync
```

## Development Commands

### Running Server
```bash
uv run main.py          # Standard
uv run --reload main.py # Development with auto-reload
```

### Testing (pytest)
```bash
uv run pytest               # All tests
uv run pytest test_file.py  # Specific file
uv run pytest test_file.py::test_function  # Specific function
uv run pytest --cov=src --cov-report=term-missing  # With coverage
```

### Linting & Formatting (Ruff)
```bash
uv run ruff check .         # Check for errors
uv run ruff check --fix .   # Auto-fix errors
uv run ruff format .        # Format code
uv run ruff format --check . # Check formatting only
```

### Type Checking (MyPy)
```bash
uv run mypy .
```

## Code Style Guidelines

### Imports
1. Order: Standard library → Third-party → Local
2. Group with blank lines between sections
3. Use absolute imports from project root
4. Avoid wildcard imports
5. Example:
```python
# Standard library
import asyncio
from typing import List, Optional

# Third-party
import httpx
from fastapi import FastAPI

# Local application
from fathom_mcp.core.config import Settings
```

### Formatting
- Line length: 88 characters (Ruff/Black default)
- Indentation: 4 spaces (no tabs)
- Blank lines: 2 between top-level definitions, 1 between methods
- Trailing commas in multi-line constructs
- Quotes: Double quotes for strings, single for characters

### Types
- Always annotate function parameters and return values
- Use typing module for complex types (List, Dict, Optional)
- Use is None/is not None for None checks
- Example:
```python
async def process_transcript(
    transcript_id: str,
    content: str,
    metadata: Optional[dict] = None
) -> List[VectorChunk]:
    if content is None:
        return []
    # implementation
```

### Naming Conventions
- Modules/packages: snake_case
- Classes: PascalCase
- Functions/methods: snake_case
- Constants: UPPER_SNAKE_CASE
- Variables: snake_case
- Private: _single_underscore
- Strongly private: __double_underscore

### Error Handling
- Prefer specific exceptions over bare except
- Log exceptions with context before handling
- Provide clear, actionable error messages
- Use context managers for resources
- Example:
```python
try:
    result = await external_api_call()
except httpx.HTTPStatusError as e:
    logger.error("API failed", status_code=e.response.status_code)
    raise ExternalServiceError(str(e)) from e
except Exception as e:
    logger.exception("Unexpected error")
    raise
```

### Documentation
- Google-style docstrings for public modules/classes/functions
- Module docstring: File purpose at top
- Function docstring: Args, returns, raises, examples
- Comments: Explain why, not what
- Example:
```python
def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector as list of floats
        vec2: Second vector as list of floats

    Returns:
        Cosine similarity score between -1 and 1

    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same dimension")
    # implementation
```

### Async/Await
- Mark I/O functions as async
- Always await async functions (unless creating background task)
- Avoid blocking calls; use asyncio.to_thread for CPU-bound tasks
- Use async context managers for clients
- Example:
```python
async def fetch_transcript(meeting_id: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.fathom_api_url}/meetings/{meeting_id}/transcript"
        )
        response.raise_for_status()
        return response.text
```

### Security
- Never hardcode secrets; use environment variables
- Validate all external inputs (webhooks, API parameters)
- Use parameterized queries with SQLite (prevents SQL injection)
- Configure CORS appropriately for deployment

## Git Conventions

### Commit Messages
- Format: `<type>(<scope>): <subject>`
- Types: feat, fix, docs, style, refactor, test, chore, perf, ci
- Subject line under 50 characters
- Include blank line and description for complex changes
- Example:
```
feat(transcript): add vector chunking service

Implement automatic chunking using semantic boundaries.
Add configuration for chunk size and overlap.
```

### Branching
- Main: main (stable)
- Features: feature/<short-description>
- Fixes: fix/<short-description>
- Releases: release/<version> (when needed)

### Pull Requests
- Focus on single concern
- Include tests for new features/fixes
- Update documentation when changing behavior
- Request review from at least one team member
- Ensure all checks pass before merging

## Additional Notes

### Database
SQLite with sqlite-vec. The database interface is implemented in `src/fathom_mcp/vector/database.py` with:

**Tables:**
- `meetings`: Stores meeting data from Fathom API including action items as JSON
- `transcripts`: Stores individual transcript chunks with vector embeddings as BLOBs
- `vec_transcripts`: Virtual table for efficient vector similarity search using sqlite-vec

**Key Functions:**
- `init_database()`: Initialize database tables (called on startup)
- `insert_meeting(meeting_data)`: Insert meeting data from Fathom API
- `insert_transcript_chunk(...)`: Insert transcript chunk with vector embedding
- `search_similar_transcripts(query_embedding, limit)`: Find similar transcript chunks across all meetings
- `search_similar_transcripts_in_meeting(meeting_id, query_embedding, limit)`: Find similar transcript chunks within a specific meeting
- `search_meetings_by_title_substring(substring, limit)`: Search meetings by substring in title or meeting_title
- `search_meetings_by_date_range(start_date, end_date, limit)`: Search meetings by date range on created_at
- `get_meeting_transcripts(meeting_id)`: Get all transcripts for a meeting
- `get_meeting_by_id(meeting_id)`: Get meeting by ID
- `meeting_exists(meeting_id)`: Check if meeting exists

### API Client
The Fathom API client is implemented in `src/fathom_mcp/api/client.py` and provides:

**Key Classes:**
- `FathomClient`: Main client for interacting with the Fathom External API
- `FathomAPIError`: Exception class for API-related errors

**Key Methods:**
- `list_meetings()`: List meetings with optional filtering (teams, dates, etc.)
- `get_meeting_details(recording_id)`: Get detailed information for a specific meeting
- `get_meeting_transcript(recording_id)`: Get transcript for a specific meeting
- `get_meeting_summary(recording_id)`: Get summary for a specific meeting
- `create_fathom_client()`: Convenience function to create a client from settings

### API Service
The meeting service is implemented in `src/fathom_mcp/api/service.py` and provides:

**Key Classes:**
- `MeetingService`: Service for fetching meetings from Fathom API and storing them in the database

**Key Methods:**
- `fetch_and_store_meetings()`: Fetch meetings from Fathom API and store new ones in the database
- `fetch_recent_meetings()`: Fetch recent meetings from the last N days
- `fetch_meetings_by_team()`: Fetch meetings for specific teams
- `fetch_and_store_recent_meetings()`: Convenience function to fetch and store recent meetings

**Features:**
- Automatic deduplication (skips existing meetings when store_new_only=True)
- Flexible filtering (by teams, dates, etc.)
- Configurable what data to include (action items, summary, transcript)
- Proper error handling and logging
- Async/await based for non-blocking operations

### Vector Store
Uses OpenAI-compatible embeddings by default (1536 dimensions). To change provider:
1. Update `src/fathom_mcp/vector/embedder.py` (to be implemented)
2. Adjust `src/fathom_mcp/core/config.py`
3. Ensure dimension compatibility

### Embedder
The embedder module is implemented in `src/fathom_mcp/vector/embedder.py` and provides:
- `Embedder` class for generating embeddings from OpenAI-compatible endpoints (including OpenAI, llama.cpp, etc.)
- Convenience function `get_embeddings()` for quick usage
- Supports both single strings and lists of strings
- Handles API authentication and error handling

### Webhook Handling
In `src/fathom_mcp/webhooks/handler.py` (to be implemented):
- Validate signatures with shared secret
- Idempotent processing for duplicates
- Dead letter queue for failing webhooks

### Performance
- Connection pooling for HTTP clients
- Batch vector insertions
- Async generators for large transcripts
- Monitor memory usage during processing

## Database Initialization
The database is automatically initialized on application startup in `main.py`:
- Creates tables if they don't exist
- Sets up the vec_transcripts virtual table for vector search
- No migrations needed - schema is created on first startup