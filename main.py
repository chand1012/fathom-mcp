"""
Fathom MCP server — hybrid FastAPI + FastMCP ASGI application.

Endpoints
---------
POST /webhook
    Receive Fathom ``newMeeting`` webhook events and store them locally.
    Validates the Svix-style HMAC-SHA256 signature before dispatching.

POST /sync
    Trigger a full download of recent meetings from the Fathom API.
    Requires ``Authorization: Bearer <SERVICE_API_KEY>`` or localhost origin.

/mcp
    FastMCP endpoint (Streamable HTTP).
    Requires ``Authorization: Bearer <SERVICE_API_KEY>``.
    Exposes tools: search_meetings, search_transcripts,
    search_meeting_transcripts, get_meeting, get_meeting_transcript.
"""

# Standard library
import sys
from pathlib import Path

# Ensure src-layout imports work when running as a script in Docker or locally.
SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load environment variables from .env file at the very beginning
from fathom_mcp.vector.embedder import ensure_embedding_model
from fathom_mcp.vector.database import init_database
from fathom_mcp.mcp.server import mcp
from fathom_mcp.core.config import settings
from fathom_mcp.api.router import router
from fastmcp.utilities.lifespan import combine_lifespans
from fastapi import FastAPI
import uvicorn
from typing import AsyncIterator
from contextlib import asynccontextmanager
import logging
import asyncio
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP ASGI sub-application
# path="/" means the MCP endpoint is at /mcp/ once mounted at /mcp below.
# ---------------------------------------------------------------------------
mcp_app = mcp.http_app(path="/")


# ---------------------------------------------------------------------------
# Lifespan: app startup/shutdown logic
# ---------------------------------------------------------------------------


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan handler.

    Startup
    -------
    1. Download the GGUF embedding model if not present (runs in a thread to
       avoid blocking the event loop during potentially long downloads).
    2. Initialise the SQLite database tables and vector index.

    Shutdown
    --------
    Runs in reverse order via combine_lifespans.
    """
    logger.info("Fathom MCP server starting up…")

    try:
        model_path = await asyncio.to_thread(ensure_embedding_model)
        logger.info("Embedding model ready: %s", model_path)
    except Exception:
        logger.exception("Failed to prepare embedding model")
        raise

    try:
        init_database()
        logger.info("Database initialised")
    except Exception:
        logger.exception("Failed to initialise database")
        raise

    logger.info("Fathom MCP server ready")
    yield
    logger.info("Fathom MCP server shut down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

api = FastAPI(
    title="Fathom MCP",
    description=(
        "Hybrid FastAPI + FastMCP server for Fathom meeting transcript ingestion "
        "and semantic search via the Model Context Protocol."
    ),
    lifespan=combine_lifespans(app_lifespan, mcp_app.lifespan),
)

api.include_router(router)

# Mount the FastMCP ASGI app at /mcp — MCP endpoint is therefore at /mcp/
api.mount("/mcp", mcp_app)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(api, host=settings.host, port=settings.port)
