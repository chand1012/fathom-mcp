"""Tests for service API key authentication middleware."""

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from fathom_mcp.core.auth import ServiceApiKeyMiddleware, extract_service_api_key
from fathom_mcp.core.config import Settings, settings


@pytest.mark.parametrize(
    ("authorization", "x_api_key", "expected"),
    [
        (f"Bearer {settings.service_api_key}", None, settings.service_api_key),
        ("bearer wrong-key", None, "wrong-key"),
        (None, settings.service_api_key, settings.service_api_key),
        (None, "   ", None),
        ("Token abc", None, None),
    ],
)
def test_extract_service_api_key(
    authorization: str | None,
    x_api_key: str | None,
    expected: str | None,
) -> None:
    assert extract_service_api_key(authorization, x_api_key) == expected


@pytest.mark.asyncio
async def test_middleware_protects_api_and_mcp_routes() -> None:
    app = FastAPI()
    app.add_middleware(
        ServiceApiKeyMiddleware,
        service_api_key=settings.service_api_key,
    )

    @app.get("/protected")
    async def protected() -> dict[str, bool]:
        return {"ok": True}

    mcp_app = FastAPI()

    @mcp_app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/webhook")
    async def webhook() -> dict[str, bool]:
        return {"ok": True}

    app.mount("/mcp", mcp_app)

    transport = httpx.ASGITransport(app=app, client=("203.0.113.10", 12345))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        unauthorized_response = await client.get("/protected")
        bearer_response = await client.get(
            "/protected",
            headers={"Authorization": f"Bearer {settings.service_api_key}"},
        )
        x_api_key_response = await client.get(
            "/protected",
            headers={"X-API-Key": settings.service_api_key},
        )
        invalid_response = await client.get(
            "/protected",
            headers={"Authorization": "Bearer wrong-key"},
        )
        mcp_response = await client.get(
            "/mcp/ping",
            headers={"Authorization": f"Bearer {settings.service_api_key}"},
        )
        webhook_response = await client.post("/webhook")

    assert unauthorized_response.status_code == 401
    assert bearer_response.status_code == 200
    assert x_api_key_response.status_code == 200
    assert invalid_response.status_code == 401
    assert mcp_response.status_code == 200
    assert webhook_response.status_code == 200


@pytest.mark.asyncio
async def test_middleware_rejects_missing_api_key() -> None:
    app = FastAPI()
    app.add_middleware(
        ServiceApiKeyMiddleware,
        service_api_key=settings.service_api_key,
    )

    @app.get("/protected")
    async def protected() -> dict[str, bool]:
        return {"ok": True}

    transport = httpx.ASGITransport(app=app, client=("203.0.113.10", 12345))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get("/protected")

    assert response.status_code == 401


def test_settings_reject_placeholder_service_api_key() -> None:
    with pytest.raises(ValueError, match="SERVICE_API_KEY"):
        Settings(
            fathom_api_url="https://api.fathom.ai/external/v1",
            fathom_api_key="test-fathom-key",
            fathom_webhook_secret="whsec_test",
            service_api_key="your-service-api-key-here",
        )


@pytest.mark.parametrize(
    "service_api_key",
    ["your_service_api_key", "  ", "ChangeMe"],
)
def test_settings_reject_other_placeholder_service_api_keys(
    service_api_key: str,
) -> None:
    with pytest.raises(ValueError, match="SERVICE_API_KEY"):
        Settings(
            fathom_api_url="https://api.fathom.ai/external/v1",
            fathom_api_key="test-fathom-key",
            fathom_webhook_secret="whsec_test",
            service_api_key=service_api_key,
        )
