"""Tests for service API key authentication."""

import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException
from starlette.requests import Request

from fathom_mcp.api.router import _extract_service_api_key, require_local_or_apikey
from fathom_mcp.core.config import Settings, settings


def _make_request(host: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/sync",
            "headers": [],
            "client": (host, 12345),
            "scheme": "http",
            "server": ("testserver", 80),
            "query_string": b"",
            "http_version": "1.1",
        }
    )


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
    assert _extract_service_api_key(authorization, x_api_key) == expected


@pytest.mark.asyncio
async def test_require_local_or_apikey_accepts_bearer_token() -> None:
    await require_local_or_apikey(
        _make_request("203.0.113.10"),
        authorization=f"Bearer {settings.service_api_key}",
    )


@pytest.mark.asyncio
async def test_require_local_or_apikey_accepts_x_api_key() -> None:
    await require_local_or_apikey(
        _make_request("203.0.113.10"),
        x_api_key=settings.service_api_key,
    )


@pytest.mark.asyncio
async def test_require_local_or_apikey_rejects_invalid_remote_key() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await require_local_or_apikey(
            _make_request("203.0.113.10"),
            authorization="Bearer wrong-key",
        )

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_route_auth_accepts_bearer_and_x_api_key() -> None:
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_local_or_apikey)])
    async def protected() -> dict[str, bool]:
        return {"ok": True}

    transport = httpx.ASGITransport(app=app, client=("203.0.113.10", 12345))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        bearer_response = await client.get(
            "/protected",
            headers={"Authorization": f"Bearer {settings.service_api_key}"},
        )
        x_api_key_response = await client.get(
            "/protected",
            headers={"X-API-Key": settings.service_api_key},
        )

    assert bearer_response.status_code == 200
    assert x_api_key_response.status_code == 200


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
