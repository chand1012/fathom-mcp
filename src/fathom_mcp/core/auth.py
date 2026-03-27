"""Shared service API key authentication middleware."""

# Standard library
import hmac

# Third-party
from fastapi import status
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


DEFAULT_AUTH_EXEMPT_PATHS = (
    "/docs",
    "/docs/oauth2-redirect",
    "/openapi.json",
    "/redoc",
    "/webhook",
)


def extract_service_api_key(
    authorization: str | None,
    x_api_key: str | None,
) -> str | None:
    """Return the candidate service API key from supported auth headers."""
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            candidate = token.strip()
            if candidate:
                return candidate

    if x_api_key:
        candidate = x_api_key.strip()
        if candidate:
            return candidate

    return None


def _is_exempt_path(path: str, exempt_paths: tuple[str, ...]) -> bool:
    for exempt_path in exempt_paths:
        if path == exempt_path or path.startswith(f"{exempt_path}/"):
            return True
    return False


class ServiceApiKeyMiddleware:
    """Reject requests that do not present the configured service API key."""

    def __init__(
        self,
        app: ASGIApp,
        service_api_key: str,
        exempt_paths: tuple[str, ...] = DEFAULT_AUTH_EXEMPT_PATHS,
    ) -> None:
        self.app = app
        self.service_api_key = service_api_key.strip()
        self.exempt_paths = exempt_paths

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if _is_exempt_path(path, self.exempt_paths):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        token = extract_service_api_key(
            headers.get("authorization"),
            headers.get("x-api-key"),
        )
        if token and hmac.compare_digest(token, self.service_api_key):
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "detail": (
                    "Valid API key required via Authorization: Bearer <key> or "
                    "X-API-Key."
                )
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)
