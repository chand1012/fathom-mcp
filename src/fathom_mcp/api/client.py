"""
Fathom API client for fetching meeting data.
"""

import httpx
from typing import List, Optional, Dict, Any

from fathom_mcp.core.config import settings


class FathomAPIError(Exception):
    """Exception raised for Fathom API errors."""
    pass


class FathomClient:
    """Client for interacting with the Fathom External API."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize the Fathom API client.

        Args:
            api_key: Fathom API key for authentication
            base_url: Base URL for the API (defaults to settings.fathom_api_url)
        """
        self.api_key = api_key
        self.base_url = base_url or settings.fathom_api_url
        self.headers = {
            "X-Api-Key": self.api_key,
            "Accept": "application/json"
        }

    async def _make_request(self, endpoint: str, params: Optional[Any] = None) -> Any:
        """
        Make an HTTP request to the Fathom API.

        Args:
            endpoint: API endpoint to call
            params: Query parameters

        Returns:
            JSON response from the API

        Raises:
            FathomAPIError: If the API returns an error
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                raise FathomAPIError(
                    f"Fathom API request failed: {e.response.status_code} - {e.response.text}"
                ) from e
            except httpx.RequestError as e:
                raise FathomAPIError(
                    f"Fathom API request failed: {str(e)}") from e

    async def list_meetings(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        teams: Optional[List[str]] = None,
        recorded_by: Optional[List[str]] = None,
        calendar_invitees_domains: Optional[List[str]] = None,
        calendar_invitees_domains_type: Optional[str] = None,
        include_action_items: bool = True,
        include_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        List meetings with optional filtering.

        Args:
            limit: Maximum number of meetings to return
            cursor: Pagination cursor
            created_after: Filter meetings created after this timestamp (ISO 8601)
            created_before: Filter meetings created before this timestamp (ISO 8601)
            teams: Filter by team names
            recorded_by: Filter by recorder email addresses
            calendar_invitees_domains: Filter by company domains of calendar invitees
            calendar_invitees_domains_type: Filter by internal/external participants
            include_action_items: Include action items in response
            include_summary: Include summary in response

        Returns:
            Dictionary containing meetings list and pagination info
        """
        params = []

        # Add basic parameters
        params.append(("limit", limit))
        params.append(("include_action_items", str(
            include_action_items).lower()))
        params.append(("include_summary", str(include_summary).lower()))
        params.append(("include_transcript", "true"))

        if cursor:
            params.append(("cursor", cursor))
        if created_after:
            params.append(("created_after", created_after))
        if created_before:
            params.append(("created_before", created_before))
        if teams:
            for team in teams:
                params.append(("teams[]", team))
        if recorded_by:
            for email in recorded_by:
                params.append(("recorded_by[]", email))
        if calendar_invitees_domains:
            for domain in calendar_invitees_domains:
                params.append(("calendar_invitees_domains[]", domain))
        if calendar_invitees_domains_type:
            params.append(("calendar_invitees_domains_type",
                          calendar_invitees_domains_type))

        return await self._make_request("meetings", params)

    async def get_meeting_details(self, recording_id: int) -> Dict[str, Any]:
        """
        Get detailed information for a specific meeting.

        Args:
            recording_id: The ID of the meeting recording

        Returns:
            Detailed meeting information
        """
        return await self._make_request(f"recordings/{recording_id}")

    async def get_meeting_summary(self, recording_id: int) -> Dict[str, Any]:
        """
        Get summary for a specific meeting.

        Args:
            recording_id: The ID of the meeting recording

        Returns:
            Meeting summary information
        """
        return await self._make_request(f"recordings/{recording_id}/summary")


# Convenience function to create a client from settings
def create_fathom_client() -> FathomClient:
    """
    Create a Fathom client using settings from the environment.

    Returns:
        Configured FathomClient instance
    """
    return FathomClient(
        api_key=settings.fathom_api_key,
        base_url=settings.fathom_api_url
    )
