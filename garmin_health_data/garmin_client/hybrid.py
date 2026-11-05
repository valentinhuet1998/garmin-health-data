"""
Hybrid Garmin client: Personal API key first, SSO tokens as a fallback.

The Personal API (API-key authentication) only serves part of the endpoints the
extractor needs. :class:`HybridClient` wraps an optional :class:`.personal.
PersonalApiClient` and an optional SSO-authenticated :class:`.client.GarminClient` and
routes each API method to the first one that can serve it:

1. The Personal API client, when configured and the method is in its
   :data:`~.personal.PersonalApiClient.SUPPORTED_METHODS`.
2. The SSO client, when configured and it defines the method (it lacks the
   Personal-API-only ones such as the daily summary or HRV status).

A method neither client can serve raises :class:`.exceptions.
GarminUnsupportedEndpointError`; callers check :meth:`HybridClient.supports` first to
skip such data types cleanly.
"""

from typing import Any, Optional

from .api import ActivityDownloadFormat
from .client import GarminClient
from .exceptions import GarminUnsupportedEndpointError
from .personal import PersonalApiClient


def missing_auth_hint(method_name: str) -> str:
    """
    Explain which authentication method an API method needs.

    :param method_name: Extractor-facing API method name.
    :return: Human-readable hint naming the ``garmin auth`` command to run.
    """
    if method_name in PersonalApiClient.SUPPORTED_METHODS:
        return (
            f"{method_name} needs a Personal API key. Run 'garmin auth' to set one up."
        )
    return (
        f"{method_name} is not available with a Personal API key. Run "
        "'garmin auth --sso' to also log in with email/password."
    )


class HybridClient:
    """
    Route Garmin API calls to the Personal API or the SSO client.
    """

    ActivityDownloadFormat = ActivityDownloadFormat

    def __init__(
        self,
        personal: Optional[PersonalApiClient] = None,
        sso: Optional[GarminClient] = None,
    ) -> None:
        """
        :param personal: Authenticated Personal API client, if an API key is set.
        :param sso: Authenticated SSO client, if email/password tokens exist.
        :raises ValueError: If neither client is provided.
        """
        if personal is None and sso is None:
            raise ValueError("HybridClient needs a Personal API or an SSO client")
        self.personal = personal
        self.sso = sso

    @property
    def full_name(self) -> Optional[str]:
        """
        :return: User's full name (only the SSO API exposes it).
        """
        return self.sso.full_name if self.sso is not None else None

    @property
    def auth_methods(self) -> str:
        """
        :return: Human-readable description of the configured auth methods.
        """
        methods = []
        if self.personal is not None:
            methods.append("Personal API key")
        if self.sso is not None:
            methods.append("email/password tokens")
        return " + ".join(methods)

    def _client_for(self, method_name: str) -> Optional[Any]:
        """
        :param method_name: Extractor-facing API method name.
        :return: The client that should serve ``method_name``, or None.
        """
        if self.personal is not None and self.personal.supports(method_name):
            return self.personal
        # The SSO client lacks the Personal-API-only methods (daily summary, HRV
        # status, ...), so only route there when it actually defines the method.
        if self.sso is not None and hasattr(self.sso, method_name):
            return self.sso
        return None

    def supports(self, method_name: str) -> bool:
        """
        :param method_name: Extractor-facing API method name.
        :return: True if a configured client can serve this method.
        """
        return self._client_for(method_name) is not None

    def __getattr__(self, name: str) -> Any:
        """
        Resolve public API methods to the routed client's bound method.

        Only called for attributes not found on the instance, so ``personal``, ``sso``
        and the properties above are never routed.

        :param name: Attribute name.
        :return: Bound method of the routed client.
        :raises GarminUnsupportedEndpointError: If no client can serve the method.
        :raises AttributeError: For private or unknown attributes.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        client = self._client_for(name)
        if client is None:
            raise GarminUnsupportedEndpointError(missing_auth_hint(name))
        return getattr(client, name)
