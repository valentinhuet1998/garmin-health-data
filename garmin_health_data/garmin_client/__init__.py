"""
Vendored Garmin Connect client.

A self-contained replacement for the ``python-garminconnect`` PyPI package,
purpose-built for the openetl pipeline. Provides:

- Five-strategy SSO login fallback chain (portal+cffi, portal+requests,
  mobile+cffi, mobile+requests, widget+cffi) with anti-rate-limit delays
  and ``curl_cffi`` TLS impersonation.
- DI OAuth2 token exchange and rotating-refresh persistence.
- Token storage at ``~/.garminconnect/<user_id>/garmin_tokens.json`` (same
  layout as the upstream library so existing tokens read cleanly).
- The 15 Garmin Connect API endpoints that the openetl pipeline consumes.
- Drop-in interface compatible with ``garminconnect.Garmin``: same method
  names, same signatures, same ``ActivityDownloadFormat`` enum values.

Public surface:

- :class:`GarminClient`: SSO (email/password) client.
- :class:`PersonalApiClient`: Personal API (API-key) client, serving a subset of
  the endpoints.
- :class:`HybridClient`: routes each call to the Personal API when it serves it,
  falling back to the SSO client.
- :class:`ActivityDownloadFormat`: enum of supported binary download formats.
- :class:`GarminAuthenticationError`,
  :class:`GarminConnectionError`,
  :class:`GarminTooManyRequestsError`,
  :class:`GarminUnsupportedEndpointError`: exceptions matching the upstream
  library's error semantics with shorter names.

The internal modules (:mod:`.client`, :mod:`.strategies`, :mod:`.api`,
:mod:`.tokens`, :mod:`.constants`, :mod:`.exceptions`, :mod:`.personal`,
:mod:`.hybrid`) are implementation detail; consumers should only import from this
package's top level.
"""

from .api import ActivityDownloadFormat
from .client import GarminClient
from .exceptions import (
    GarminAuthenticationError,
    GarminConnectionError,
    GarminTooManyRequestsError,
    GarminUnsupportedEndpointError,
)
from .hybrid import HybridClient
from .personal import PersonalApiClient

__all__ = [
    "ActivityDownloadFormat",
    "GarminAuthenticationError",
    "GarminClient",
    "GarminConnectionError",
    "GarminTooManyRequestsError",
    "GarminUnsupportedEndpointError",
    "HybridClient",
    "PersonalApiClient",
]
