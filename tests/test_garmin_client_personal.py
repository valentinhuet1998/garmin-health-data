"""
Tests for the Personal API client and the hybrid (API key + SSO) router.

All HTTP traffic goes through an in-memory fake session; no test touches the network.
"""

import base64
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from garmin_health_data.garmin_client import (
    ActivityDownloadFormat,
    GarminAuthenticationError,
    GarminConnectionError,
    GarminTooManyRequestsError,
    GarminUnsupportedEndpointError,
    HybridClient,
    PersonalApiClient,
)
from garmin_health_data.garmin_client import personal as personal_module
from garmin_health_data.garmin_client.personal import (
    PERSONAL_API_BASE_URL,
    PERSONAL_API_TOKEN_PATH,
    PERSONAL_DAILY_SUMMARIES_URL,
    PERSONAL_HEART_RATE_URL,
    PERSONAL_SLEEP_URL,
    _jwt_expiration,
    _retry_after_seconds,
)

API_KEY = "test-api-key"  # noqa: S105
ACTIVITIES_PATH = "/activitylist-service/activities/search/activities"


def _make_jwt(exp: float) -> str:
    """
    Build an unsigned JWT carrying an ``exp`` claim.

    :param exp: Expiration in seconds since the epoch.
    :return: JWT string.
    """

    def _b64(obj: Dict[str, Any]) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{_b64({'alg': 'none'})}.{_b64({'exp': exp})}.sig"


class FakeResponse:
    """
    Minimal stand-in for ``requests.Response``.
    """

    def __init__(
        self, status_code: int = 200, body: Any = None, content: Optional[bytes] = None
    ) -> None:
        """
        :param status_code: HTTP status code.
        :param body: JSON-serializable body, or a raw string.
        :param content: Raw bytes body; overrides ``body`` when given.
        """
        self.status_code = status_code
        if content is not None:
            self.content = content
        elif body is None:
            self.content = b""
        elif isinstance(body, str):
            self.content = body.encode()
        else:
            self.content = json.dumps(body).encode()

    @property
    def ok(self) -> bool:
        """
        :return: True for 2xx/3xx statuses.
        """
        return self.status_code < 400

    @property
    def text(self) -> str:
        """
        :return: Decoded body.
        """
        return self.content.decode(errors="replace")

    def json(self) -> Any:
        """
        :return: Parsed JSON body.
        """
        return json.loads(self.content)


class FakeSession:
    """
    Fake ``requests.Session`` returning queued responses per path.
    """

    def __init__(self) -> None:
        """
        Initialize empty response queues and call logs.
        """
        self.token_responses: List[FakeResponse] = []
        self.get_responses: Dict[str, List[FakeResponse]] = {}
        self.posts: List[Dict[str, Any]] = []
        self.gets: List[Tuple[str, Dict[str, Any]]] = []

    def queue_get(self, path: str, *responses: FakeResponse) -> None:
        """
        Queue responses for GETs to ``path``; the last one repeats.

        :param path: API path.
        :param responses: Responses to return in order.
        """
        self.get_responses.setdefault(path, []).extend(responses)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        """
        :param url: Request URL.
        :param kwargs: Request kwargs.
        :return: Next queued token response.
        """
        assert url == PERSONAL_API_BASE_URL + PERSONAL_API_TOKEN_PATH
        self.posts.append(kwargs)
        if len(self.token_responses) > 1:
            return self.token_responses.pop(0)
        return self.token_responses[0]

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        """
        :param url: Request URL.
        :param kwargs: Request kwargs.
        :return: Next queued response for the URL's path.
        """
        path = url[len(PERSONAL_API_BASE_URL) :]
        self.gets.append((path, kwargs))
        queue = self.get_responses.get(path)
        if not queue:
            return FakeResponse(404, {"error": "not found"})
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]


@pytest.fixture
def session() -> FakeSession:
    """
    Fake session that issues a valid 24h token.

    :return: FakeSession instance.
    """
    fake = FakeSession()
    fake.token_responses.append(FakeResponse(200, _make_jwt(time.time() + 86400)))
    return fake


@pytest.fixture
def client(session: FakeSession) -> PersonalApiClient:
    """
    Personal API client wired to the fake session.

    :param session: Fake session fixture.
    :return: PersonalApiClient instance.
    """
    return PersonalApiClient(API_KEY, session=session)  # type: ignore[arg-type]


class TestPersonalApiAuthentication:
    """
    Tests for the API-key-to-bearer-token exchange and user ID resolution.
    """

    def test_empty_key_rejected(self) -> None:
        """
        An empty or whitespace key fails fast, before any request.
        """
        with pytest.raises(GarminAuthenticationError):
            PersonalApiClient("   ")

    def test_authenticate_exchanges_key_and_resolves_user_id(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        The key is sent as ``ApiKey`` and the user ID read from ``userProfilePK``.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.queue_get(
            PERSONAL_HEART_RATE_URL, FakeResponse(200, {"userProfilePK": 42})
        )

        client.authenticate()

        assert client.user_id == 42
        assert session.posts[0]["headers"]["Authorization"] == f"ApiKey {API_KEY}"
        _, kwargs = session.gets[0]
        assert kwargs["headers"]["Authorization"].startswith("Bearer ")

    def test_user_id_falls_back_to_daily_summaries(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        Without a heart-rate ``userProfilePK``, the daily summaries ID is used.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.queue_get(PERSONAL_HEART_RATE_URL, FakeResponse(204))
        session.queue_get(
            PERSONAL_DAILY_SUMMARIES_URL,
            FakeResponse(200, {"data": [{"userProfilePk": 7}]}),
        )

        client.authenticate()

        assert client.user_id == 7

    def test_user_id_unresolvable_raises(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        No endpoint exposing the ID raises an authentication error.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.queue_get(PERSONAL_HEART_RATE_URL, FakeResponse(204))
        session.queue_get(PERSONAL_DAILY_SUMMARIES_URL, FakeResponse(200, {"data": []}))
        session.queue_get(ACTIVITIES_PATH, FakeResponse(200, []))

        with pytest.raises(GarminAuthenticationError):
            client.authenticate()

    def test_transient_error_falls_back_to_next_source(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        A 500 on the heart-rate source falls through to the activity owner ID.

        Reproduces a real failure: Garmin answered HTTP 500 to the heart-rate
        endpoint for the current day, which used to abort the whole account.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.queue_get(PERSONAL_HEART_RATE_URL, FakeResponse(500, {}))
        session.queue_get(PERSONAL_DAILY_SUMMARIES_URL, FakeResponse(500, {}))
        session.queue_get(ACTIVITIES_PATH, FakeResponse(200, [{"ownerId": 31}]))

        client.authenticate()

        assert client.user_id == 31

    def test_all_sources_failing_raises_retryable_error(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        When every source errors, a (retryable) connection error is raised.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.queue_get(PERSONAL_HEART_RATE_URL, FakeResponse(500, {}))

        with pytest.raises(GarminConnectionError, match="user ID"):
            client.authenticate()

    @pytest.mark.parametrize("status", [400, 401, 403])
    def test_rejected_key_raises_auth_error(
        self, client: PersonalApiClient, session: FakeSession, status: int
    ) -> None:
        """
        A rejected key raises an authentication error naming the settings page.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        :param status: HTTP status returned by the token endpoint.
        """
        session.token_responses[0] = FakeResponse(status, "denied")

        with pytest.raises(GarminAuthenticationError, match="personalApiToken"):
            client.authenticate()

    def test_rate_limited_token_exchange(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        HTTP 429 on the token endpoint raises the rate-limit error.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.token_responses[0] = FakeResponse(429, "slow down")

        with pytest.raises(GarminTooManyRequestsError):
            client.authenticate()

    def test_non_jwt_token_rejected(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        A token without an ``exp`` claim is rejected.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.token_responses[0] = FakeResponse(200, "not-a-jwt")

        with pytest.raises(GarminAuthenticationError):
            client.authenticate()

    def test_expiring_token_is_renewed(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        A token expiring within the refresh window is replaced before the request.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.token_responses[:] = [
            FakeResponse(200, _make_jwt(time.time() + 60)),
            FakeResponse(200, _make_jwt(time.time() + 86400)),
        ]
        session.queue_get(PERSONAL_HEART_RATE_URL, FakeResponse(200, {"x": 1}))

        client.get_heart_rates("2026-01-01")  # Obtains the short-lived token.
        client.get_heart_rates("2026-01-02")  # Renews it first.

        assert len(session.posts) == 2

    def test_401_renews_token_once_and_retries(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        A 401 triggers one token renewal and a retry.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.queue_get(
            PERSONAL_HEART_RATE_URL,
            FakeResponse(401),
            FakeResponse(200, {"heartRateValues": []}),
        )

        assert client.get_heart_rates("2026-01-01") == {"heartRateValues": []}
        assert len(session.posts) == 2

    def test_persistent_401_raises(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        A 401 after renewal raises an authentication error.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.queue_get(PERSONAL_HEART_RATE_URL, FakeResponse(401))

        with pytest.raises(GarminAuthenticationError):
            client.get_heart_rates("2026-01-01")

    def test_429_waits_retry_in_seconds_and_retries(
        self,
        client: PersonalApiClient,
        session: FakeSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A 429 waits ``retryInSeconds`` (+1s margin) and retries.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        :param monkeypatch: Pytest monkeypatch fixture.
        """
        sleeps: List[float] = []
        monkeypatch.setattr(personal_module.time, "sleep", sleeps.append)
        session.queue_get(
            PERSONAL_HEART_RATE_URL,
            FakeResponse(429, {"count": 101, "limit": 100, "retryInSeconds": 12}),
            FakeResponse(200, {"ok": True}),
        )

        assert client.get_heart_rates("2026-01-01") == {"ok": True}
        assert sleeps == [13.0]

    def test_persistent_429_raises_after_retries(
        self,
        client: PersonalApiClient,
        session: FakeSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A 429 that persists raises the rate-limit error after bounded waits.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        :param monkeypatch: Pytest monkeypatch fixture.
        """
        sleeps: List[float] = []
        monkeypatch.setattr(personal_module.time, "sleep", sleeps.append)
        session.queue_get(PERSONAL_HEART_RATE_URL, FakeResponse(429, "busy"))

        with pytest.raises(GarminTooManyRequestsError):
            client.get_heart_rates("2026-01-01")
        assert len(sleeps) == 3
        assert all(s == 30.0 for s in sleeps)  # Default wait without retryInSeconds.

    @pytest.mark.parametrize(
        "body, expected",
        [({"retryInSeconds": 500}, 90.0), ({"retryInSeconds": -5}, 1.0)],
    )
    def test_retry_wait_is_bounded(self, body: Dict[str, Any], expected: float) -> None:
        """
        The rate-limit wait is clamped to a sane range.

        :param body: 429 response body.
        :param expected: Expected wait in seconds.
        """
        assert _retry_after_seconds(FakeResponse(429, body)) == expected  # type: ignore

    def test_repr_hides_secrets(self, client: PersonalApiClient) -> None:
        """
        Neither the key nor the token appears in ``repr``.

        :param client: Personal API client fixture.
        """
        assert API_KEY not in repr(client)

    def test_jwt_expiration_parses_exp(self) -> None:
        """
        ``_jwt_expiration`` reads ``exp`` and returns None for garbage.
        """
        assert _jwt_expiration(_make_jwt(123.0)) == 123.0
        assert _jwt_expiration("garbage") is None


class TestPersonalApiMethods:
    """
    Tests for the Personal API data methods.
    """

    def test_sleep_wrapped_under_daily_sleep_dto(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        The bare sleep DTO is wrapped so the SLEEP processor parses it unchanged.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        dto = {"sleepStartTimestampGMT": 1, "sleepEndTimestampGMT": 2}
        session.queue_get(PERSONAL_SLEEP_URL, FakeResponse(200, dto))

        assert client.get_sleep_data("2026-01-01") == {"dailySleepDTO": dto}
        _, kwargs = session.gets[0]
        assert kwargs["params"] == {"date": "2026-01-01"}

    @pytest.mark.parametrize(
        "response",
        [FakeResponse(204), FakeResponse(200, {"sleepStartTimestampGMT": None})],
    )
    def test_sleep_without_data_returns_empty(
        self, client: PersonalApiClient, session: FakeSession, response: FakeResponse
    ) -> None:
        """
        Nights without a recorded sleep return a falsy payload.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        :param response: Empty or timestamp-less sleep response.
        """
        session.queue_get(PERSONAL_SLEEP_URL, response)

        assert client.get_sleep_data("2026-01-01") == {}

    def test_heart_rate_uses_personal_path(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        Heart rate hits the display-name-free Personal API path.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.queue_get(PERSONAL_HEART_RATE_URL, FakeResponse(200, {"a": 1}))

        assert client.get_heart_rates("2026-01-01") == {"a": 1}
        assert session.gets[0][1]["params"] == {"date": "2026-01-01"}

    def test_invalid_date_rejected(self, client: PersonalApiClient) -> None:
        """
        Malformed dates are rejected before any request.

        :param client: Personal API client fixture.
        """
        with pytest.raises(ValueError):
            client.get_heart_rates("01/01/2026")

    def test_shared_endpoints_reuse_api_wrappers(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        Activities, weigh-ins, and FIT downloads use the shared SSO paths.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.queue_get(
            "/activitylist-service/activities/search/activities",
            FakeResponse(200, [{"activityId": 1}]),
            FakeResponse(200, []),
        )
        session.queue_get(
            "/weight-service/weight/range/2026-01-01/2026-01-02",
            FakeResponse(200, {"dailyWeightSummaries": [{"summaryDate": "x"}]}),
        )
        session.queue_get(
            "/download-service/files/activity/1", FakeResponse(200, content=b"PK")
        )

        assert client.get_activities_by_date("2026-01-01", "2026-01-02") == [
            {
                "activityId": 1,
                "hasPolyline": False,
                "hasHeatMap": False,
                "purposeful": False,
            }
        ]
        assert client.get_body_composition("2026-01-01", "2026-01-02")
        assert client.download_activity(1) == b"PK"

    def test_menstrual_dayview_empty_returns_none(
        self, client: PersonalApiClient, session: FakeSession
    ) -> None:
        """
        A 204 dayview (no cycle data) returns None.

        :param client: Personal API client fixture.
        :param session: Fake session fixture.
        """
        session.queue_get(
            "/periodichealth-service/menstrualcycle/dayview/2026-01-01",
            FakeResponse(204),
        )

        assert client.get_menstrual_data_for_date("2026-01-01") is None

    def test_export_formats_not_served(self, client: PersonalApiClient) -> None:
        """
        Only ORIGINAL downloads are served by the Personal API.

        :param client: Personal API client fixture.
        """
        with pytest.raises(ValueError):
            client.download_activity(1, dl_fmt=ActivityDownloadFormat.TCX)

    def test_unsupported_path_raises_connection_error(
        self, client: PersonalApiClient
    ) -> None:
        """
        Paths outside the Personal API surface 404 as a connection error.

        :param client: Personal API client fixture.
        """
        with pytest.raises(GarminConnectionError, match="404"):
            client._connectapi("/metrics-service/metrics/trainingreadiness/x")


class TestHybridClient:
    """
    Tests for routing between the Personal API and SSO clients.
    """

    def test_requires_a_client(self) -> None:
        """
        A hybrid client without any underlying client is a programming error.
        """
        with pytest.raises(ValueError):
            HybridClient()

    def test_supported_methods_route_to_personal(self) -> None:
        """
        Personal-API methods prefer the Personal API even when SSO exists.
        """
        personal = PersonalApiClient(API_KEY)
        personal.get_heart_rates = MagicMock(return_value="personal")  # type: ignore
        sso = MagicMock()

        hybrid = HybridClient(personal=personal, sso=sso)

        assert hybrid.get_heart_rates("2026-01-01") == "personal"
        sso.get_heart_rates.assert_not_called()

    def test_other_methods_route_to_sso(self) -> None:
        """
        Methods the Personal API lacks go to the SSO client.
        """
        sso = MagicMock()
        sso.get_training_status.return_value = "sso"

        hybrid = HybridClient(personal=PersonalApiClient(API_KEY), sso=sso)

        assert hybrid.supports("get_training_status")
        assert hybrid.get_training_status("2026-01-01") == "sso"

    def test_unsupported_without_sso_raises(self) -> None:
        """
        Without SSO tokens, non-Personal-API methods are unsupported.
        """
        hybrid = HybridClient(personal=PersonalApiClient(API_KEY))

        assert not hybrid.supports("get_training_status")
        assert hybrid.supports("download_activity")
        with pytest.raises(GarminUnsupportedEndpointError, match="--sso"):
            hybrid.get_training_status("2026-01-01")

    def test_sso_only_serves_everything(self) -> None:
        """
        An SSO-only hybrid client behaves like the plain SSO client.
        """
        sso = MagicMock()
        sso.full_name = "Jane Doe"
        hybrid = HybridClient(sso=sso)

        assert hybrid.supports("get_heart_rates")
        assert hybrid.full_name == "Jane Doe"
        assert hybrid.auth_methods == "email/password tokens"

    def test_full_name_none_without_sso(self) -> None:
        """
        The Personal API exposes no full name.
        """
        hybrid = HybridClient(personal=PersonalApiClient(API_KEY))

        assert hybrid.full_name is None
        assert hybrid.auth_methods == "Personal API key"

    def test_private_attributes_not_routed(self) -> None:
        """
        Private attribute lookups raise AttributeError instead of routing.
        """
        hybrid = HybridClient(sso=MagicMock())

        with pytest.raises(AttributeError):
            hybrid._connectapi  # noqa: B018
