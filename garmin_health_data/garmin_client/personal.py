"""
Garmin Connect Personal API client (API-key authentication).

Garmin's Personal API (https://developer.garmin.com/gc-developer-program/personal/)
lets a user read their own Garmin Connect data with a Personal API key created at
https://connect.garmin.com/app/settings/personalApiToken, instead of an email/password
SSO login. The key is exchanged for a short-lived (24h) JWT bearer token; there is no
refresh token, so a new bearer token is simply requested from the same endpoint when
the current one is about to expire.

The Personal API gateway (``personal.connectapi.garmin.com``) serves the same Connect
service paths as ``connectapi.garmin.com``, but only a documented subset of them; every
other path returns HTTP 404. :data:`PersonalApiClient.SUPPORTED_METHODS` lists the
extractor-facing methods this client can serve. :class:`.hybrid.HybridClient` routes
the remaining ones to the SSO-authenticated :class:`.client.GarminClient` when one is
available.

The API key and the bearer token are only ever held in memory. They are never logged
or written to disk by this module.
"""

import base64
import json
import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import requests

from . import api
from .api import ActivityDownloadFormat, _validate_date_format
from .exceptions import (
    GarminAuthenticationError,
    GarminConnectionError,
    GarminTooManyRequestsError,
)

_LOGGER = logging.getLogger(__name__)

PERSONAL_API_BASE_URL = "https://personal.connectapi.garmin.com"
PERSONAL_API_TOKEN_PATH = (
    "/personalapitoken-service/personal-api/personal-api-key/token"  # noqa: S105
)

# Personal-API-specific endpoint paths. Paths shared with the SSO API (activities,
# activity detail, FIT download, weight range, menstrual dayview) are reused from
# :mod:`.constants` through the :mod:`.api` wrappers.
PERSONAL_SLEEP_URL = "/sleep-service/sleep/dailySleep"
PERSONAL_HEART_RATE_URL = "/wellness-service/wellness/dailyHeartRate"
PERSONAL_DAILY_SUMMARIES_URL = "/usersummary-service/v2/dailysummaries"
PERSONAL_USER_SUMMARY_URL = "/usersummary-service/usersummary/daily"
PERSONAL_HRV_URL = "/hrv-service/hrv"
PERSONAL_BODY_BATTERY_URL = "/wellness-service/wellness/bodyBattery/reports/daily"
PERSONAL_FITNESS_AGE_URL = "/fitnessage-service/fitnessage/v2"
PERSONAL_HR_ZONES_URL = "/biometric-service/heartRateZones/v2"
PERSONAL_FOOD_LOGS_URL = "/nutrition-service/food/logs"
PERSONAL_MENSTRUAL_SUMMARY_URL = "/periodichealth-service/menstrualcycle/summary"

# The body battery report endpoint rejects ranges longer than 31 days (HTTP 400), and
# returns one level per ``chunks`` slice of the day: 288 chunks gives the same 3-minute
# resolution as the SSO stress endpoint.
_BODY_BATTERY_MAX_RANGE_DAYS = 31
_BODY_BATTERY_CHUNKS = 288

# Fitness age moves slowly; sample it weekly (plus the last day of the range) to keep
# long backfills within the Personal API rate limit.
_FITNESS_AGE_SAMPLE_DAYS = 7

# Daily-summary fields that are all None on days without any wellness data (dates
# before the device was worn, days the watch was not worn).
_DAILY_SUMMARY_SIGNAL_FIELDS = (
    "totalSteps",
    "restingHeartRate",
    "averageStressLevel",
    "bodyBatteryHighestValue",
    "totalKilocalories",
)

# Request a new bearer token when the current one expires within this window.
_TOKEN_REFRESH_WINDOW_S = 15 * 60
_DEFAULT_TIMEOUT_S = 30

# The Personal API allows ~100 requests per rolling window and answers HTTP 429 with
# a JSON body carrying ``retryInSeconds``. Wait that long (capped) and retry, a few
# times, before surfacing the rate-limit error.
_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_MAX_WAIT_S = 90.0
_RATE_LIMIT_DEFAULT_WAIT_S = 30.0

# Boolean activity-list flags the SSO API always returns but the Personal API omits.
# The processor treats them as required; they are informational only (the DDL
# defaults them to 0), so missing ones are filled with False.
_ACTIVITY_FLAGS_DEFAULT_FALSE = ("hasPolyline", "hasHeatMap", "purposeful")


class PersonalApiClient:
    """
    Read Garmin Connect data with a Personal API key.

    Exposes the same method names and signatures as :class:`.client.GarminClient` for
    the subset of endpoints the Personal API serves, so the extractor can call either
    client interchangeably.
    """

    # Extractor-facing methods this client can serve. Anything else must go through
    # the SSO-authenticated GarminClient.
    SUPPORTED_METHODS = frozenset(
        {
            "get_sleep_data",
            "get_heart_rates",
            "get_menstrual_data_for_date",
            "get_body_composition",
            "get_activities_by_date",
            "get_activity_details",
            "download_activity",
            # Personal-API-only data types (no SSO equivalent in this project).
            "get_daily_summary",
            "get_hrv_data",
            "get_body_battery",
            "get_fitness_age",
            "get_heart_rate_zones",
            "get_activity_workouts",
            "get_food_logs",
            "get_menstrual_cycle_summary",
            "get_earliest_activity_date",
        }
    )

    ActivityDownloadFormat = ActivityDownloadFormat

    def __init__(
        self,
        api_key: str,
        base_url: str = PERSONAL_API_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_S,
        session: Optional[requests.Session] = None,
    ) -> None:
        """
        :param api_key: Garmin Connect Personal API key.
        :param base_url: Personal API base URL.
        :param timeout: Per-request timeout in seconds.
        :param session: Optional ``requests.Session`` (injected in tests).
        :raises GarminAuthenticationError: If the API key is empty.
        """
        api_key = (api_key or "").strip()
        if not api_key:
            raise GarminAuthenticationError("Personal API key is empty")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        self._bearer_token: Optional[str] = None
        self._bearer_expires_at: float = 0.0

        # Profile state, mirroring GarminClient. The Personal API exposes no endpoint
        # returning the user's full name, so ``full_name`` stays None.
        self.user_id: Optional[int] = None
        self.display_name: Optional[str] = None
        self.full_name: Optional[str] = None

    def __repr__(self) -> str:
        """
        :return: Representation that never includes the API key or token.
        """
        return f"{type(self).__name__}(base_url={self.base_url!r})"

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def supports(self, method_name: str) -> bool:
        """
        :param method_name: Extractor-facing API method name.
        :return: True if the Personal API serves this method.
        """
        return method_name in self.SUPPORTED_METHODS

    def authenticate(self) -> None:
        """
        Exchange the API key for a bearer token and resolve the Garmin user ID.

        :raises GarminAuthenticationError: If the key is rejected or the user ID cannot
            be determined.
        :raises GarminConnectionError: On transport or server errors.
        """
        self._obtain_bearer_token()
        self.user_id = self._resolve_user_id()

    def _obtain_bearer_token(self) -> None:
        """
        Exchange the Personal API key for a new bearer token.

        :raises GarminAuthenticationError: On HTTP 401/403 or a malformed token.
        :raises GarminTooManyRequestsError: On HTTP 429.
        :raises GarminConnectionError: On transport errors or other HTTP errors.
        """
        url = f"{self.base_url}{PERSONAL_API_TOKEN_PATH}"
        try:
            resp = self._session.post(
                url,
                headers={"Accept": "*/*", "Authorization": f"ApiKey {self._api_key}"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise GarminConnectionError(
                f"Personal API token exchange failed: {exc}"
            ) from exc

        if resp.status_code == 429:
            raise GarminTooManyRequestsError("Personal API token exchange rate limited")
        if resp.status_code in (400, 401, 403):
            raise GarminAuthenticationError(
                f"Personal API key rejected (HTTP {resp.status_code}). The key may be "
                "expired, deleted, or mistyped. Create a new one at "
                "https://connect.garmin.com/app/settings/personalApiToken."
            )
        if not resp.ok:
            raise GarminConnectionError(
                f"Personal API token exchange failed: HTTP {resp.status_code}"
            )

        token = resp.text.strip()
        expires_at = _jwt_expiration(token)
        if expires_at is None:
            raise GarminAuthenticationError(
                "Personal API token endpoint did not return a JWT with an exp claim"
            )
        self._bearer_token = token
        self._bearer_expires_at = expires_at

    def _token_expires_soon(self) -> bool:
        """
        :return: True if no bearer token is held or it expires within 15 minutes.
        """
        return (
            self._bearer_token is None
            or time.time() > self._bearer_expires_at - _TOKEN_REFRESH_WINDOW_S
        )

    def _resolve_user_id(self) -> int:
        """
        Determine the Garmin user (profile) ID of the key's owner.

        The Personal API profile-settings endpoint omits the numeric ID, so it is read
        from payloads that carry it, trying each source in turn: ``userProfilePK`` on
        today's and yesterday's heart-rate payloads, the daily summaries'
        ``userProfilePk``, then the newest activity's ``ownerId``. All match the
        ``id`` the SSO API reports. A failing source (e.g. a transient HTTP 500 on
        today's heart rate) moves on to the next one instead of failing the account.

        :return: Garmin user ID.
        :raises GarminAuthenticationError: If no source exposes the ID.
        :raises GarminConnectionError: If every source failed with an HTTP or
            transport error (so callers can retry).
        """
        today = date.today()
        yesterday = (today - timedelta(days=1)).isoformat()
        today_str = today.isoformat()

        def _from_heart_rate(day: str) -> Any:
            data = self._connectapi(PERSONAL_HEART_RATE_URL, params={"date": day})
            return (data or {}).get("userProfilePK") if isinstance(data, dict) else None

        def _from_summaries() -> Any:
            data = self._connectapi(
                PERSONAL_DAILY_SUMMARIES_URL,
                params={"startDate": today_str, "endDate": today_str},
            )
            rows = (data or {}).get("data") if isinstance(data, dict) else None
            first = rows[0] if isinstance(rows, list) and rows else None
            return first.get("userProfilePk") if isinstance(first, dict) else None

        def _from_activities() -> Any:
            page = self._connectapi(api.ACTIVITIES_URL, params={"start": 0, "limit": 1})
            first = page[0] if isinstance(page, list) and page else None
            return first.get("ownerId") if isinstance(first, dict) else None

        sources = (
            lambda: _from_heart_rate(today_str),
            lambda: _from_heart_rate(yesterday),
            _from_summaries,
            _from_activities,
        )
        last_error: Optional[GarminConnectionError] = None
        for source in sources:
            try:
                user_id = source()
            except GarminConnectionError as e:
                _LOGGER.debug("User ID source failed: %s", e)
                last_error = e
                continue
            if user_id:
                return int(user_id)
        if last_error is not None:
            raise GarminConnectionError(
                f"Could not determine the Garmin user ID from the Personal API: "
                f"{last_error}"
            ) from last_error
        raise GarminAuthenticationError(
            "Could not determine the Garmin user ID from the Personal API"
        )

    # ------------------------------------------------------------------
    # Request helpers (same contract as GarminClient, used by api.py)
    # ------------------------------------------------------------------

    def _request(self, path: str, accept: str, **kwargs: Any) -> requests.Response:
        """
        Issue an authenticated GET, renewing the bearer token as needed.

        :param path: API path.
        :param accept: ``Accept`` header value.
        :param kwargs: Forwarded to ``requests.Session.get`` (e.g. ``params``).
        :return: The successful response.
        :raises GarminAuthenticationError: On HTTP 401 after one token renewal.
        :raises GarminTooManyRequestsError: On HTTP 429 after the rate-limit retries.
        :raises GarminConnectionError: On transport errors or other HTTP errors.
        """
        if self._token_expires_soon():
            self._obtain_bearer_token()

        url = f"{self.base_url}/{path.lstrip('/')}"
        kwargs.setdefault("timeout", self.timeout)

        def _send() -> requests.Response:
            headers = {
                "Authorization": f"Bearer {self._bearer_token}",
                "Accept": accept,
            }
            try:
                return self._session.get(url, headers=headers, **kwargs)
            except requests.RequestException as exc:
                raise GarminConnectionError(f"API request failed: {exc}") from exc

        resp = _send()
        for _ in range(_RATE_LIMIT_MAX_RETRIES):
            if resp.status_code != 429:
                break
            wait_s = _retry_after_seconds(resp)
            _LOGGER.info(
                "Personal API rate limit reached; waiting %.0fs before retrying.",
                wait_s,
            )
            time.sleep(wait_s)
            resp = _send()
        if resp.status_code == 401:
            # Token revoked or expired early: renew once and retry.
            self._obtain_bearer_token()
            resp = _send()
            if resp.status_code == 401:
                raise GarminAuthenticationError(
                    "Personal API request unauthorized after token renewal"
                )
        if resp.status_code == 429:
            raise GarminTooManyRequestsError(
                f"API rate limited (HTTP 429): {resp.text[:200]}"
            )
        if resp.status_code >= 400:
            raise GarminConnectionError(
                f"API Error {resp.status_code} for {path}: {resp.text[:200]}"
            )
        return resp

    def _connectapi(self, path: str, **kwargs: Any) -> Any:
        """
        GET a JSON endpoint.

        :param path: API path.
        :param kwargs: Forwarded to :meth:`_request` (e.g. ``params``).
        :return: Parsed JSON, or an empty dict on an empty (e.g. HTTP 204) response.
        :raises GarminConnectionError: If the body is not valid JSON.
        """
        resp = self._request(path, accept="application/json", **kwargs)
        if resp.status_code == 204 or not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as err:
            preview = " ".join(resp.text.split())[:200]
            raise GarminConnectionError(
                f"Invalid JSON response from Garmin (status {resp.status_code}): "
                f"{preview!r}"
            ) from err

    def _download(self, path: str, **kwargs: Any) -> bytes:
        """
        GET a binary endpoint.

        :param path: API path.
        :param kwargs: Forwarded to :meth:`_request`.
        :return: Raw response bytes.
        """
        kwargs.pop("headers", None)
        return self._request(path, accept="*/*", **kwargs).content

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def get_sleep_data(self, cdate: str) -> Dict[str, Any]:
        """
        Fetch the nightly sleep summary for the given date.

        The Personal API returns the bare sleep DTO, without the per-minute arrays
        (``sleepLevels``, ``sleepMovement``, ...) of the SSO endpoint. It is wrapped
        under ``dailySleepDTO`` so the SLEEP processor parses it unchanged; the sleep
        detail tables stay empty for these nights.

        :param cdate: Date in ``YYYY-MM-DD`` format.
        :return: ``{"dailySleepDTO": {...}}``, or an empty dict when no sleep was
            recorded.
        """
        cdate = _validate_date_format(cdate, "cdate")
        dto = self._connectapi(PERSONAL_SLEEP_URL, params={"date": cdate})
        if not isinstance(dto, dict) or not dto.get("sleepStartTimestampGMT"):
            return {}
        return {"dailySleepDTO": dto}

    def get_heart_rates(self, cdate: str) -> Dict[str, Any]:
        """
        Fetch heart rate data (2-minute time series) for the given date.

        :param cdate: Date in ``YYYY-MM-DD`` format.
        :return: Heart rate dictionary, same shape as the SSO endpoint.
        """
        cdate = _validate_date_format(cdate, "cdate")
        return self._connectapi(PERSONAL_HEART_RATE_URL, params={"date": cdate})

    def get_menstrual_data_for_date(self, cdate: str) -> Optional[Dict[str, Any]]:
        """
        See :func:`.api.get_menstrual_data_for_date`.
        """
        return api.get_menstrual_data_for_date(self, cdate)  # type: ignore[arg-type]

    def get_body_composition(
        self, startdate: str, enddate: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        See :func:`.api.get_body_composition`.
        """
        return api.get_body_composition(  # type: ignore[arg-type]
            self, startdate, enddate
        )

    def get_activities_by_date(
        self,
        startdate: str,
        enddate: Optional[str] = None,
        activitytype: Optional[str] = None,
        sortorder: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch activities between two dates; see :func:`.api.get_activities_by_date`.

        Fills the boolean flags the Personal API omits (``hasPolyline``,
        ``hasHeatMap``, ``purposeful``) with False so the activity processor, which
        requires them, accepts the payload.

        :param startdate: Start date in ``YYYY-MM-DD`` format.
        :param enddate: Optional end date in ``YYYY-MM-DD`` format.
        :param activitytype: Optional activity type filter.
        :param sortorder: Optional sort direction.
        :return: List of activity dictionaries.
        """
        activities = api.get_activities_by_date(  # type: ignore[arg-type]
            self, startdate, enddate, activitytype, sortorder
        )
        for activity in activities:
            if isinstance(activity, dict):
                for flag in _ACTIVITY_FLAGS_DEFAULT_FALSE:
                    activity.setdefault(flag, False)
        return activities

    def get_activity_details(self, activity_id: int) -> Optional[Dict[str, Any]]:
        """
        See :func:`.api.get_activity_details`.
        """
        return api.get_activity_details(self, activity_id)  # type: ignore[arg-type]

    def download_activity(
        self,
        activity_id: Any,
        dl_fmt: ActivityDownloadFormat = ActivityDownloadFormat.ORIGINAL,
    ) -> bytes:
        """
        Download the original activity file (zipped FIT/TCX).

        Only ``ORIGINAL`` is served by the Personal API; the export formats are not.

        :param activity_id: Garmin activity ID.
        :param dl_fmt: Download format; must be ``ORIGINAL``.
        :return: Raw ZIP bytes.
        :raises ValueError: For any format other than ``ORIGINAL``.
        """
        if dl_fmt != ActivityDownloadFormat.ORIGINAL:
            raise ValueError("The Personal API only serves ORIGINAL activity downloads")
        return api.download_activity(  # type: ignore[arg-type]
            self, activity_id, dl_fmt
        )

    # ------------------------------------------------------------------
    # Personal-API-only data methods
    # ------------------------------------------------------------------

    def get_daily_summary(self, cdate: str) -> Dict[str, Any]:
        """
        Fetch the daily wellness summary (steps, stress, body battery, ...).

        :param cdate: Date in ``YYYY-MM-DD`` format.
        :return: Summary dictionary, or an empty dict for days without wellness data
            (Garmin returns an all-null object for those).
        """
        cdate = _validate_date_format(cdate, "cdate")
        summary = self._connectapi(
            PERSONAL_USER_SUMMARY_URL, params={"calendarDate": cdate}
        )
        if not isinstance(summary, dict) or all(
            summary.get(field) is None for field in _DAILY_SUMMARY_SIGNAL_FIELDS
        ):
            return {}
        return summary

    def get_hrv_data(self, cdate: str) -> Dict[str, Any]:
        """
        Fetch the nightly HRV status and overnight readings for the given date.

        :param cdate: Date in ``YYYY-MM-DD`` format (the morning the night ended).
        :return: HRV dictionary with ``hrvSummary`` and ``hrvReadings``, or an empty
            dict when no HRV was recorded (HTTP 204).
        """
        cdate = _validate_date_format(cdate, "cdate")
        data = self._connectapi(f"{PERSONAL_HRV_URL}/{cdate}")
        if not isinstance(data, dict) or not data.get("hrvSummary"):
            return {}
        return data

    def get_body_battery(
        self, startdate: str, enddate: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch daily body battery reports (3-minute levels and impact events).

        Splits ranges longer than 31 days into consecutive windows, since the endpoint
        rejects longer ones.

        :param startdate: Start date in ``YYYY-MM-DD`` format.
        :param enddate: Optional end date in ``YYYY-MM-DD`` format (inclusive).
        :return: One report per day, each with ``date``,
            ``bodyBatteryValuesArray`` (``[timestamp_ms, level]`` pairs) and
            ``bodyBatteryActivityEvent``.
        """
        reports: List[Dict[str, Any]] = []
        for window_start, window_end in _date_windows(
            startdate, enddate, _BODY_BATTERY_MAX_RANGE_DAYS
        ):
            page = self._connectapi(
                PERSONAL_BODY_BATTERY_URL,
                params={
                    "startDate": window_start,
                    "endDate": window_end,
                    "chunks": _BODY_BATTERY_CHUNKS,
                },
            )
            if isinstance(page, list):
                reports.extend(r for r in page if isinstance(r, dict))
        return reports

    def get_fitness_age(
        self, startdate: str, enddate: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch fitness age estimates, sampled weekly plus the last day of the range.

        :param startdate: Start date in ``YYYY-MM-DD`` format.
        :param enddate: Optional end date in ``YYYY-MM-DD`` format (inclusive).
        :return: Estimates with an added ``calendarDate``; days Garmin has no estimate
            for (no ``fitnessAge``) are dropped.
        """
        estimates: List[Dict[str, Any]] = []
        for sample_date in _sample_dates(startdate, enddate, _FITNESS_AGE_SAMPLE_DAYS):
            data = self._connectapi(f"{PERSONAL_FITNESS_AGE_URL}/{sample_date}")
            if isinstance(data, dict) and data.get("fitnessAge") is not None:
                estimates.append({"calendarDate": sample_date, **data})
        return estimates

    def get_heart_rate_zones(self) -> List[Dict[str, Any]]:
        """
        Fetch the current heart rate zone configuration per sport.

        :return: List of zone configurations (one per sport).
        """
        data = self._connectapi(PERSONAL_HR_ZONES_URL)
        return (
            [z for z in data if isinstance(z, dict)] if isinstance(data, list) else []
        )

    def get_activity_workouts(self, activity_id: Any) -> List[Dict[str, Any]]:
        """
        Fetch the structured workouts followed during an activity.

        :param activity_id: Garmin activity ID.
        :return: List of workouts (empty when the activity followed none).
        :raises ValueError: If ``activity_id`` is not a positive integer.
        """
        aid = int(activity_id)
        if aid <= 0:
            raise ValueError(
                f"activity_id must be a positive integer, got {activity_id}"
            )
        data = self._connectapi(f"/activity-service/activity/{aid}/workouts")
        return (
            [w for w in data if isinstance(w, dict)] if isinstance(data, list) else []
        )

    def get_food_logs(self, cdate: str) -> Dict[str, Any]:
        """
        Fetch the food log for a day.

        :param cdate: Date in ``YYYY-MM-DD`` format.
        :return: Food-log dictionary, or an empty dict when nothing was logged.
        """
        cdate = _validate_date_format(cdate, "cdate")
        data = self._connectapi(f"{PERSONAL_FOOD_LOGS_URL}/{cdate}")
        if not isinstance(data, dict) or not (
            data.get("loggedFoodsWithServingSizes") or data.get("mealDetails")
        ):
            return {}
        return data

    def get_menstrual_cycle_summary(self, cdate: str) -> Dict[str, Any]:
        """
        Fetch the menstrual cycle summary covering a date.

        Used to decide whether the per-day menstrual dayview fan-out is worth running:
        accounts that do not track cycles get an empty (HTTP 204) response.

        :param cdate: Date in ``YYYY-MM-DD`` format.
        :return: Summary dictionary, or an empty dict when no cycle covers the date.
        """
        cdate = _validate_date_format(cdate, "cdate")
        data = self._connectapi(f"{PERSONAL_MENSTRUAL_SUMMARY_URL}/{cdate}")
        return data if isinstance(data, dict) else {}

    def get_earliest_activity_date(self) -> Optional[str]:
        """
        Find the local start date of the oldest recorded activity.

        Used as the default start of a full-history sync.

        :return: ``YYYY-MM-DD`` date, or None when the account has no activities.
        """
        page = self._connectapi(
            api.ACTIVITIES_URL,
            params={"start": 0, "limit": 1, "sortOrder": "asc"},
        )
        if isinstance(page, list) and page and isinstance(page[0], dict):
            start_local = page[0].get("startTimeLocal")
            if isinstance(start_local, str) and len(start_local) >= 10:
                return start_local[:10]
        return None


def _date_windows(startdate: str, enddate: Optional[str], max_days: int) -> List[tuple]:
    """
    Split an inclusive date range into consecutive windows of at most ``max_days``.

    :param startdate: Start date in ``YYYY-MM-DD`` format.
    :param enddate: Optional end date (inclusive); defaults to ``startdate``.
    :param max_days: Maximum window length in days.
    :return: List of ``(window_start, window_end)`` ISO date strings.
    :raises ValueError: If a date is malformed or the range is inverted.
    """
    start = date.fromisoformat(_validate_date_format(startdate, "startdate"))
    end = (
        date.fromisoformat(_validate_date_format(enddate, "enddate"))
        if enddate
        else start
    )
    if end < start:
        raise ValueError(f"enddate ({end}) must be on or after startdate ({start})")
    windows = []
    window_start = start
    while window_start <= end:
        window_end = min(window_start + timedelta(days=max_days - 1), end)
        windows.append((window_start.isoformat(), window_end.isoformat()))
        window_start = window_end + timedelta(days=1)
    return windows


def _sample_dates(startdate: str, enddate: Optional[str], step_days: int) -> List[str]:
    """
    Pick every ``step_days``-th date of an inclusive range, always including the end.

    :param startdate: Start date in ``YYYY-MM-DD`` format.
    :param enddate: Optional end date (inclusive); defaults to ``startdate``.
    :param step_days: Sampling interval in days.
    :return: Sorted list of ISO date strings.
    """
    windows = _date_windows(startdate, enddate, step_days)
    samples = [window_start for window_start, _ in windows]
    last = windows[-1][1]
    if last not in samples:
        samples.append(last)
    return samples


def _retry_after_seconds(resp: requests.Response) -> float:
    """
    Read how long to wait after an HTTP 429 from the Personal API.

    :param resp: The 429 response.
    :return: Seconds to wait: ``retryInSeconds`` from the JSON body (or the
        ``Retry-After`` header) plus one second of margin, capped; a default when
        neither is present or parseable.
    """
    wait: Any = None
    try:
        body = resp.json()
        if isinstance(body, dict):
            wait = body.get("retryInSeconds")
    except ValueError:
        pass
    if wait is None:
        wait = (getattr(resp, "headers", None) or {}).get("Retry-After")
    try:
        seconds = float(wait) + 1.0
    except (TypeError, ValueError):
        return _RATE_LIMIT_DEFAULT_WAIT_S
    return min(max(seconds, 1.0), _RATE_LIMIT_MAX_WAIT_S)


def _jwt_expiration(token: str) -> Optional[float]:
    """
    Read the ``exp`` claim of a JWT without verifying its signature.

    :param token: JWT string.
    :return: Expiration as seconds since the epoch, or None if unparseable.
    """
    try:
        payload_b64 = token.split(".")[1]
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4))
        )
        return float(payload["exp"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None
