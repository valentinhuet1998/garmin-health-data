"""
Tests for the Personal API sync data types: client methods, processing into the
database, extractor plumbing, and the ``garmin sync`` command.

Payload shapes mirror real Personal API responses (trimmed). No test touches the
network.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from sqlalchemy import select
from sqlalchemy.orm import Session

from garmin_health_data.cli import cli
from garmin_health_data.constants import GARMIN_FILE_TYPES
from garmin_health_data.extractor import (
    GarminExtractor,
    _split_body_battery_by_day,
    _split_by_calendar_date,
)
from garmin_health_data.garmin_client import HybridClient, PersonalApiClient
from garmin_health_data.garmin_client.hybrid import missing_auth_hint
from garmin_health_data.garmin_client.personal import _date_windows, _sample_dates
from garmin_health_data.models import (
    HRV,
    ActivityWorkout,
    BodyBattery,
    BodyBatteryEvent,
    DailySummary,
    FitnessAge,
    HeartRateZone,
    HrvSummary,
    NutritionLog,
    Sleep,
)
from garmin_health_data.processor import GarminProcessor
from garmin_health_data.processor_helpers import FileSet
from tests.test_processor import _seed_activity

USER_ID = 1
STAMP = "2026-11-03T12-00-00Z"

# Night of 2026-11-02 -> 2026-11-03, 21:25:43Z to 05:43:43Z.
SLEEP_START_MS = 1793654743000
SLEEP_END_MS = 1793684623000

SLEEP_PAYLOAD = {
    "dailySleepDTO": {
        "calendarDate": "2026-11-03",
        "sleepStartTimestampGMT": SLEEP_START_MS,
        "sleepEndTimestampGMT": SLEEP_END_MS,
        "sleepStartTimestampLocal": SLEEP_START_MS + 3600000,
        "sleepEndTimestampLocal": SLEEP_END_MS + 3600000,
        "sleepTimeSeconds": 29760,
        "sleepScores": {"overall": {"value": 84, "qualifierKey": "GOOD"}},
    }
}

HRV_PAYLOAD = {
    "hrvSummary": {
        "calendarDate": "2026-11-03",
        "weeklyAvg": 89,
        "lastNightAvg": 85,
        "lastNight5MinHigh": 129,
        "baseline": {
            "lowUpper": 79,
            "balancedLow": 85,
            "balancedUpper": 115,
            "markerValue": 0.31666565,
        },
        "status": "BALANCED",
        "feedbackPhrase": "HRV_BALANCED_3",
    },
    "hrvReadings": [
        {"hrvValue": 54, "readingTimeGMT": "2026-11-02T21:29:14.0"},
        {"hrvValue": 61, "readingTimeGMT": "2026-11-02T21:34:14.0"},
    ],
    "sleepStartTimestampGMT": "2026-11-02T21:25:43.0",
    "sleepEndTimestampGMT": "2026-11-03T05:43:43.0",
}

DAILY_SUMMARY_PAYLOAD = {
    "calendarDate": "2026-11-03",
    "totalSteps": 14502,
    "dailyStepGoal": 12450,
    "totalDistanceMeters": 15224,
    "floorsAscended": 30.68438,
    "restingHeartRate": 46,
    "averageStressLevel": 38,
    "stressQualifier": "BALANCED",
    "highStressDuration": 10260,
    "bodyBatteryAtWakeTime": 75,
    "bodyBatteryHighestValue": 75,
    "bodyBatteryLowestValue": 8,
    "vigorousIntensityMinutes": 69,
    "averageSpo2": 96.0,
    "avgWakingRespirationValue": 15.0,
    "includesWellnessData": True,
    "bodyBatteryDynamicFeedbackEvent": {
        "feedbackShortType": "SLEEP_PREPARATION_STRESSFUL_AND_INTENSIVE_EXERCISE",
        "feedbackLongType": "LONG_TYPE",
    },
    "endOfDayBodyBatteryDynamicFeedbackEvent": {"feedbackShortType": "EOD_SHORT"},
    "someFieldNotStored": "ignored",
}

BODY_BATTERY_PAYLOAD = [
    {
        "date": "2026-11-03",
        "bodyBatteryValuesArray": [
            [1793660400000, 14],
            [1793660580000, 15],
            [1793660760000, None],
        ],
        "bodyBatteryActivityEvent": [
            {
                "eventType": "SLEEP",
                "eventStartTimeGmt": "2026-11-02T21:25:43.0",
                "timezoneOffset": 3600000,
                "durationInMilliseconds": 29880000,
                "bodyBatteryImpact": 67,
                "feedbackType": "NONE",
                "shortFeedback": "NONE",
            }
        ],
    }
]

FITNESS_AGE_PAYLOAD = [
    {
        "calendarDate": "2026-11-03",
        "chronologicalAge": 28,
        "fitnessAge": 18.0,
        "previousFitnessAge": 18.0,
        "components": {
            "bmi": {"value": 21.0},
            "rhr": {"value": 44},
            "vigorousDaysAvg": {"value": 6.3},
            "vigorousMinutesAvg": {"value": 484.5},
        },
        "lastUpdated": "2026-11-03T00:00:00.0",
    }
]

HR_ZONES_PAYLOAD = [
    {
        "sport": "DEFAULT",
        "trainingMethod": "LACTATE_THRESHOLD",
        "lactateThresholdHeartRateUsed": 175,
        "maxHeartRateUsed": 189,
        "restingHeartRateUsed": 45,
        "restingHrAutoUpdateUsed": False,
        "zone1Floor": 95,
        "zone2Floor": 113,
        "zone3Floor": 132,
        "zone4Floor": 151,
        "zone5Floor": 170,
    },
    {"sport": "RUNNING", "zone1Floor": 100},
]

NUTRITION_PAYLOAD = {
    "mealDate": "2026-11-03",
    "loggedFoodsWithServingSizes": [{"foodName": "Banana"}],
    "mealDetails": [],
}

WORKOUT_STEPS = [
    {
        "stepIndex": 0,
        "intensity": "WARMUP",
        "durationType": "TIME",
        "durationValue": 900.0,
        "targetType": "POWER_3S",
        "targetValueLow": 0.0,
        "targetValueHigh": 127.0,
    }
]


def _write(tmp_path: Path, data_type: str, payload: Any, extra: str = "") -> Path:
    """
    Write a payload as an extractor-style JSON file.

    :param tmp_path: Directory to write into.
    :param data_type: Registry data type name.
    :param payload: JSON-serializable payload.
    :param extra: Optional ``_<id>`` infix (per-activity files).
    :return: Path to the written file.
    """
    path = tmp_path / f"{USER_ID}_{data_type}{extra}_{STAMP}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _process(session: Session, paths: List[Path]) -> None:
    """
    Run a FileSet through GarminProcessor and commit.

    :param session: Database session.
    :param paths: Files of one (user, day) FileSet.
    """
    files: Dict[Any, List[Path]] = {}
    for path in paths:
        enum = next(e for e in GARMIN_FILE_TYPES if e.value.match(path.name))
        files.setdefault(enum, []).append(path)
    file_set = FileSet(file_paths=paths, files=files)
    GarminProcessor(file_set, session).process_file_set(file_set, session)
    session.commit()


class TestProcessing:
    """
    End-to-end processing of Personal API files into a real SQLite schema.
    """

    def test_full_day_file_set(self, db_session: Session, tmp_path: Path) -> None:
        """
        A day with every Personal API file type lands in the expected tables.

        :param db_session: Database session over the production DDL.
        :param tmp_path: Pytest temporary directory fixture.
        """
        _process(
            db_session,
            [
                # HRV listed before SLEEP on purpose: processing order must still put
                # SLEEP first so the readings find their sleep session.
                _write(tmp_path, "HRV", HRV_PAYLOAD),
                _write(tmp_path, "SLEEP", SLEEP_PAYLOAD),
                _write(tmp_path, "DAILY_SUMMARY", DAILY_SUMMARY_PAYLOAD),
                _write(tmp_path, "BODY_BATTERY", BODY_BATTERY_PAYLOAD),
                _write(tmp_path, "FITNESS_AGE", FITNESS_AGE_PAYLOAD),
                _write(tmp_path, "HR_ZONES", HR_ZONES_PAYLOAD),
                _write(tmp_path, "NUTRITION", NUTRITION_PAYLOAD),
            ],
        )

        summary = db_session.execute(select(DailySummary)).scalar_one()
        assert summary.date == date(2026, 11, 3)
        assert summary.total_steps == 14502
        assert summary.body_battery_at_wake_time == 75
        assert summary.stress_qualifier == "BALANCED"
        assert summary.average_spo2 == 96.0
        assert summary.body_battery_feedback_short.startswith("SLEEP_PREPARATION")
        assert summary.body_battery_end_of_day_feedback_short == "EOD_SHORT"
        assert summary.min_heart_rate is None  # Absent from the payload.

        hrv_summary = db_session.execute(select(HrvSummary)).scalar_one()
        assert (hrv_summary.last_night_avg, hrv_summary.weekly_avg) == (85, 89)
        assert hrv_summary.last_night_5min_high == 129
        assert hrv_summary.baseline_balanced_upper == 115
        assert hrv_summary.status == "BALANCED"

        sleep = db_session.execute(select(Sleep)).scalar_one()
        assert sleep.avg_overnight_hrv == 85
        assert sleep.hrv_status == "BALANCED"
        readings = db_session.execute(select(HRV)).scalars().all()
        assert sorted(r.value for r in readings) == [54, 61]
        assert {r.sleep_id for r in readings} == {sleep.sleep_id}

        levels = db_session.execute(select(BodyBattery)).scalars().all()
        assert sorted(level.value for level in levels) == [14, 15]  # Null skipped.
        event = db_session.execute(select(BodyBatteryEvent)).scalar_one()
        assert (event.event_type, event.impact) == ("SLEEP", 67)

        fitness = db_session.execute(select(FitnessAge)).scalar_one()
        assert (fitness.fitness_age, fitness.resting_heart_rate) == (18.0, 44)
        assert fitness.vigorous_minutes_avg == 484.5

        zones = db_session.execute(
            select(HeartRateZone).order_by(HeartRateZone.sport)
        ).scalars()
        assert [(z.sport, z.zone1_floor) for z in zones] == [
            ("DEFAULT", 95),
            ("RUNNING", 100),
        ]

        nutrition = db_session.execute(select(NutritionLog)).scalar_one()
        assert (nutrition.food_count, nutrition.meal_count) == (1, 0)
        assert nutrition.log_json["loggedFoodsWithServingSizes"][0]["foodName"] == (
            "Banana"
        )

    def test_reprocessing_is_idempotent(
        self, db_session: Session, tmp_path: Path
    ) -> None:
        """
        Processing the same day twice updates rows instead of duplicating them.

        :param db_session: Database session over the production DDL.
        :param tmp_path: Pytest temporary directory fixture.
        """
        paths = [
            _write(tmp_path, "SLEEP", SLEEP_PAYLOAD),
            _write(tmp_path, "HRV", HRV_PAYLOAD),
            _write(tmp_path, "DAILY_SUMMARY", DAILY_SUMMARY_PAYLOAD),
            _write(tmp_path, "BODY_BATTERY", BODY_BATTERY_PAYLOAD),
        ]
        _process(db_session, paths)

        updated = dict(DAILY_SUMMARY_PAYLOAD, totalSteps=15000)
        paths[2] = _write(tmp_path, "DAILY_SUMMARY", updated)
        _process(db_session, paths)

        for model, expected in (
            (DailySummary, 1),
            (HrvSummary, 1),
            (HRV, 2),
            (BodyBattery, 2),
            (BodyBatteryEvent, 1),
        ):
            assert len(db_session.execute(select(model)).scalars().all()) == expected
        summary = db_session.execute(select(DailySummary)).scalar_one()
        assert summary.total_steps == 15000

    def test_hrv_without_sleep_keeps_summary_only(
        self, db_session: Session, tmp_path: Path
    ) -> None:
        """
        Without a matching sleep row the summary is stored, the readings are not.

        :param db_session: Database session over the production DDL.
        :param tmp_path: Pytest temporary directory fixture.
        """
        _process(db_session, [_write(tmp_path, "HRV", HRV_PAYLOAD)])

        assert db_session.execute(select(HrvSummary)).scalar_one().status == "BALANCED"
        assert db_session.execute(select(HRV)).scalars().all() == []

    def test_workouts_linked_to_activity(
        self, db_session: Session, tmp_path: Path
    ) -> None:
        """
        Workouts replace any previous rows of their activity.

        :param db_session: Database session over the production DDL.
        :param tmp_path: Pytest temporary directory fixture.
        """
        _seed_activity(db_session, activity_id=555)
        db_session.commit()
        first = {
            "activityId": 555,
            "workouts": [
                {"index": 0, "workoutName": "Old", "steps": WORKOUT_STEPS * 2},
                {"index": 1, "workoutName": "Extra", "steps": []},
            ],
        }
        _process(db_session, [_write(tmp_path, "ACTIVITY_WORKOUT", first, "_555")])
        second = {
            "activityId": 555,
            "workouts": [
                {
                    "index": 0,
                    "workoutName": "RT 02/11",
                    "sport": "CYCLING",
                    "steps": WORKOUT_STEPS,
                }
            ],
        }
        _process(db_session, [_write(tmp_path, "ACTIVITY_WORKOUT", second, "_555")])

        workout = db_session.execute(select(ActivityWorkout)).scalar_one()
        assert (workout.workout_name, workout.sport) == ("RT 02/11", "CYCLING")
        assert workout.step_count == 1
        assert workout.steps_json[0]["intensity"] == "WARMUP"

    def test_workouts_for_unknown_activity_skipped(
        self, db_session: Session, tmp_path: Path
    ) -> None:
        """
        Workouts of an activity missing from the database are skipped, not failed.

        :param db_session: Database session over the production DDL.
        :param tmp_path: Pytest temporary directory fixture.
        """
        payload = {"activityId": 999, "workouts": [{"index": 0, "steps": []}]}
        _process(db_session, [_write(tmp_path, "ACTIVITY_WORKOUT", payload, "_999")])

        assert db_session.execute(select(ActivityWorkout)).scalars().all() == []


class TestClientMethods:
    """
    Tests for the Personal-API-only client methods (HTTP layer mocked).
    """

    @staticmethod
    def _client(responses: Dict[str, Any]) -> PersonalApiClient:
        """
        Build a client whose ``_connectapi`` answers from a path-prefix table.

        :param responses: Mapping of path prefix to response (or callable).
        :return: PersonalApiClient with a stubbed ``_connectapi``.
        """
        client = PersonalApiClient("k")
        calls: List[tuple] = []

        def fake(path: str, params: Any = None, **_: Any) -> Any:
            calls.append((path, params))
            for prefix, response in responses.items():
                if path.startswith(prefix):
                    return response(path, params) if callable(response) else response
            return {}

        client._connectapi = fake  # type: ignore[method-assign]
        client.calls = calls  # type: ignore[attr-defined]
        return client

    def test_daily_summary_all_null_day_is_empty(self) -> None:
        """
        Days without wellness data (all signal fields null) return an empty dict.
        """
        client = self._client(
            {"/usersummary-service": {"calendarDate": "2020-01-15", "totalSteps": None}}
        )
        assert client.get_daily_summary("2020-01-15") == {}

    def test_daily_summary_with_data(self) -> None:
        """
        A day with data is returned as-is and requested by calendarDate.
        """
        client = self._client({"/usersummary-service": DAILY_SUMMARY_PAYLOAD})
        assert client.get_daily_summary("2026-11-03")["totalSteps"] == 14502
        assert client.calls[0][1] == {"calendarDate": "2026-11-03"}

    def test_hrv_no_content_is_empty(self) -> None:
        """
        HTTP 204 (no HRV that night) returns an empty dict.
        """
        assert self._client({}).get_hrv_data("2024-06-15") == {}

    def test_body_battery_split_into_31_day_windows(self) -> None:
        """
        A 40-day range is requested as two windows at 3-minute resolution.
        """
        client = self._client(
            {"/wellness-service": lambda path, params: [{"date": params["startDate"]}]}
        )
        reports = client.get_body_battery("2026-01-01", "2026-02-09")

        windows = [(p["startDate"], p["endDate"]) for _, p in client.calls]
        assert windows == [
            ("2026-01-01", "2026-01-31"),
            ("2026-02-01", "2026-02-09"),
        ]
        assert all(p["chunks"] == 288 for _, p in client.calls)
        assert [r["date"] for r in reports] == ["2026-01-01", "2026-02-01"]

    def test_fitness_age_sampled_weekly_plus_end(self) -> None:
        """
        Fitness age is sampled every 7 days plus the end date; empty days dropped.
        """

        def answer(path: str, params: Any) -> Dict[str, Any]:
            day = path.rsplit("/", 1)[1]
            return {} if day == "2026-01-08" else {"fitnessAge": 30.0}

        client = self._client({"/fitnessage-service": answer})
        estimates = client.get_fitness_age("2026-01-01", "2026-01-17")

        assert [p.rsplit("/", 1)[1] for p, _ in client.calls] == [
            "2026-01-01",
            "2026-01-08",
            "2026-01-15",
            "2026-01-17",
        ]
        assert [e["calendarDate"] for e in estimates] == [
            "2026-01-01",
            "2026-01-15",
            "2026-01-17",
        ]

    def test_food_logs_empty_day(self) -> None:
        """
        A day with no logged food or meal returns an empty dict.
        """
        client = self._client(
            {
                "/nutrition-service": {
                    "mealDate": "2026-11-03",
                    "loggedFoodsWithServingSizes": [],
                    "mealDetails": [],
                }
            }
        )
        assert client.get_food_logs("2026-11-03") == {}

    def test_zones_and_workouts_filter_non_dicts(self) -> None:
        """
        List endpoints drop malformed entries and tolerate non-list responses.
        """
        client = self._client(
            {"/biometric-service": [{"sport": "DEFAULT"}, "junk"], "/activity": {}}
        )
        assert client.get_heart_rate_zones() == [{"sport": "DEFAULT"}]
        assert client.get_activity_workouts(5) == []
        with pytest.raises(ValueError):
            client.get_activity_workouts(0)

    def test_earliest_activity_date(self) -> None:
        """
        The oldest activity is looked up with sortOrder=asc.
        """
        client = self._client(
            {"/activitylist": [{"startTimeLocal": "2026-03-13 06:46:10"}]}
        )
        assert client.get_earliest_activity_date() == "2026-03-13"
        assert client.calls[0][1]["sortOrder"] == "asc"
        assert self._client({"/activitylist": []}).get_earliest_activity_date() is None

    def test_date_helpers(self) -> None:
        """
        Window splitting and weekly sampling cover the range exactly.
        """
        assert _date_windows("2026-01-01", "2026-01-01", 31) == [
            ("2026-01-01", "2026-01-01")
        ]
        assert _sample_dates("2026-01-01", "2026-01-07", 7) == [
            "2026-01-01",
            "2026-01-07",
        ]
        with pytest.raises(ValueError):
            _date_windows("2026-01-02", "2026-01-01", 31)


class TestHybridRouting:
    """
    Tests for routing the Personal-API-only methods.
    """

    def test_sso_without_method_is_unsupported(self) -> None:
        """
        An SSO-only client cannot serve the Personal-API-only methods.
        """

        class FakeSso:
            """
            SSO stand-in that only knows heart rates.
            """

            full_name = None

            def get_heart_rates(self, cdate: str) -> str:
                """
                :param cdate: Date.
                :return: Marker.
                """
                return "sso"

        hybrid = HybridClient(sso=FakeSso())  # type: ignore[arg-type]

        assert hybrid.supports("get_heart_rates")
        assert not hybrid.supports("get_daily_summary")
        assert "garmin auth'" in missing_auth_hint("get_daily_summary")
        assert "--sso" in missing_auth_hint("get_training_status")


class TestExtractorPlumbing:
    """
    Tests for the extractor side of the new data types.
    """

    def test_body_battery_splitter_drops_empty_days(self) -> None:
        """
        Days without levels or events produce no file.
        """
        payload = [
            {"date": "2026-11-01", "bodyBatteryValuesArray": [[1, None]]},
            {"date": "2026-11-02", "bodyBatteryValuesArray": [[1, 40]]},
            {"date": "2026-11-03", "bodyBatteryActivityEvent": [{"eventType": "X"}]},
            "junk",
        ]
        assert sorted(_split_body_battery_by_day(payload)) == [
            date(2026, 11, 2),
            date(2026, 11, 3),
        ]

    def test_calendar_date_splitter(self) -> None:
        """
        Rows group by their calendarDate.
        """
        split = _split_by_calendar_date(
            [{"calendarDate": "2026-11-01"}, {"calendarDate": "2026-11-08"}]
        )
        assert sorted(split) == [date(2026, 11, 1), date(2026, 11, 8)]

    def test_workout_file_saved(self, tmp_path: Path) -> None:
        """
        Activities following a structured workout get an ACTIVITY_WORKOUT file.

        :param tmp_path: Pytest temporary directory fixture.
        """
        extractor = GarminExtractor(
            date(2026, 11, 3),
            date(2026, 11, 3),
            tmp_path,
            data_types=["ACTIVITY_WORKOUT"],
        )
        client = MagicMock()
        client.supports.return_value = True
        client.get_activities_by_date.return_value = [
            {
                "activityId": 7,
                "startTimeLocal": "2026-11-03T18:00:00.0",
                "activityType": {"typeKey": "cycling"},
            },
            {
                "activityId": 8,
                "startTimeLocal": "2026-11-03T19:00:00.0",
                "activityType": {"typeKey": "running"},
            },
        ]
        client.get_activity_workouts.side_effect = lambda aid: (
            [{"workoutName": "Sprint", "steps": []}] if aid == 7 else []
        )
        extractor.garmin_client = client
        extractor.user_id = USER_ID

        files = extractor.extract_fit_activities()

        assert [f.name for f in files] == [f"{USER_ID}_ACTIVITY_WORKOUT_7_{STAMP}.json"]
        saved = json.loads(files[0].read_text())
        assert saved["activityId"] == 7
        client.download_activity.assert_not_called()  # ACTIVITY not requested.

    @pytest.mark.parametrize("summary, expected", [({}, False), ({"x": 1}, True)])
    def test_menstrual_probe_via_personal_summary(
        self, tmp_path: Path, summary: Dict[str, Any], expected: bool
    ) -> None:
        """
        Without the SSO calendar, the Personal API cycle summary gates the fan-out.

        :param tmp_path: Pytest temporary directory fixture.
        :param summary: Cycle summary response for the window end.
        :param expected: Whether the dayview fan-out should run.
        """
        extractor = GarminExtractor(date(2026, 1, 1), date(2026, 1, 1), tmp_path)
        client = MagicMock()
        client.supports.side_effect = lambda name: name != "get_menstrual_calendar_data"
        client.get_menstrual_cycle_summary.return_value = summary
        extractor.garmin_client = client

        result = extractor._menstrual_days_have_data(
            date(2025, 10, 1), date(2026, 1, 1)
        )

        assert result is expected
        client.get_menstrual_cycle_summary.assert_called_once_with("2026-01-01")


class TestSyncCommand:
    """
    Tests for the ``garmin sync`` date window logic.
    """

    @staticmethod
    def _invoke(args: List[str], **patches: Any) -> MagicMock:
        """
        Run ``garmin sync`` with the extract command stubbed out.

        :param args: CLI arguments after ``sync``.
        :param patches: Extra ``garmin_health_data.cli`` attributes to patch.
        :return: The stub standing in for the extract command.
        """
        fake_extract = MagicMock()
        with (
            patch("garmin_health_data.cli.extract", fake_extract),
            patch("garmin_health_data.cli.check_for_newer_version"),
        ):
            for name, value in patches.items():
                patcher = patch(f"garmin_health_data.cli.{name}", value)
                patcher.start()
            try:
                result = CliRunner().invoke(cli, ["sync", *args])
            finally:
                patch.stopall()
        assert result.exit_code == 0, result.output
        return fake_extract

    @staticmethod
    def _window(fake_extract: MagicMock) -> tuple:
        """
        :param fake_extract: Stubbed extract command.
        :return: ``(start, end)`` dates passed to extract.
        """
        kwargs = fake_extract.call_args.kwargs
        return kwargs["start_date"].date(), kwargs["end_date"].date()

    def test_empty_database_syncs_last_30_days_through_today(
        self, tmp_path: Path
    ) -> None:
        """
        A new database syncs the last 30 days, today included.

        :param tmp_path: Pytest temporary directory fixture.
        """
        fake = self._invoke(["--db-path", str(tmp_path / "new.db")])

        today = datetime.now().date()
        assert self._window(fake) == (
            today - timedelta(days=30),
            today + timedelta(days=1),
        )
        assert fake.call_args.kwargs["data_types"] == ()

    def test_incremental_sync_overlaps(self, tmp_path: Path) -> None:
        """
        An existing database restarts --overlap-days before the last stored date.

        :param tmp_path: Pytest temporary directory fixture.
        """
        db_path = tmp_path / "db.sqlite"
        db_path.write_text("")
        fake = self._invoke(
            ["--db-path", str(db_path), "--overlap-days", "3"],
            get_latest_date=MagicMock(return_value=date(2026, 11, 3)),
        )
        assert self._window(fake)[0] == date(2026, 10, 31)

    def test_incremental_sync_starts_from_most_behind_account(
        self, tmp_path: Path
    ) -> None:
        """
        With --accounts, the window starts from the account that is furthest behind.

        :param tmp_path: Pytest temporary directory fixture.
        """
        db_path = tmp_path / "db.sqlite"
        db_path.write_text("")
        latest = {"1": date(2026, 11, 3), "2": date(2026, 10, 20)}
        fake = self._invoke(
            ["--db-path", str(db_path), "--accounts", "1,2", "--overlap-days", "2"],
            get_latest_date=MagicMock(side_effect=lambda _db, ids=None: latest[ids[0]]),
        )
        assert self._window(fake)[0] == date(2026, 10, 18)

    def test_incremental_sync_account_without_data_uses_30_days(
        self, tmp_path: Path
    ) -> None:
        """
        An account with no stored data falls back to the 30-day window.

        :param tmp_path: Pytest temporary directory fixture.
        """
        db_path = tmp_path / "db.sqlite"
        db_path.write_text("")
        fake = self._invoke(
            ["--db-path", str(db_path), "--accounts", "1"],
            get_latest_date=MagicMock(return_value=None),
        )
        assert self._window(fake)[0] == datetime.now().date() - timedelta(days=30)

    def test_full_sync_starts_at_oldest_activity(self, tmp_path: Path) -> None:
        """
        --full starts at the oldest recorded activity.

        :param tmp_path: Pytest temporary directory fixture.
        """
        fake = self._invoke(
            ["--full", "--db-path", str(tmp_path / "db.sqlite"), "--accounts", "1,2"],
            ensure_authenticated=MagicMock(),
            find_earliest_activity_date=MagicMock(return_value=date(2026, 3, 13)),
        )
        assert self._window(fake)[0] == date(2026, 3, 13)
        assert fake.call_args.kwargs["accounts"] == ("1", "2")

    def test_full_sync_without_history_aborts(self, tmp_path: Path) -> None:
        """
        --full without any discoverable history asks for --start-date.

        :param tmp_path: Pytest temporary directory fixture.
        """
        with (
            patch("garmin_health_data.cli.ensure_authenticated"),
            patch(
                "garmin_health_data.cli.find_earliest_activity_date", return_value=None
            ),
            patch("garmin_health_data.cli.check_for_newer_version"),
        ):
            result = CliRunner().invoke(
                cli, ["sync", "--full", "--db-path", str(tmp_path / "db.sqlite")]
            )
        assert result.exit_code != 0
        assert "--start-date" in result.output

    def test_explicit_start_date_wins(self, tmp_path: Path) -> None:
        """
        --start-date overrides both --full and the automatic start.

        :param tmp_path: Pytest temporary directory fixture.
        """
        fake = self._invoke(
            ["--full", "--start-date", "2026-01-05", "--db-path", str(tmp_path / "d")]
        )
        assert self._window(fake)[0] == date(2026, 1, 5)


def test_new_tables_registered_in_schema(db_session: Session) -> None:
    """
    The production DDL creates every Personal API table with its CHECK constraints.

    :param db_session: Database session over the production DDL.
    """
    rows = db_session.connection().exec_driver_sql(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
    )
    ddl = {name: sql for name, sql in rows}
    for table in (
        "daily_summary",
        "hrv_summary",
        "body_battery_event",
        "fitness_age",
        "heart_rate_zone",
        "activity_workout",
        "nutrition_log",
    ):
        assert table in ddl, table
    assert "JSON_VALID(steps_json)" in ddl["activity_workout"]
