"""
Tests for Personal API key authentication: key sources, storage, CLI routing, and how
the extractor combines a key with SSO tokens.
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from garmin_health_data import auth as auth_module
from garmin_health_data.auth import (
    API_KEY_ENV_VAR,
    API_KEY_FILENAME,
    check_authentication,
    has_sso_tokens,
    load_account_api_key,
    read_env_api_key,
    save_api_key,
)
from garmin_health_data.cli import auth
from garmin_health_data.extractor import GarminExtractor, extract
from garmin_health_data.garmin_client import HybridClient


class TestReadEnvApiKey:
    """
    Tests for reading the key from the environment and ``.env``.
    """

    def test_none_when_unset(self) -> None:
        """
        No env var and no ``.env`` file yields None (autouse isolation fixture).
        """
        assert read_env_api_key() is None

    def test_env_var_wins_over_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """
        The environment variable takes precedence over the ``.env`` file.

        :param monkeypatch: Pytest monkeypatch fixture.
        :param tmp_path: Pytest temporary directory fixture.
        """
        dotenv = tmp_path / ".env"
        dotenv.write_text(f"{API_KEY_ENV_VAR}=from-file\n")
        monkeypatch.setenv(API_KEY_ENV_VAR, " from-env ")

        assert read_env_api_key(dotenv) == "from-env"

    @pytest.mark.parametrize(
        "content",
        [
            f"{API_KEY_ENV_VAR}=abc123",
            f'{API_KEY_ENV_VAR}="abc123"',
            f"{API_KEY_ENV_VAR}='abc123'\n",
            f"# comment\n\nOTHER=x\nexport {API_KEY_ENV_VAR} = abc123\n",
            f"{API_KEY_ENV_VAR}=abc123\r\n",
        ],
    )
    def test_dotenv_formats(self, tmp_path: Path, content: str) -> None:
        """
        Common ``.env`` syntaxes are parsed.

        :param tmp_path: Pytest temporary directory fixture.
        :param content: ``.env`` file content.
        """
        dotenv = tmp_path / ".env"
        dotenv.write_text(content)

        assert read_env_api_key(dotenv) == "abc123"

    def test_empty_dotenv_value_is_none(self, tmp_path: Path) -> None:
        """
        An empty value is treated as unset.

        :param tmp_path: Pytest temporary directory fixture.
        """
        dotenv = tmp_path / ".env"
        dotenv.write_text(f"{API_KEY_ENV_VAR}=\n")

        assert read_env_api_key(dotenv) is None

    def test_default_dotenv_path_used(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """
        Without an explicit path, ``DEFAULT_DOTENV_PATH`` is read.

        :param monkeypatch: Pytest monkeypatch fixture.
        :param tmp_path: Pytest temporary directory fixture.
        """
        dotenv = tmp_path / ".env"
        dotenv.write_text(f"{API_KEY_ENV_VAR}=k\n")
        monkeypatch.setattr(auth_module, "DEFAULT_DOTENV_PATH", dotenv)

        assert read_env_api_key() == "k"
        assert check_authentication(str(tmp_path / "no-tokens")) is True


class TestKeyStorage:
    """
    Tests for saving and loading per-account API keys.
    """

    @patch("garmin_health_data.auth.PersonalApiClient")
    def test_save_api_key_writes_account_file(
        self, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        """
        A validated key is saved under ``<base>/<user_id>/personal_api_key``.

        :param mock_client_cls: Mock PersonalApiClient class.
        :param tmp_path: Pytest temporary directory fixture.
        """
        mock_client_cls.return_value.user_id = 12345

        user_id = save_api_key(" my-key ", base_token_dir=str(tmp_path), silent=True)

        assert user_id == "12345"
        key_file = tmp_path / "12345" / API_KEY_FILENAME
        assert key_file.read_text() == "my-key\n"
        assert load_account_api_key(tmp_path / "12345") == "my-key"
        assert not list((tmp_path / "12345").glob(".*.tmp.*"))
        if sys.platform != "win32":
            assert key_file.stat().st_mode & 0o777 == 0o600

    @patch("garmin_health_data.auth.PersonalApiClient")
    def test_save_api_key_rejected(
        self, mock_client_cls: MagicMock, tmp_path: Path
    ) -> None:
        """
        A rejected key raises a ClickException and writes nothing.

        :param mock_client_cls: Mock PersonalApiClient class.
        :param tmp_path: Pytest temporary directory fixture.
        """
        mock_client_cls.return_value.authenticate.side_effect = RuntimeError("401")

        with pytest.raises(click.ClickException, match="validation failed"):
            save_api_key("bad", base_token_dir=str(tmp_path), silent=True)
        assert not any(tmp_path.iterdir())

    def test_load_missing_key_is_none(self, tmp_path: Path) -> None:
        """
        No key file yields None.

        :param tmp_path: Pytest temporary directory fixture.
        """
        assert load_account_api_key(tmp_path) is None

    def test_has_sso_tokens(self, tmp_path: Path) -> None:
        """
        SSO tokens are detected by ``garmin_tokens.json``.

        :param tmp_path: Pytest temporary directory fixture.
        """
        assert has_sso_tokens(tmp_path) is False
        (tmp_path / "garmin_tokens.json").write_text("{}")
        assert has_sso_tokens(tmp_path) is True

    @patch("garmin_health_data.auth.save_api_key", return_value="1")
    @patch("click.prompt", return_value=" typed-key ")
    def test_api_key_auth_prompts_when_no_source(
        self, mock_prompt: MagicMock, mock_save: MagicMock
    ) -> None:
        """
        Without an explicit or env key, the key is prompted with hidden input.

        :param mock_prompt: Mock click.prompt.
        :param mock_save: Mock save_api_key.
        """
        assert auth_module.api_key_auth(silent=True) == "1"
        assert mock_prompt.call_args.kwargs["hide_input"] is True
        assert mock_save.call_args.args[0] == "typed-key"


class TestCliAuthRouting:
    """
    Tests for ``garmin auth`` choosing between API key and SSO login.
    """

    @patch("garmin_health_data.cli.refresh_tokens")
    @patch("garmin_health_data.cli.api_key_auth")
    def test_default_is_api_key(
        self, mock_api_key_auth: MagicMock, mock_refresh: MagicMock
    ) -> None:
        """
        Plain ``garmin auth`` sets up a Personal API key.

        :param mock_api_key_auth: Mock api_key_auth.
        :param mock_refresh: Mock refresh_tokens.
        """
        result = CliRunner().invoke(auth, [], env={"GARMIN_EMAIL": ""})

        assert result.exit_code == 0, result.output
        mock_api_key_auth.assert_called_once_with(None)
        mock_refresh.assert_not_called()

    @patch("garmin_health_data.cli.api_key_auth")
    def test_api_key_option(self, mock_api_key_auth: MagicMock) -> None:
        """
        ``--api-key`` passes the key through.

        :param mock_api_key_auth: Mock api_key_auth.
        """
        result = CliRunner().invoke(auth, ["--api-key", "k"])

        assert result.exit_code == 0, result.output
        mock_api_key_auth.assert_called_once_with("k")

    @patch("garmin_health_data.cli.api_key_auth")
    @patch("garmin_health_data.cli.refresh_tokens")
    @patch("garmin_health_data.cli.get_credentials", return_value=("a@b.co", "pw"))
    def test_sso_flag_uses_email_password(
        self,
        mock_credentials: MagicMock,
        mock_refresh: MagicMock,
        mock_api_key_auth: MagicMock,
    ) -> None:
        """
        ``--sso`` runs the email/password login.

        :param mock_credentials: Mock get_credentials.
        :param mock_refresh: Mock refresh_tokens.
        :param mock_api_key_auth: Mock api_key_auth.
        """
        result = CliRunner().invoke(auth, ["--sso"])

        assert result.exit_code == 0, result.output
        mock_refresh.assert_called_once_with("a@b.co", "pw")
        mock_api_key_auth.assert_not_called()


class TestExtractorAuthentication:
    """
    Tests for ``GarminExtractor.authenticate`` combining key and SSO tokens.
    """

    @staticmethod
    def _extractor(tmp_path: Path) -> GarminExtractor:
        """
        :param tmp_path: Pytest temporary directory fixture.
        :return: Extractor over a one-day window.
        """
        return GarminExtractor(date(2026, 1, 1), date(2026, 1, 1), tmp_path)

    @patch("garmin_health_data.extractor.GarminClient")
    @patch("garmin_health_data.extractor.PersonalApiClient")
    def test_key_file_only(
        self, mock_personal_cls: MagicMock, mock_sso_cls: MagicMock, tmp_path: Path
    ) -> None:
        """
        A saved key alone yields a Personal-API-only hybrid client.

        :param mock_personal_cls: Mock PersonalApiClient class.
        :param mock_sso_cls: Mock GarminClient class.
        :param tmp_path: Pytest temporary directory fixture.
        """
        (tmp_path / API_KEY_FILENAME).write_text("k\n")
        mock_personal_cls.return_value.user_id = 99

        extractor = self._extractor(tmp_path)
        extractor.authenticate(str(tmp_path))

        assert isinstance(extractor.garmin_client, HybridClient)
        assert extractor.garmin_client.sso is None
        assert extractor.user_id == 99
        mock_personal_cls.assert_called_once_with("k")
        mock_sso_cls.from_tokens.assert_not_called()

    @patch("garmin_health_data.extractor.GarminClient")
    def test_sso_tokens_only(self, mock_sso_cls: MagicMock, tmp_path: Path) -> None:
        """
        SSO tokens alone keep the previous behavior (user ID from user settings).

        :param mock_sso_cls: Mock GarminClient class.
        :param tmp_path: Pytest temporary directory fixture.
        """
        (tmp_path / "garmin_tokens.json").write_text("{}")
        mock_sso_cls.from_tokens.return_value.get_user_profile.return_value = {"id": 5}

        extractor = self._extractor(tmp_path)
        extractor.authenticate(str(tmp_path))

        assert extractor.garmin_client.personal is None
        assert extractor.user_id == 5

    @patch("garmin_health_data.extractor.GarminClient")
    @patch("garmin_health_data.extractor.PersonalApiClient")
    def test_broken_key_falls_back_to_sso(
        self, mock_personal_cls: MagicMock, mock_sso_cls: MagicMock, tmp_path: Path
    ) -> None:
        """
        A rejected key is a warning when SSO tokens still work.

        :param mock_personal_cls: Mock PersonalApiClient class.
        :param mock_sso_cls: Mock GarminClient class.
        :param tmp_path: Pytest temporary directory fixture.
        """
        (tmp_path / API_KEY_FILENAME).write_text("expired\n")
        (tmp_path / "garmin_tokens.json").write_text("{}")
        mock_personal_cls.return_value.authenticate.side_effect = RuntimeError("401")
        mock_sso_cls.from_tokens.return_value.get_user_profile.return_value = {"id": 5}

        extractor = self._extractor(tmp_path)
        extractor.authenticate(str(tmp_path))

        assert extractor.garmin_client.personal is None
        assert extractor.user_id == 5

    def test_no_credentials_raises(self, tmp_path: Path) -> None:
        """
        Neither a key nor tokens raises a RuntimeError pointing at ``garmin auth``.

        :param tmp_path: Pytest temporary directory fixture.
        """
        with pytest.raises(RuntimeError, match="garmin auth"):
            self._extractor(tmp_path).authenticate(str(tmp_path))

    def test_pre_authenticated_client_used(self, tmp_path: Path) -> None:
        """
        A client passed in (from the env key) is used without a key file.

        :param tmp_path: Pytest temporary directory fixture.
        """
        personal = MagicMock()
        personal.user_id = 77

        extractor = self._extractor(tmp_path)
        extractor.authenticate(str(tmp_path), personal_client=personal)

        assert extractor.garmin_client.personal is personal
        assert extractor.user_id == 77


class TestExtractorSkipsUnsupported:
    """
    Tests for skipping data types the configured auth cannot serve.
    """

    def test_unsupported_types_skipped_not_failed(self, tmp_path: Path) -> None:
        """
        Types outside the Personal API are skipped and not recorded as failures.

        :param tmp_path: Pytest temporary directory fixture.
        """
        extractor = GarminExtractor(
            date(2026, 1, 1),
            date(2026, 1, 1),
            tmp_path,
            data_types=["HEART_RATE", "TRAINING_STATUS"],
        )
        client = MagicMock()
        client.supports.side_effect = lambda name: name == "get_heart_rates"
        client.get_heart_rates.return_value = {"heartRateValues": [[1, 60]]}
        extractor.garmin_client = client
        extractor.user_id = 1

        files = extractor.extract_garmin_data()

        assert [f.name for f in files] == ["1_HEART_RATE_2026-01-01T12-00-00Z.json"]
        assert extractor.skipped_data_types == ["TRAINING_STATUS"]
        assert extractor.failures == []
        client.get_training_status.assert_not_called()

    def test_exercise_sets_skipped_when_unsupported(self, tmp_path: Path) -> None:
        """
        Strength activities skip the exercise-sets call the Personal API lacks.

        :param tmp_path: Pytest temporary directory fixture.
        """
        extractor = GarminExtractor(date(2026, 1, 1), date(2026, 1, 1), tmp_path)
        client = MagicMock()
        client.supports.side_effect = lambda name: name != "get_activity_exercise_sets"
        client.get_activities_by_date.return_value = [
            {
                "activityId": 10,
                "startTimeLocal": "2026-01-01T08:00:00.0",
                "activityType": {"typeKey": "strength_training"},
            }
        ]
        client.download_activity.return_value = b"not a zip"
        extractor.garmin_client = client
        extractor.user_id = 1

        extractor.extract_fit_activities()

        client.download_activity.assert_called_once()
        client.get_activity_exercise_sets.assert_not_called()

    def test_menstrual_probe_fails_open_without_calendar(self, tmp_path: Path) -> None:
        """
        Without the calendar endpoint, the dayview fan-out runs (fail open).

        :param tmp_path: Pytest temporary directory fixture.
        """
        extractor = GarminExtractor(date(2026, 1, 1), date(2026, 1, 1), tmp_path)
        client = MagicMock()
        client.supports.side_effect = lambda name: name != "get_menstrual_calendar_data"
        extractor.garmin_client = client

        assert extractor._menstrual_days_have_data(date(2026, 1, 1), date(2026, 1, 2))
        client.get_menstrual_calendar_data.assert_not_called()


class TestExtractEnvKeyAccount:
    """
    Tests for ``extract`` discovering the account owning an env/.env API key.
    """

    @patch("garmin_health_data.extractor.GarminExtractor")
    @patch("garmin_health_data.extractor.PersonalApiClient")
    @patch(
        "garmin_health_data.auth.discover_accounts",
        side_effect=FileNotFoundError("no dir"),
    )
    def test_env_key_alone_is_an_account(
        self,
        mock_discover: MagicMock,
        mock_personal_cls: MagicMock,
        mock_extractor_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        An env key works without ``garmin auth`` or a ``~/.garminconnect`` dir.

        :param mock_discover: Mock discover_accounts (no directory).
        :param mock_personal_cls: Mock PersonalApiClient class.
        :param mock_extractor_cls: Mock GarminExtractor class.
        :param monkeypatch: Pytest monkeypatch fixture.
        :param tmp_path: Pytest temporary directory fixture.
        """
        monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
        mock_personal_cls.return_value.user_id = 321
        mock_extractor_cls.return_value.extract_garmin_data.return_value = []
        mock_extractor_cls.return_value.extract_fit_activities.return_value = []
        mock_extractor_cls.return_value.failures = []

        extract(tmp_path, "2026-01-01", "2026-01-02")

        mock_personal_cls.assert_called_once_with("env-key")
        kwargs = mock_extractor_cls.return_value.authenticate.call_args.kwargs
        assert kwargs["personal_client"] is mock_personal_cls.return_value
        assert kwargs["token_store_dir"].endswith("321")

    @patch("garmin_health_data.extractor.GarminExtractor")
    @patch("garmin_health_data.extractor.PersonalApiClient")
    @patch("garmin_health_data.auth.discover_accounts")
    def test_env_key_merges_with_existing_account(
        self,
        mock_discover: MagicMock,
        mock_personal_cls: MagicMock,
        mock_extractor_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        An env key for an already-discovered account is attached, not duplicated.

        :param mock_discover: Mock discover_accounts.
        :param mock_personal_cls: Mock PersonalApiClient class.
        :param mock_extractor_cls: Mock GarminExtractor class.
        :param monkeypatch: Pytest monkeypatch fixture.
        :param tmp_path: Pytest temporary directory fixture.
        """
        monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
        mock_discover.return_value = [("321", tmp_path / "321")]
        mock_personal_cls.return_value.user_id = 321
        mock_extractor_cls.return_value.extract_garmin_data.return_value = []
        mock_extractor_cls.return_value.extract_fit_activities.return_value = []
        mock_extractor_cls.return_value.failures = []

        extract(tmp_path, "2026-01-01", "2026-01-02")

        assert mock_extractor_cls.return_value.authenticate.call_count == 1

    @patch("garmin_health_data.extractor.GarminExtractor")
    @patch("garmin_health_data.extractor.PersonalApiClient")
    @patch(
        "garmin_health_data.auth.discover_accounts",
        side_effect=FileNotFoundError("no dir"),
    )
    def test_bad_env_key_and_no_accounts(
        self,
        mock_discover: MagicMock,
        mock_personal_cls: MagicMock,
        mock_extractor_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        A rejected env key with no other account extracts nothing.

        :param mock_discover: Mock discover_accounts (no directory).
        :param mock_personal_cls: Mock PersonalApiClient class.
        :param mock_extractor_cls: Mock GarminExtractor class.
        :param monkeypatch: Pytest monkeypatch fixture.
        :param tmp_path: Pytest temporary directory fixture.
        """
        monkeypatch.setenv(API_KEY_ENV_VAR, "bad")
        mock_personal_cls.return_value.authenticate.side_effect = RuntimeError("401")

        result = extract(tmp_path, "2026-01-01", "2026-01-02")

        assert result["garmin_files"] == 0
        mock_extractor_cls.assert_not_called()
