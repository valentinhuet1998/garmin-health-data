"""
Garmin Connect authentication module.

Supports two authentication methods, which can be combined per account:

- **Personal API key** (default): a key created at
  https://connect.garmin.com/app/settings/personalApiToken, read from the
  ``GARMIN_PERSONAL_API_KEY`` environment variable, a ``.env`` file in the current
  directory, or ``~/.garminconnect/<user_id>/personal_api_key``. Serves activities,
  FIT files, heart rate, sleep summaries, body composition and menstrual cycle days.
- **Email/password (SSO)**: OAuth token management with Garmin Connect, including
  Multi-Factor Authentication (MFA) support. Needed for the data types the Personal
  API does not serve (training status/readiness, records, respiration, ...).
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlparse, urlunparse

import click
from garmin_health_data.garmin_client import GarminClient, PersonalApiClient

# Personal API key sources.
API_KEY_ENV_VAR = "GARMIN_PERSONAL_API_KEY"
API_KEY_FILENAME = "personal_api_key"
SSO_TOKENS_FILENAME = "garmin_tokens.json"
# ``.env`` file read for the API key, relative to the current directory.
DEFAULT_DOTENV_PATH = Path(".env")

# Manual-ticket bootstrap constants (workaround for the Cloudflare-blocked
# automated login). The mobile-app service is used to mint the ticket because a
# browser never validates (consumes) that service's ticket, so it stays usable
# for the DI token exchange. See the ``garmin auth --manual`` flow.
MANUAL_SERVICE_URL = "https://mobile.integration.garmin.com/gcm/android"
MANUAL_MINT_URL = (
    "https://sso.garmin.com/sso/signin"
    "?service=https%3A%2F%2Fmobile.integration.garmin.com%2Fgcm%2Fandroid"
)


def get_credentials() -> Tuple[str, str]:
    """
    Get Garmin Connect credentials from user input or environment variables.

    :return: Tuple of (email, password).
    """
    # Try environment variables first.
    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")

    if email and password:
        click.echo(
            click.style("📧 Using credentials from environment variables", fg="cyan")
        )
        click.echo(f"   Email: {email}")
        return email, password

    # Prompt for credentials interactively.
    click.echo(click.style("🔐 Garmin Connect Authentication", fg="cyan", bold=True))
    click.echo()

    email = click.prompt("   Email", type=str)
    password = click.prompt("   Password", type=str, hide_input=True)

    if not email or not password:
        raise click.ClickException("Email and password are required")

    return email, password


def get_mfa_code() -> str:
    """
    Prompt user for MFA code.

    :return: MFA code string.
    """
    click.echo()
    click.echo(click.style("🔢 Multi-Factor Authentication Required", fg="yellow"))
    click.echo("   Check your email or phone for the MFA code")
    click.echo()

    mfa_code = click.prompt("   Enter 6-digit MFA code", type=str)

    if not mfa_code.isdigit() or len(mfa_code) != 6:
        click.secho("⚠️  Warning: MFA code should be 6 digits", fg="yellow")

    return mfa_code


def _handle_mfa_authentication(garmin: GarminClient, result2) -> None:
    """
    Handle MFA authentication with one retry attempt.

    :param garmin: Garmin client instance.
    :param result2: MFA continuation token from login result.
    """
    click.secho("✅ Initial authentication successful", fg="green")

    for attempt in range(2):  # Allow 2 attempts.
        try:
            mfa_code = get_mfa_code()
            click.echo("🔢 Completing MFA authentication...")
            garmin.resume_login(result2, mfa_code)
            click.secho("✅ MFA authentication successful", fg="green", bold=True)
            return  # Success.

        except Exception as e:
            if attempt == 0:  # First attempt failed.
                click.secho(f"❌ MFA authentication failed: {str(e)}", fg="red")
                click.echo("🔄 Please try again with a fresh MFA code")
                continue
            # Second attempt failed.
            click.secho(
                f"❌ MFA authentication failed after 2 attempts", fg="red", bold=True
            )
            raise


def _print_troubleshooting() -> None:
    """
    Print common troubleshooting steps.
    """
    click.echo()
    click.secho("🔍 Troubleshooting:", fg="yellow", bold=True)
    click.echo("   - Verify your email and password are correct")
    click.echo("   - Check for typos or case sensitivity")
    click.echo("   - Ensure you have internet connectivity")
    click.echo("   - If MFA is enabled, make sure the MFA code is current")
    click.echo("   - Try running the command again")
    click.echo("   - Check if Garmin Connect services are operational")
    click.echo()


def discover_accounts(
    base_token_dir: str = "~/.garminconnect",
) -> List[Tuple[str, Path]]:
    """
    Discover Garmin Connect accounts by scanning token subdirectories.

    Each numeric subdirectory in the base token directory represents a user_id with
    saved OAuth tokens.

    :param base_token_dir: Base directory containing per-account token subdirectories.
    :return: Sorted list of (user_id, token_dir_path) tuples.
    :raises FileNotFoundError: If base directory does not exist.
    :raises NotADirectoryError: If base path is not a directory.
    :raises RuntimeError: If no accounts are found.
    """
    base_path = Path(base_token_dir).expanduser()

    if not base_path.exists():
        raise FileNotFoundError(f"Token directory does not exist: {base_path}")

    if not base_path.is_dir():
        raise NotADirectoryError(f"Token path is not a directory: {base_path}")

    # Scan for numeric subdirectories that contain token files.
    accounts = [
        (entry.name, entry)
        for entry in sorted(base_path.iterdir())
        if entry.is_dir() and entry.name.isdigit() and any(entry.iterdir())
    ]

    if accounts:
        return accounts

    # Legacy fallback: check for token files at root level.
    token_files = list(base_path.glob("*token*.json"))
    if token_files:
        click.secho(
            "Warning: Found legacy token layout (tokens at root level). "
            "Run 'garmin auth' to migrate to per-account subdirectories.",
            fg="yellow",
        )
        return [("legacy", base_path)]

    raise RuntimeError(
        f"No accounts found in {base_path}. Run 'garmin auth' to authenticate."
    )


def _save_client_tokens(
    garmin: GarminClient, base_path: Path, silent: bool = False
) -> str:
    """
    Persist a logged-in client's tokens under ``base_path/<user_id>/``.

    Auto-detects the Garmin user ID from the profile, creates the per-account
    subdirectory with restrictive permissions, and writes ``garmin_tokens.json``.

    :param garmin: Authenticated Garmin client (DI tokens already set).
    :param base_path: Base directory for per-account token storage.
    :param silent: If True, suppress non-essential output.
    :return: The authenticated Garmin user ID as a string.
    :raises RuntimeError: If the user ID cannot be read from the profile.
    """
    user_id = garmin.get_user_profile().get("id")
    if not user_id:
        raise RuntimeError(
            "Could not determine user ID from Garmin profile. "
            "The 'id' field was missing from get_user_profile() response."
        )

    if not silent:
        click.echo(f"👤 Detected user ID: {user_id}")
        click.echo("💾 Saving authentication tokens...")

    # Ensure base and per-account token directories exist with proper permissions.
    base_path.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        base_path.chmod(0o700)

    token_path = base_path / str(user_id)
    token_path.mkdir(exist_ok=True)
    if sys.platform != "win32":
        token_path.chmod(0o700)

    # dump() writes garmin_tokens.json with 0o600 from creation time.
    garmin.dump(str(token_path))

    if not silent:
        click.echo()
        click.secho("✅ Tokens successfully saved!", fg="green", bold=True)
        click.echo(f"   User ID:  {user_id}")
        click.echo(f"   Location: {token_path}")

    return str(user_id)


def _parse_manual_ticket(raw: str) -> Tuple[str, str]:
    """
    Parse a pasted manual-login value into a ticket and its service URL.

    Accepts either a bare ``ST-...`` service ticket (in which case the default mobile
    service URL is assumed) or the full redirect URL that carries the ticket in its
    ``ticket`` query parameter.

    :param raw: Raw ``ST-...`` ticket or a full redirect URL.
    :return: Tuple of (ticket, service_url).
    :raises click.ClickException: If no ticket can be extracted.
    """
    raw = raw.strip()
    if raw.startswith("ST-"):
        return raw, MANUAL_SERVICE_URL

    parsed = urlparse(raw)
    ticket = parse_qs(parsed.query).get("ticket", [None])[0]
    if not ticket:
        raise click.ClickException(
            "No ticket found. Paste the 'ST-...' value or the full URL that "
            "contains 'ticket=ST-...'."
        )
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise click.ClickException(
            "Could not read a service URL from the pasted value. Paste the bare "
            "'ST-...' ticket, or the full 'https://...' redirect URL that carries it."
        )
    # Rebuild the service URL as the redirect target with only the ``ticket``
    # param removed, preserving the exact original encoding and order of any
    # remaining params. A CAS ticket is bound to the exact service-URL string, so
    # re-encoding the query (parse + urlencode) could invalidate the exchange.
    remaining_query = "&".join(
        part
        for part in parsed.query.split("&")
        if part and not part.startswith("ticket=")
    )
    service_url = urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, "", remaining_query, "")
    )
    return ticket, service_url


def bootstrap_from_ticket(
    ticket_input: str,
    base_token_dir: str = "~/.garminconnect",
    service_url: Optional[str] = None,
    silent: bool = False,
) -> str:
    """
    Exchange a browser-captured CAS service ticket for DI tokens and save them.

    Bypasses the Cloudflare-blocked automated login: the user obtains a ticket from
    a real, already-authenticated browser session (which passes Cloudflare) and
    provides it here. The resulting DI tokens auto-refresh for ~30 days, so this is
    roughly a monthly step rather than a per-run one.

    :param ticket_input: Raw ``ST-...`` ticket or the full redirect URL carrying it.
    :param base_token_dir: Base directory for per-account token storage.
    :param service_url: SSO service the ticket was minted for. Defaults to the value
        parsed from ``ticket_input`` (or the mobile service for a bare ticket).
    :param silent: If True, suppress non-essential output.
    :return: The authenticated Garmin user ID as a string.
    :raises click.ClickException: If the ticket exchange or the token save fails.
    """
    ticket, parsed_service = _parse_manual_ticket(ticket_input)
    service = service_url or parsed_service
    base_path = Path(base_token_dir).expanduser()

    garmin = GarminClient()
    try:
        garmin._exchange_service_ticket(ticket, service_url=service)
    except click.Abort:
        # Let user interrupts propagate rather than masking them as a failure.
        raise
    except Exception as e:
        raise click.ClickException(
            f"Ticket exchange failed: {e}. The ticket may be expired or already "
            "used (they are single-use and last ~1 minute). Mint a fresh one and "
            "paste it right away."
        ) from e

    try:
        return _save_client_tokens(garmin, base_path, silent=silent)
    except (RuntimeError, OSError) as e:
        # Missing profile id (RuntimeError) or a filesystem failure while writing
        # tokens (OSError) should exit cleanly, not dump a traceback.
        raise click.ClickException(f"Could not save tokens: {e}") from e


def _print_manual_instructions() -> None:
    """
    Print step-by-step instructions for capturing a login ticket from the browser.
    """
    # Console/bookmarklet snippet: read the ticket out of the SSO "Success" page
    # HTML (same-origin) and copy it to the clipboard via the devtools copy() helper.
    snippet = r"copy(document.documentElement.outerHTML.match(/ST-[\w.-]+/)[0])"

    click.echo()
    click.secho("🔑 Manual authentication (browser ticket)", fg="cyan", bold=True)
    click.echo(
        "Garmin's Cloudflare protection blocks automated login. Grab a login "
        "ticket from your own browser instead:"
    )
    click.echo()
    click.echo("  1. Log into Garmin Connect in your browser (any normal login).")
    click.echo(
        "  2. Open this URL in that same browser. You'll land on a page titled "
        '"Success":'
    )
    click.echo()
    click.secho(f"     {MANUAL_MINT_URL}", fg="blue")
    click.echo()
    click.echo(
        "  3. Open the browser console (Cmd+Opt+J on Chrome, or Ctrl+Shift+J), "
        "paste this line, and press Enter."
    )
    click.echo("     It copies the ticket to your clipboard:")
    click.echo()
    click.secho(f"     {snippet}", fg="green")
    click.echo()
    click.echo("  4. Paste the ticket below (it is single-use and expires quickly).")
    click.echo()


def manual_auth(base_token_dir: str = "~/.garminconnect", silent: bool = False) -> str:
    """
    Run the interactive manual-ticket login flow.

    Prints capture instructions, prompts for the pasted ticket, and exchanges it.

    :param base_token_dir: Base directory for per-account token storage.
    :param silent: If True, suppress the instructional output.
    :return: The authenticated Garmin user ID as a string.
    """
    if not silent:
        _print_manual_instructions()
    ticket_input = click.prompt("   Ticket (ST-... or full URL)", type=str)
    return bootstrap_from_ticket(
        ticket_input, base_token_dir=base_token_dir, silent=silent
    )


def _read_dotenv_value(name: str, dotenv_path: Path) -> Optional[str]:
    """
    Read one variable from a ``.env`` file without loading the rest into the env.

    Supports ``KEY=value``, ``export KEY=value``, single/double-quoted values, blank
    lines and ``#`` comment lines.

    :param name: Variable name to look up.
    :param dotenv_path: Path to the ``.env`` file.
    :return: The value, or None if the file or variable is missing or empty.
    """
    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        return value.strip() or None
    return None


def read_env_api_key(dotenv_path: Optional[Path] = None) -> Optional[str]:
    """
    Read the Personal API key from the environment or a ``.env`` file.

    The ``GARMIN_PERSONAL_API_KEY`` environment variable wins over the ``.env`` file.

    :param dotenv_path: ``.env`` file to read. Defaults to ``DEFAULT_DOTENV_PATH``
        (``./.env``).
    :return: The API key, or None if neither source sets it.
    """
    value = (os.getenv(API_KEY_ENV_VAR) or "").strip()
    if value:
        return value
    return _read_dotenv_value(API_KEY_ENV_VAR, dotenv_path or DEFAULT_DOTENV_PATH)


def load_account_api_key(token_dir: Path) -> Optional[str]:
    """
    Read a Personal API key saved for one account by ``garmin auth``.

    :param token_dir: Per-account directory (``~/.garminconnect/<user_id>/``).
    :return: The API key, or None if no key file exists or it is empty.
    """
    key_path = Path(token_dir).expanduser() / API_KEY_FILENAME
    try:
        value = key_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def has_sso_tokens(token_dir: Path) -> bool:
    """
    :param token_dir: Per-account directory, or a direct token-file path.
    :return: True if email/password (SSO) tokens exist there.
    """
    path = Path(token_dir).expanduser()
    return path.is_file() or (path / SSO_TOKENS_FILENAME).is_file()


def _write_secret_file(path: Path, content: str) -> None:
    """
    Write a secret to disk, readable only by the owner on POSIX systems.

    Writes a sibling temp file created with mode ``0o600`` and atomically replaces the
    destination, so an interrupted write never leaves a truncated key file.

    :param path: Destination file path.
    :param content: Secret content to write.
    """
    temp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
    fd = os.open(str(temp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        if sys.platform != "win32":
            os.chmod(str(temp_path), 0o600)
        os.replace(str(temp_path), str(path))
    except OSError:
        if temp_path.exists():
            temp_path.unlink()
        raise


def save_api_key(
    api_key: str,
    base_token_dir: str = "~/.garminconnect",
    silent: bool = False,
) -> str:
    """
    Validate a Personal API key and save it under ``base_token_dir/<user_id>/``.

    The key is exchanged for a bearer token to prove it works and to detect the Garmin
    user ID, then stored in ``personal_api_key`` next to any existing SSO tokens.

    :param api_key: Garmin Connect Personal API key.
    :param base_token_dir: Base directory for per-account storage.
    :param silent: If True, suppress non-essential output.
    :return: The Garmin user ID the key belongs to, as a string.
    :raises click.ClickException: If the key is rejected or cannot be saved.
    """
    base_path = Path(base_token_dir).expanduser()

    if not silent:
        click.echo()
        click.echo(click.style("🔑 Validating Personal API key...", fg="cyan"))

    client = PersonalApiClient(api_key)
    try:
        client.authenticate()
    except Exception as e:
        raise click.ClickException(f"Personal API key validation failed: {e}") from e

    user_id = str(client.user_id)
    try:
        base_path.mkdir(parents=True, exist_ok=True)
        account_path = base_path / user_id
        account_path.mkdir(exist_ok=True)
        if sys.platform != "win32":
            base_path.chmod(0o700)
            account_path.chmod(0o700)
        _write_secret_file(account_path / API_KEY_FILENAME, api_key.strip() + "\n")
    except OSError as e:
        raise click.ClickException(f"Could not save the API key: {e}") from e

    if not silent:
        click.secho("✅ Personal API key saved!", fg="green", bold=True)
        click.echo(f"   User ID:  {user_id}")
        click.echo(f"   Location: {account_path / API_KEY_FILENAME}")
        click.echo()
        click.echo("   You can now run: garmin extract")
        click.echo(
            "ℹ️  Some data types (training status/readiness, personal records, race "
            "predictions, respiration, floors, ...) are not served by the Personal "
            "API. Run 'garmin auth --sso' to also log in with email/password and "
            "extract them."
        )
    return user_id


def api_key_auth(
    api_key: Optional[str] = None,
    base_token_dir: str = "~/.garminconnect",
    silent: bool = False,
) -> str:
    """
    Run the Personal API key setup flow.

    Uses ``api_key`` when given, otherwise the ``GARMIN_PERSONAL_API_KEY`` environment
    variable or ``.env`` file, otherwise prompts for it (input hidden).

    :param api_key: Optional API key supplied by the caller.
    :param base_token_dir: Base directory for per-account storage.
    :param silent: If True, suppress non-essential output.
    :return: The Garmin user ID the key belongs to, as a string.
    """
    api_key = (api_key or "").strip() or read_env_api_key()
    if api_key:
        if not silent:
            click.echo(
                click.style(
                    f"🔑 Using Personal API key from {API_KEY_ENV_VAR} / .env",
                    fg="cyan",
                )
            )
    else:
        click.echo(click.style("🔑 Garmin Personal API key", fg="cyan", bold=True))
        click.echo(
            "   Create one at https://connect.garmin.com/app/settings/personalApiToken"
        )
        click.echo()
        api_key = click.prompt("   API key", type=str, hide_input=True).strip()
    return save_api_key(api_key, base_token_dir=base_token_dir, silent=silent)


def refresh_tokens(
    email: str,
    password: str,
    base_token_dir: str = "~/.garminconnect",
    silent: bool = False,
) -> None:
    """
    Refresh Garmin Connect tokens with MFA support.

    Authenticates the user, auto-detects their Garmin user ID, and stores tokens in a
    per-account subdirectory under base_token_dir.

    :param email: Garmin Connect email.
    :param password: Garmin Connect password.
    :param base_token_dir: Base directory for per-account token storage.
    :param silent: If True, suppress non-essential output.
    """
    base_path = Path(base_token_dir).expanduser()

    if not silent:
        click.echo()
        click.echo(click.style("🔄 Authenticating with Garmin Connect...", fg="cyan"))
        click.echo(f"   Token storage: {base_path}")
        click.echo()

    try:
        # Initialize vendored client and attempt login with MFA support.
        garmin = GarminClient()
        login_result = garmin.login(email, password, return_on_mfa=True)

        # Handle different return value formats.
        if isinstance(login_result, tuple) and len(login_result) == 2:
            result1, result2 = login_result

            # Handle MFA if required.
            if result1 == "needs_mfa":
                _handle_mfa_authentication(garmin, result2)
            else:
                if not silent:
                    click.secho(
                        "✅ Authentication successful (no MFA required)",
                        fg="green",
                        bold=True,
                    )
        else:
            # Handle case where login() returns single value or None (no MFA).
            if not silent:
                click.secho(
                    "✅ Authentication successful (no MFA required)",
                    fg="green",
                    bold=True,
                )

        # Persist tokens under a per-account subdirectory.
        _save_client_tokens(garmin, base_path, silent=silent)

        if not silent:
            click.echo()
            click.secho(
                "🎉 Success! You're authenticated with Garmin Connect", fg="green"
            )
            click.echo("   You can now run: garmin extract")
            click.echo()
            click.echo("ℹ️  Tokens auto-refresh transparently during extraction.")

    except Exception as e:
        click.echo()
        click.secho(f"❌ Authentication failed: {str(e)}", fg="red", bold=True)
        _print_troubleshooting()
        raise click.ClickException("Authentication failed")


def check_authentication(base_token_dir: str = "~/.garminconnect") -> bool:
    """
    Check if credentials exist for at least one account.

    Counts a Personal API key in the environment or ``.env``, and any per-account
    directory holding SSO tokens or a saved API key.

    :param base_token_dir: Base directory where per-account tokens are stored.
    :return: True if at least one account has credentials, False otherwise.
    """
    if read_env_api_key():
        return True

    base_path = Path(base_token_dir).expanduser()

    if not base_path.exists():
        return False

    # Check for per-account subdirectories with tokens.
    for entry in base_path.iterdir():
        if entry.is_dir() and entry.name.isdigit() and any(entry.iterdir()):
            return True

    # Legacy fallback: check for token files at root level.
    return any(base_path.glob("*token*.json"))


def ensure_authenticated(base_token_dir: str = "~/.garminconnect") -> None:
    """
    Ensure user is authenticated, prompt for credentials if not.

    :param base_token_dir: Base directory where per-account tokens are stored.
    :raises click.ClickException: If authentication fails.
    """
    if not check_authentication(base_token_dir):
        click.echo()
        click.secho(
            "No authentication tokens found. Please authenticate first.",
            fg="yellow",
            bold=True,
        )
        click.echo()

        if click.confirm(
            "Would you like to set up a Personal API key now?", default=True
        ):
            api_key_auth(base_token_dir=base_token_dir)
        else:
            raise click.ClickException(
                "Authentication required. Run 'garmin auth' to authenticate."
            )
