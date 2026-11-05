"""
Database initialization and management for garmin-health-data.

Handles SQLite database creation, session management, and query utilities.
"""

import sqlite3
import sys
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import click
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

# Handle importlib.resources for different Python versions
if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

from garmin_health_data.models import (
    Activity,
    BodyBattery,
    DailySummary,
    Floors,
    HeartRate,
    HrvSummary,
    IntensityMinutes,
    Respiration,
    Sleep,
    Steps,
    Stress,
    TrainingReadiness,
    User,
)

# Minimum SQLite version. 3.35.0 (2021-03-12) added support for
# `INSERT ... ON CONFLICT ... RETURNING`, which the bulk upsert helper relies on
# to read back database-assigned columns (e.g. the auto-increment `sleep_id`).
_MIN_SQLITE_VERSION = (3, 35, 0)


def check_sqlite_version() -> None:
    """
    Validate that the linked SQLite library meets the minimum required version.

    :raises RuntimeError: If the installed SQLite version is too old.
    """
    if sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION:
        required = ".".join(str(v) for v in _MIN_SQLITE_VERSION)
        raise RuntimeError(
            f"garmin-health-data requires SQLite >= {required} "
            f"(found {sqlite3.sqlite_version}). "
            "Upgrade your Python build or system SQLite library."
        )


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    """
    Enable SQLite foreign-key enforcement on every new connection.

    SQLite defaults `PRAGMA foreign_keys` to OFF per connection, so `ON DELETE CASCADE`
    clauses in the schema are silently inert without this. Registered as a SQLAlchemy
    `connect` event listener on engines created by :func:`get_engine`.

    :param dbapi_connection: DB-API connection instance (sqlite3.Connection).
    :param _connection_record: SQLAlchemy connection record (unused).
    """
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def get_engine(db_path: str = "garmin_data.db"):
    """
    Create SQLAlchemy engine for SQLite database.

    Attaches a `connect` event listener that runs `PRAGMA foreign_keys=ON` on every new
    connection so `ON DELETE CASCADE` clauses in the schema actually fire.

    :param db_path: Path to SQLite database file.
    :return: SQLAlchemy engine.
    """
    check_sqlite_version()
    db_file = Path(db_path).expanduser().resolve()

    # Create database URL (use as_posix() for cross-platform compatibility).
    db_url = f"sqlite:///{db_file.as_posix()}"

    # Create engine with sensible defaults for SQLite.
    engine = create_engine(
        db_url,
        echo=False,  # Set to True for SQL debugging.
        connect_args={"check_same_thread": False},  # Allow multi-threading.
    )
    event.listen(engine, "connect", _enable_sqlite_foreign_keys)

    return engine


def create_tables(db_path: str = "garmin_data.db") -> None:
    """
    Create all tables in the database by executing the DDL file.

    The schema is defined in tables.ddl which includes inline comments preserved in the
    database.

    :param db_path: Path to SQLite database file.
    """
    # Fail fast if the linked SQLite library can't support the bulk upsert
    # helper's RETURNING-based path. Mirrors the gate in `get_engine` so
    # `garmin init` surfaces the same clear error as the read/write path.
    check_sqlite_version()

    # Execute DDL file to create all tables with inline comments.
    # Use importlib.resources to read the packaged resource file.
    try:
        ddl_sql = files("garmin_health_data").joinpath("tables.ddl").read_text()
    except (FileNotFoundError, TypeError):
        # Fallback for development/editable installs
        ddl_file = Path(__file__).parent / "tables.ddl"
        if not ddl_file.exists():
            raise FileNotFoundError(f"Schema DDL file not found: {ddl_file}")
        ddl_sql = ddl_file.read_text()

    # Use raw SQLite connection to execute the script (supports multiple statements).
    # This preserves inline comments in the database schema.
    db_file = Path(db_path).expanduser()
    conn = sqlite3.connect(str(db_file))
    try:
        conn.executescript(ddl_sql)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_session(db_path: str = "garmin_data.db"):
    """
    Context manager for database sessions.

    :param db_path: Path to SQLite database file.
    :yield: SQLAlchemy Session.
    """
    engine = get_engine(db_path)
    session = Session(engine)

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def initialize_database(db_path: str = "garmin_data.db") -> None:
    """
    Initialize a new database with all tables and indexes.

    :param db_path: Path to SQLite database file.
    """
    db_file = Path(db_path).expanduser()

    if db_file.exists():
        click.echo(f"Database already exists at: {db_file}")
    else:
        click.echo(f"Creating new database at: {db_file}")

    create_tables(db_path)
    click.secho("✅ Database initialized successfully", fg="green")
    click.echo(
        "\nSchema includes inline documentation. To view table definitions:\n"
        f"  sqlite3 {db_file} \"SELECT sql FROM sqlite_master WHERE type='table';\""
    )


# (data type, model, column) tracked for the last update date. ``date`` columns are
# returned as-is, ``datetime`` ones are truncated to their date.
_LAST_UPDATE_COLUMNS = (
    ("sleep", Sleep, Sleep.start_ts),
    ("heart_rate", HeartRate, HeartRate.timestamp),
    ("activity", Activity, Activity.start_ts),
    ("stress", Stress, Stress.timestamp),
    ("body_battery", BodyBattery, BodyBattery.timestamp),
    ("steps", Steps, Steps.timestamp),
    ("respiration", Respiration, Respiration.timestamp),
    ("floors", Floors, Floors.timestamp),
    ("intensity_minutes", IntensityMinutes, IntensityMinutes.timestamp),
    ("training_readiness", TrainingReadiness, TrainingReadiness.timestamp),
    # Personal API data types.
    ("daily_summary", DailySummary, DailySummary.date),
    ("hrv_summary", HrvSummary, HrvSummary.date),
)


def get_last_update_dates(
    db_path: str = "garmin_data.db",
    user_ids: Optional[Iterable[Union[int, str]]] = None,
) -> Dict[str, Optional[date]]:
    """
    Get the last update date for each data type.

    :param db_path: Path to SQLite database file.
    :param user_ids: Optional Garmin user IDs to restrict the lookup to. If None, all
        accounts stored in the database are considered.
    :return: Dictionary mapping data type name to last update date.
    """
    ids = [int(u) for u in user_ids] if user_ids is not None else None
    with get_session(db_path) as session:
        dates: Dict[str, Optional[date]] = {}
        for name, model, column in _LAST_UPDATE_COLUMNS:
            query = select(func.max(column))
            if ids is not None:
                query = query.where(model.user_id.in_(ids))
            value = session.execute(query).scalar()
            if isinstance(value, datetime):
                value = value.date()
            dates[name] = value
        return dates


def get_latest_date(
    db_path: str = "garmin_data.db",
    user_ids: Optional[Iterable[Union[int, str]]] = None,
) -> Optional[date]:
    """
    Get the most recent date across all data types.

    :param db_path: Path to SQLite database file.
    :param user_ids: Optional Garmin user IDs to restrict the lookup to.
    :return: Most recent date or None if database is empty (for these accounts).
    """
    dates = get_last_update_dates(db_path, user_ids)
    valid_dates = [d for d in dates.values() if d is not None]

    if not valid_dates:
        return None

    return max(valid_dates)


def get_record_counts(db_path: str = "garmin_data.db") -> Dict[str, int]:
    """
    Get record counts for all major tables.

    :param db_path: Path to SQLite database file.
    :return: Dictionary mapping table name to record count.
    """
    with get_session(db_path) as session:
        counts = {}

        counts["users"] = session.execute(select(func.count(User.user_id))).scalar()
        counts["activities"] = session.execute(
            select(func.count(Activity.activity_id))
        ).scalar()
        counts["sleep_sessions"] = session.execute(
            select(func.count(Sleep.sleep_id))
        ).scalar()
        counts["heart_rate_readings"] = session.execute(
            select(func.count(HeartRate.timestamp))
        ).scalar()
        counts["stress_readings"] = session.execute(
            select(func.count(Stress.timestamp))
        ).scalar()
        counts["body_battery_readings"] = session.execute(
            select(func.count(BodyBattery.timestamp))
        ).scalar()
        counts["step_readings"] = session.execute(
            select(func.count(Steps.timestamp))
        ).scalar()
        counts["respiration_readings"] = session.execute(
            select(func.count(Respiration.timestamp))
        ).scalar()
        counts["daily_summaries"] = session.execute(
            select(func.count(DailySummary.date))
        ).scalar()
        counts["hrv_summaries"] = session.execute(
            select(func.count(HrvSummary.date))
        ).scalar()

        return counts


def get_database_size(db_path: str = "garmin_data.db") -> int:
    """
    Get size of database file in bytes.

    :param db_path: Path to SQLite database file.
    :return: Size in bytes, or 0 if file doesn't exist.
    """
    db_file = Path(db_path).expanduser()

    if not db_file.exists():
        return 0

    return db_file.stat().st_size


def database_exists(db_path: str = "garmin_data.db") -> bool:
    """
    Check if database file exists.

    :param db_path: Path to SQLite database file.
    :return: True if database exists, False otherwise.
    """
    db_file = Path(db_path).expanduser()
    return db_file.exists()
