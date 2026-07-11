import sqlite3
from collections.abc import Callable, Mapping


Migration = Callable[[sqlite3.Connection], None]


class UnsupportedSchemaVersionError(RuntimeError):
    """Raised when a database was created by a newer application version."""


def apply_sqlite_migrations(
    conn: sqlite3.Connection,
    *,
    database_name: str,
    migrations: Mapping[int, Migration],
) -> int:
    if not migrations:
        raise ValueError("at least one database migration is required")

    latest_version = max(migrations)
    current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if current_version > latest_version:
        raise UnsupportedSchemaVersionError(
            f"{database_name} schema version {current_version} is newer than "
            f"supported version {latest_version}"
        )

    for target_version in range(current_version + 1, latest_version + 1):
        migration = migrations.get(target_version)
        if migration is None:
            raise RuntimeError(
                f"{database_name} migration {target_version} is missing"
            )
        try:
            conn.execute("BEGIN IMMEDIATE")
            migration(conn)
            conn.execute(f"PRAGMA user_version = {target_version}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return latest_version
