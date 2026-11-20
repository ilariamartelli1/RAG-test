"""Initialize the local document and build registry."""

import os
import sqlite3
from contextlib import closing


_APPLICATION_ID = 0x52414752
_SCHEMA_VERSION = 1

_DOCUMENTS_TABLE = """CREATE TABLE documents (
    document_id TEXT NOT NULL PRIMARY KEY
)"""

_BUILDS_TABLE = """CREATE TABLE builds (
    build_id TEXT NOT NULL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE RESTRICT,
    collection_name TEXT NOT NULL UNIQUE,
    index_settings_json TEXT NOT NULL
)"""


def initialize_registry(database_path: str | os.PathLike[str]) -> None:
    """Create or verify the registry schema at a persistent SQLite path.

    An existing database must either be empty or have this application's
    expected schema and version. No document or build records are created.
    Settings JSON is not validated by the schema. Future registry operations
    must validate it and enable foreign keys on each connection.
    """
    path = os.fspath(database_path)
    if not path or path == ":memory:" or path.startswith("file:"):
        raise ValueError("registry requires a persistent filesystem path, not a SQLite URI")

    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            application_id = connection.execute("PRAGMA application_id").fetchone()[0]
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            objects = connection.execute(
                "SELECT name, type, sql FROM sqlite_schema "
                "WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()

            if application_id == 0 and version == 0 and not objects:
                connection.execute(_DOCUMENTS_TABLE)
                connection.execute(_BUILDS_TABLE)
                connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                return

            if application_id != _APPLICATION_ID or version != _SCHEMA_VERSION:
                raise ValueError("database is not a supported document registry")

            tables = {name: sql for name, kind, sql in objects if kind == "table"}
            if (
                tables.get("documents") != _DOCUMENTS_TABLE
                or tables.get("builds") != _BUILDS_TABLE
            ):
                raise ValueError("document registry schema does not match the expected version")
