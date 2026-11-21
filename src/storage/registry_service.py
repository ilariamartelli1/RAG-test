"""Manage the local document and build registry."""

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from src.models.registry_models import DocumentRegistryRecord


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


def _registry_path(database_path: str | os.PathLike[str]) -> str:
    path = os.fspath(database_path)
    if not path or path == ":memory:" or path.startswith("file:"):
        raise ValueError("registry requires a persistent filesystem path, not a SQLite URI")
    return path


def _prepare_registry(connection: sqlite3.Connection, *, create_if_empty: bool) -> None:
    application_id = connection.execute("PRAGMA application_id").fetchone()[0]
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    objects = connection.execute(
        "SELECT name, type, sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()

    if create_if_empty and application_id == 0 and version == 0 and not objects:
        connection.execute(_DOCUMENTS_TABLE)
        connection.execute(_BUILDS_TABLE)
        connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        return

    if application_id != _APPLICATION_ID or version != _SCHEMA_VERSION:
        raise ValueError("database is not a supported document registry")

    tables = {name: sql for name, kind, sql in objects if kind == "table"}
    if tables.get("documents") != _DOCUMENTS_TABLE or tables.get("builds") != _BUILDS_TABLE:
        raise ValueError("document registry schema does not match the expected version")


def initialize_registry(database_path: str | os.PathLike[str]) -> None:
    """Create or verify the registry schema at a persistent SQLite path.

    An existing database must either be empty or have this application's
    expected schema and version. No document or build records are created.
    Settings JSON is not validated by the schema. Future registry operations
    must validate it and enable foreign keys on each connection.
    """
    path = _registry_path(database_path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _prepare_registry(connection, create_if_empty=True)


def register_document(
    database_path: str | os.PathLike[str], record: DocumentRegistryRecord
) -> None:
    """Save a document ID to an existing registry.

    Duplicate IDs raise sqlite3.IntegrityError. This does not register a build,
    store a source document, or establish ownership of a Qdrant collection.
    """
    path = _registry_path(database_path)
    validated_record = DocumentRegistryRecord(document_id=record.document_id)
    uri = Path(path).absolute().as_uri() + "?mode=rw"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _prepare_registry(connection, create_if_empty=False)
            connection.execute(
                "INSERT INTO documents (document_id) VALUES (?)",
                (validated_record.document_id,),
            )
