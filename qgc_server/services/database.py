"""SQLite database storage for QGC Server.

Provides persistent storage for gadgets, manifests, QASM content, and changes.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import uuid

from ..models import (
    Manifest,
    MetricsHints,
    GadgetSummary,
    Change,
    ChangeBucket,
    ChangeSource,
)


class Database:
    """SQLite database for QGC storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript("""
                -- Gadgets table: stores manifests and QASM
                CREATE TABLE IF NOT EXISTS gadgets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    ir TEXT NOT NULL DEFAULT 'openqasm3',
                    qasm TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    original_metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(name, version)
                );

                -- Index for fast lookups
                CREATE INDEX IF NOT EXISTS idx_gadgets_name ON gadgets(name);
                CREATE INDEX IF NOT EXISTS idx_gadgets_name_version ON gadgets(name, version);

                -- Changes table: stores modifications to gadgets
                CREATE TABLE IF NOT EXISTS changes (
                    id TEXT PRIMARY KEY,
                    gadget_name TEXT NOT NULL,
                    gadget_version TEXT NOT NULL,
                    qasm TEXT NOT NULL,
                    metadata_json TEXT,
                    metrics_json TEXT,
                    diff_from_base TEXT,
                    source TEXT NOT NULL DEFAULT 'manual',
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (gadget_name, gadget_version)
                        REFERENCES gadgets(name, version) ON DELETE CASCADE
                );

                -- Index for fast change lookups
                CREATE INDEX IF NOT EXISTS idx_changes_gadget
                    ON changes(gadget_name, gadget_version);
                CREATE INDEX IF NOT EXISTS idx_changes_created
                    ON changes(created_at);

                -- Tags table: normalized tags for search
                CREATE TABLE IF NOT EXISTS gadget_tags (
                    gadget_name TEXT NOT NULL,
                    gadget_version TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (gadget_name, gadget_version, tag),
                    FOREIGN KEY (gadget_name, gadget_version)
                        REFERENCES gadgets(name, version) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_tags_tag ON gadget_tags(tag);
            """)

    # =========================================================================
    # Gadget operations
    # =========================================================================

    def insert_gadget(
        self,
        manifest: Manifest,
        qasm: str,
        original_metadata: Optional[dict] = None,
    ) -> None:
        """Insert a new gadget into the database."""
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO gadgets (name, version, ir, qasm, manifest_json,
                                     original_metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.name,
                    manifest.version,
                    manifest.ir,
                    qasm,
                    manifest.model_dump_json(),
                    json.dumps(original_metadata) if original_metadata else None,
                    now,
                    now,
                ),
            )

            # Insert tags
            for tag in manifest.tags:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO gadget_tags (gadget_name, gadget_version, tag)
                    VALUES (?, ?, ?)
                    """,
                    (manifest.name, manifest.version, tag.lower()),
                )

    def get_gadget(self, name: str, version: str) -> Optional[dict]:
        """Get a gadget by name and version. Returns dict with manifest, qasm, metadata."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT name, version, qasm, manifest_json, original_metadata_json, created_at
                FROM gadgets WHERE name = ? AND version = ?
                """,
                (name, version),
            ).fetchone()

            if not row:
                return None

            return {
                "name": row["name"],
                "version": row["version"],
                "qasm": row["qasm"],
                "manifest": Manifest(**json.loads(row["manifest_json"])),
                "original_metadata": json.loads(row["original_metadata_json"])
                    if row["original_metadata_json"] else None,
                "created_at": row["created_at"],
            }

    def get_gadget_latest(self, name: str) -> Optional[dict]:
        """Get the latest version of a gadget by name."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT name, version, qasm, manifest_json, original_metadata_json, created_at
                FROM gadgets WHERE name = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (name,),
            ).fetchone()

            if not row:
                return None

            return {
                "name": row["name"],
                "version": row["version"],
                "qasm": row["qasm"],
                "manifest": Manifest(**json.loads(row["manifest_json"])),
                "original_metadata": json.loads(row["original_metadata_json"])
                    if row["original_metadata_json"] else None,
                "created_at": row["created_at"],
            }

    def get_all_gadgets(self) -> List[GadgetSummary]:
        """Get summaries of all gadgets."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT manifest_json FROM gadgets ORDER BY name, version
                """
            ).fetchall()

            summaries = []
            for row in rows:
                manifest = json.loads(row["manifest_json"])
                summaries.append(GadgetSummary(
                    name=manifest["name"],
                    version=manifest["version"],
                    description=manifest.get("description"),
                    tags=manifest.get("tags", []),
                ))
            return summaries

    def search_gadgets(self, query: str) -> List[GadgetSummary]:
        """Search gadgets by name, description, or tags."""
        query_lower = f"%{query.lower()}%"

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT g.manifest_json
                FROM gadgets g
                LEFT JOIN gadget_tags t ON g.name = t.gadget_name AND g.version = t.gadget_version
                WHERE LOWER(g.name) LIKE ?
                   OR LOWER(g.manifest_json) LIKE ?
                   OR t.tag LIKE ?
                ORDER BY g.name, g.version
                """,
                (query_lower, query_lower, query_lower),
            ).fetchall()

            summaries = []
            for row in rows:
                manifest = json.loads(row["manifest_json"])
                summaries.append(GadgetSummary(
                    name=manifest["name"],
                    version=manifest["version"],
                    description=manifest.get("description"),
                    tags=manifest.get("tags", []),
                ))
            return summaries

    def delete_gadget(self, name: str, version: str) -> bool:
        """Delete a gadget. Returns True if deleted, False if not found."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM gadgets WHERE name = ? AND version = ?",
                (name, version),
            )
            return cursor.rowcount > 0

    def gadget_exists(self, name: str, version: str) -> bool:
        """Check if a gadget exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM gadgets WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
            return row is not None

    # =========================================================================
    # Change operations
    # =========================================================================

    def add_change(
        self,
        gadget_name: str,
        gadget_version: str,
        qasm: str,
        metadata: Optional[dict] = None,
        metrics: Optional[MetricsHints] = None,
        diff_from_base: Optional[str] = None,
        source: ChangeSource = ChangeSource.manual,
        notes: Optional[str] = None,
    ) -> str:
        """Add a change to a gadget. Returns change ID."""
        change_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO changes (id, gadget_name, gadget_version, qasm,
                                     metadata_json, metrics_json, diff_from_base,
                                     source, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    change_id,
                    gadget_name,
                    gadget_version,
                    qasm,
                    json.dumps(metadata) if metadata else None,
                    metrics.model_dump_json() if metrics else None,
                    diff_from_base,
                    source.value,
                    notes,
                    now,
                ),
            )

        return change_id

    def get_change(self, change_id: str) -> Optional[Change]:
        """Get a specific change by ID."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, gadget_name, gadget_version, qasm, metadata_json,
                       metrics_json, diff_from_base, source, notes, created_at
                FROM changes WHERE id = ?
                """,
                (change_id,),
            ).fetchone()

            if not row:
                return None

            return self._row_to_change(row)

    def get_changes(self, gadget_name: str, gadget_version: str) -> List[Change]:
        """Get all changes for a gadget."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, gadget_name, gadget_version, qasm, metadata_json,
                       metrics_json, diff_from_base, source, notes, created_at
                FROM changes
                WHERE gadget_name = ? AND gadget_version = ?
                ORDER BY created_at ASC
                """,
                (gadget_name, gadget_version),
            ).fetchall()

            return [self._row_to_change(row) for row in rows]

    def get_change_bucket(self, gadget_name: str, gadget_version: str) -> Optional[ChangeBucket]:
        """Get the full change bucket for a gadget."""
        gadget = self.get_gadget(gadget_name, gadget_version)
        if not gadget:
            return None

        changes = self.get_changes(gadget_name, gadget_version)

        # Get timestamps
        with self._connect() as conn:
            row = conn.execute(
                "SELECT created_at, updated_at FROM gadgets WHERE name = ? AND version = ?",
                (gadget_name, gadget_version),
            ).fetchone()

        return ChangeBucket(
            gadget_name=gadget_name,
            gadget_version=gadget_version,
            base_qasm=gadget["qasm"],
            base_metadata=gadget["original_metadata"] or {},
            changes=changes,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def delete_change(self, change_id: str) -> bool:
        """Delete a change. Returns True if deleted."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM changes WHERE id = ?", (change_id,))
            return cursor.rowcount > 0

    def _row_to_change(self, row: sqlite3.Row) -> Change:
        """Convert a database row to a Change object."""
        return Change(
            id=row["id"],
            qasm=row["qasm"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            metrics=MetricsHints(**json.loads(row["metrics_json"]))
                if row["metrics_json"] else None,
            diff_from_base=row["diff_from_base"],
            source=ChangeSource(row["source"]),
            notes=row["notes"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # =========================================================================
    # Utility operations
    # =========================================================================

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._connect() as conn:
            gadget_count = conn.execute("SELECT COUNT(*) FROM gadgets").fetchone()[0]
            change_count = conn.execute("SELECT COUNT(*) FROM changes").fetchone()[0]
            tag_count = conn.execute("SELECT COUNT(DISTINCT tag) FROM gadget_tags").fetchone()[0]

            return {
                "gadgets": gadget_count,
                "changes": change_count,
                "unique_tags": tag_count,
            }


# Singleton instance
_db: Optional[Database] = None


def get_database() -> Database:
    """Get the database singleton."""
    global _db
    if _db is None:
        from ..config import settings
        db_path = settings.gadgets_dir.parent / "qgc.db"
        _db = Database(db_path)
    return _db


def reset_database():
    """Reset the database singleton (for testing)."""
    global _db
    _db = None
