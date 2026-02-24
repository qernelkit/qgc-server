"""Registry service for gadget storage and indexing.

Uses SQLite database for persistent storage.
"""

import json
from typing import List, Optional

from ..models import GadgetSummary, Manifest
from ..config import settings
from .database import get_database


class GadgetRecord:
    """A gadget record from the database."""

    def __init__(self, name: str, version: str, qasm: str, manifest: Manifest,
                 original_metadata: Optional[dict] = None):
        self.name = name
        self.version = version
        self.qasm = qasm
        self.manifest = manifest
        self.original_metadata = original_metadata


class RegistryService:
    """Service for managing gadget registry using SQLite."""

    def __init__(self):
        self.db = get_database()
        # Migrate any existing file-based gadgets on startup
        self._migrate_from_files()

    def _migrate_from_files(self) -> None:
        """Migrate existing file-based gadgets to the database."""
        gadgets_dir = settings.gadgets_dir
        if not gadgets_dir.exists():
            return

        migrated = 0
        for gadget_dir in gadgets_dir.iterdir():
            if not gadget_dir.is_dir():
                continue

            gadget_name = gadget_dir.name

            for version_dir in gadget_dir.iterdir():
                if not version_dir.is_dir():
                    continue

                version = version_dir.name
                manifest_path = version_dir / "gadget.json"
                artifact_path = version_dir / "main.qasm"

                # Skip if already in database
                if self.db.gadget_exists(gadget_name, version):
                    continue

                if manifest_path.exists() and artifact_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest_data = json.load(f)
                        manifest = Manifest(**manifest_data)

                        with open(artifact_path) as f:
                            qasm = f.read()

                        # Load original metadata if available
                        original_metadata = None
                        meta_path = version_dir / "original_metadata.json"
                        if meta_path.exists():
                            with open(meta_path) as f:
                                original_metadata = json.load(f)

                        # Insert into database
                        self.db.insert_gadget(manifest, qasm, original_metadata)
                        migrated += 1

                    except Exception as e:
                        print(f"Warning: Failed to migrate gadget {gadget_name}@{version}: {e}")

        if migrated > 0:
            print(f"Migrated {migrated} gadgets from files to database")

    def search(self, query: Optional[str] = None) -> List[GadgetSummary]:
        """Search for gadgets."""
        if query:
            return self.db.search_gadgets(query)
        return self.db.get_all_gadgets()

    def get_manifest(self, name: str, version: Optional[str] = None) -> Optional[Manifest]:
        """Get manifest for a gadget."""
        if version is None:
            gadget = self.db.get_gadget_latest(name)
        else:
            gadget = self.db.get_gadget(name, version)

        return gadget["manifest"] if gadget else None

    def get_artifact(self, name: str, version: str) -> Optional[bytes]:
        """Get artifact (QASM code) for a gadget."""
        gadget = self.db.get_gadget(name, version)
        if gadget:
            return gadget["qasm"].encode("utf-8")
        return None

    def get_gadget(self, name: str, version: str) -> Optional[GadgetRecord]:
        """Get full gadget record."""
        gadget = self.db.get_gadget(name, version)
        if not gadget:
            return None

        return GadgetRecord(
            name=gadget["name"],
            version=gadget["version"],
            qasm=gadget["qasm"],
            manifest=gadget["manifest"],
            original_metadata=gadget["original_metadata"],
        )

    def get_all_gadgets(self) -> List[Manifest]:
        """Get all gadget manifests."""
        summaries = self.db.get_all_gadgets()
        # For full manifests, we need to fetch each one
        manifests = []
        for s in summaries:
            gadget = self.db.get_gadget(s.name, s.version)
            if gadget:
                manifests.append(gadget["manifest"])
        return manifests

    def insert_gadget(self, manifest: Manifest, qasm: str,
                      original_metadata: Optional[dict] = None) -> None:
        """Insert a new gadget into the registry."""
        self.db.insert_gadget(manifest, qasm, original_metadata)

    def delete_gadget(self, name: str, version: str) -> bool:
        """Delete a gadget from the registry."""
        return self.db.delete_gadget(name, version)

    def gadget_exists(self, name: str, version: str) -> bool:
        """Check if a gadget exists."""
        return self.db.gadget_exists(name, version)


# Global singleton
_registry: Optional[RegistryService] = None


def get_registry() -> RegistryService:
    """Get the global registry service instance."""
    global _registry
    if _registry is None:
        _registry = RegistryService()
    return _registry


def reset_registry():
    """Reset the registry singleton (for testing)."""
    global _registry
    _registry = None
