import pytest
from fastapi.testclient import TestClient

import qgc_server.services.database as db_module
import qgc_server.services.registry as registry_module
import qgc_server.services.compiler as compiler_module
from qgc_server.services.database import Database
from qgc_server.services.registry import RegistryService
from qgc_server.main import app


def _reset_singletons():
    """Reset all service singletons."""
    db_module._db = None
    registry_module._registry = None
    compiler_module._compiler = None


@pytest.fixture
def test_db(tmp_path):
    """Create an isolated test database."""
    _reset_singletons()

    db_path = tmp_path / "qgc_test.db"
    db = Database(db_path)
    db_module._db = db

    yield db

    _reset_singletons()


@pytest.fixture
def registry(test_db):
    """Registry service backed by the test database.

    Migrates builtin gadgets from disk into the temp DB on first access.
    """
    reg = RegistryService()
    registry_module._registry = reg
    return reg


@pytest.fixture
def client(registry):
    """Test client with fully isolated database and services."""
    with TestClient(app) as c:
        yield c
