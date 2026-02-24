def test_search_all_gadgets(client):
    """GET /gadgets returns all gadgets from the database."""
    response = client.get("/gadgets")
    assert response.status_code == 200

    gadgets = response.json()
    assert isinstance(gadgets, list)
    assert len(gadgets) >= 4

    for gadget in gadgets:
        assert "name" in gadget
        assert "version" in gadget


def test_search_with_query(client):
    """GET /gadgets?q=query filters results by tag/name/description."""
    response = client.get("/gadgets?q=entanglement")
    assert response.status_code == 200

    gadgets = response.json()
    assert isinstance(gadgets, list)

    names = [g["name"] for g in gadgets]
    assert "bell-pair" in names or "ghz-state" in names


def test_get_gadget_latest(client):
    """GET /gadgets/{name} returns the latest version manifest."""
    response = client.get("/gadgets/bell-pair")
    assert response.status_code == 200

    manifest = response.json()
    assert manifest["name"] == "bell-pair"
    assert manifest["version"] == "1.0.0"
    assert "interface" in manifest
    assert "hashes" in manifest


def test_get_gadget_specific_version(client):
    """GET /gadgets/{name}/{version} returns a specific version."""
    response = client.get("/gadgets/bell-pair/1.0.0")
    assert response.status_code == 200

    manifest = response.json()
    assert manifest["name"] == "bell-pair"
    assert manifest["version"] == "1.0.0"


def test_get_gadget_not_found(client):
    """GET /gadgets/{name} returns 404 for unknown gadget."""
    response = client.get("/gadgets/nonexistent-gadget")
    assert response.status_code == 404


def test_get_gadget_version_not_found(client):
    """GET /gadgets/{name}/{version} returns 404 for unknown version."""
    response = client.get("/gadgets/bell-pair/9.9.9")
    assert response.status_code == 404


def test_download_artifact(client):
    """GET /gadgets/{name}/{version}/artifact returns QASM content."""
    response = client.get("/gadgets/bell-pair/1.0.0/artifact")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    content = response.text
    assert "OPENQASM 3.0" in content
    assert "bell_pair" in content


def test_download_artifact_not_found(client):
    """GET /gadgets/{name}/{version}/artifact returns 404 when missing."""
    response = client.get("/gadgets/nonexistent/1.0.0/artifact")
    assert response.status_code == 404


def test_health_check(client):
    """GET /health returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_database_stats(client):
    """GET /stats returns database statistics."""
    response = client.get("/stats")
    assert response.status_code == 200

    stats = response.json()
    assert "gadgets" in stats
    assert stats["gadgets"] >= 4
    assert "changes" in stats
    assert "unique_tags" in stats
    assert "database_path" in stats
