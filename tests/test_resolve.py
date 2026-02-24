def test_resolve_single_gadget(client):
    """POST /resolve with a single gadget returns it with a sha256 hash."""
    request = {
        "gadgets": [
            {"name": "bell-pair", "version": "1.0.0"}
        ]
    }

    response = client.post("/resolve", json=request)
    assert response.status_code == 200

    result = response.json()
    assert "resolved" in result
    assert len(result["resolved"]) >= 1

    resolved = result["resolved"][0]
    assert resolved["name"] == "bell-pair"
    assert resolved["version"] == "1.0.0"
    assert "sha256" in resolved


def test_resolve_multiple_gadgets(client):
    """POST /resolve with multiple gadgets returns all of them."""
    request = {
        "gadgets": [
            {"name": "bell-pair", "version": "1.0.0"},
            {"name": "ghz-state", "version": "1.0.0"},
        ]
    }

    response = client.post("/resolve", json=request)
    assert response.status_code == 200

    result = response.json()
    assert len(result["resolved"]) >= 2

    names = [g["name"] for g in result["resolved"]]
    assert "bell-pair" in names
    assert "ghz-state" in names


def test_resolve_nonexistent_gadget(client):
    """POST /resolve with unknown gadget returns 404."""
    request = {
        "gadgets": [
            {"name": "nonexistent-gadget", "version": "1.0.0"}
        ]
    }

    response = client.post("/resolve", json=request)
    assert response.status_code == 404


def test_resolve_not_under_gadgets_path(client):
    """POST /resolve is at root, not under /gadgets."""
    response = client.post("/resolve", json={
        "gadgets": [{"name": "bell-pair", "version": "1.0.0"}]
    })
    assert response.status_code == 200

    response = client.post("/gadgets/resolve", json={
        "gadgets": [{"name": "bell-pair", "version": "1.0.0"}]
    })
    assert response.status_code in (404, 405)


def test_resolve_empty_list(client):
    """POST /resolve with empty gadgets list returns empty resolved list."""
    request = {"gadgets": []}

    response = client.post("/resolve", json=request)
    assert response.status_code == 200

    result = response.json()
    assert result["resolved"] == []
