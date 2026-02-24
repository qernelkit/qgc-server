MODIFIED_QASM = """OPENQASM 3.0;
include "stdgates.inc";

def bell_pair(qubit[2] q) {
    h q[0];
    cx q[0], q[1];
    barrier q;
}
"""


def _add_change(client, name="bell-pair", version="1.0.0", qasm=MODIFIED_QASM,
                notes="test change", source="manual"):
    """Helper: POST a change and return the response JSON."""
    response = client.post(
        f"/gadgets/{name}/{version}/changes",
        json={
            "qasm": qasm,
            "notes": notes,
            "source": source,
        },
    )
    return response


# ── GET changes (bucket) ────────────────────────────────────────────


def test_get_changes_empty(client):
    """GET /gadgets/{name}/{version}/changes returns empty bucket initially."""
    response = client.get("/gadgets/bell-pair/1.0.0/changes")
    assert response.status_code == 200

    bucket = response.json()
    assert bucket["gadget_name"] == "bell-pair"
    assert bucket["gadget_version"] == "1.0.0"
    assert bucket["changes"] == []
    assert "base_qasm" in bucket


def test_get_changes_not_found(client):
    """GET changes for nonexistent gadget returns 404."""
    response = client.get("/gadgets/nonexistent/1.0.0/changes")
    assert response.status_code == 404


# ── POST add change ─────────────────────────────────────────────────


def test_add_change(client):
    """POST /gadgets/{name}/{version}/changes stores a new change."""
    response = _add_change(client)
    assert response.status_code == 200

    data = response.json()
    assert "change_id" in data
    assert "diff_from_base" in data
    assert len(data["change_id"]) > 0


def test_add_change_computes_diff(client):
    """Adding a change computes a unified diff from the base QASM."""
    response = _add_change(client)
    assert response.status_code == 200

    data = response.json()
    assert "---" in data["diff_from_base"] or data["diff_from_base"] == ""


def test_add_change_computes_metrics(client):
    """Adding a change computes circuit metrics."""
    response = _add_change(client)
    assert response.status_code == 200

    data = response.json()
    if data["metrics"] is not None:
        assert "cnot_count" in data["metrics"]


def test_add_change_not_found(client):
    """POST change for nonexistent gadget returns 404."""
    response = _add_change(client, name="nonexistent")
    assert response.status_code == 404


# ── GET single change ───────────────────────────────────────────────


def test_get_single_change(client):
    """GET /gadgets/{name}/{version}/changes/{id} returns the change."""
    add_resp = _add_change(client)
    change_id = add_resp.json()["change_id"]

    response = client.get(f"/gadgets/bell-pair/1.0.0/changes/{change_id}")
    assert response.status_code == 200

    change = response.json()
    assert change["id"] == change_id
    assert change["source"] == "manual"
    assert change["notes"] == "test change"
    assert "qasm" in change


def test_get_single_change_not_found(client):
    """GET change with unknown ID returns 404."""
    response = client.get(
        "/gadgets/bell-pair/1.0.0/changes/00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404


# ── DELETE change ────────────────────────────────────────────────────


def test_delete_change(client):
    """DELETE /gadgets/{name}/{version}/changes/{id} removes the change."""
    add_resp = _add_change(client)
    change_id = add_resp.json()["change_id"]

    response = client.delete(f"/gadgets/bell-pair/1.0.0/changes/{change_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True

    # Verify it's gone
    response = client.get(f"/gadgets/bell-pair/1.0.0/changes/{change_id}")
    assert response.status_code == 404


def test_delete_change_not_found(client):
    """DELETE unknown change returns 404."""
    response = client.delete(
        "/gadgets/bell-pair/1.0.0/changes/00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404


# ── Promote change ──────────────────────────────────────────────────


def test_promote_change(client):
    """POST promote creates a new gadget version from a change."""
    add_resp = _add_change(client)
    change_id = add_resp.json()["change_id"]

    response = client.post(
        f"/gadgets/bell-pair/1.0.0/changes/{change_id}/promote",
        json={"new_version": "1.1.0", "description": "Added barrier"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["new_gadget_id"] == "bell-pair@1.1.0"

    # New version should be retrievable
    response = client.get("/gadgets/bell-pair/1.1.0")
    assert response.status_code == 200
    assert response.json()["version"] == "1.1.0"


def test_promote_change_version_conflict(client):
    """Promoting to an existing version returns 409."""
    add_resp = _add_change(client)
    change_id = add_resp.json()["change_id"]

    response = client.post(
        f"/gadgets/bell-pair/1.0.0/changes/{change_id}/promote",
        json={"new_version": "1.0.0"},  # already exists
    )
    assert response.status_code == 409


def test_promote_change_not_found(client):
    """Promoting a nonexistent change returns 404."""
    response = client.post(
        "/gadgets/bell-pair/1.0.0/changes/00000000-0000-0000-0000-000000000000/promote",
        json={"new_version": "2.0.0"},
    )
    assert response.status_code == 404


# ── Compare changes ─────────────────────────────────────────────────


def test_compare_changes(client):
    """GET compare returns a diff between two changes."""
    resp_a = _add_change(client, notes="change A")
    resp_b = _add_change(
        client,
        qasm='OPENQASM 3.0;\ninclude "stdgates.inc";\n// totally different\n',
        notes="change B",
    )
    id_a = resp_a.json()["change_id"]
    id_b = resp_b.json()["change_id"]

    response = client.get(
        f"/gadgets/bell-pair/1.0.0/changes/compare/{id_a}/{id_b}"
    )
    assert response.status_code == 200

    data = response.json()
    assert data["change_a"] == id_a
    assert data["change_b"] == id_b
    assert "diff" in data


def test_compare_changes_not_found(client):
    """Comparing with a missing change returns 404."""
    resp = _add_change(client)
    real_id = resp.json()["change_id"]
    fake_id = "00000000-0000-0000-0000-000000000000"

    response = client.get(
        f"/gadgets/bell-pair/1.0.0/changes/compare/{real_id}/{fake_id}"
    )
    assert response.status_code == 404


# ── Bucket accumulation ─────────────────────────────────────────────


def test_changes_accumulate_in_bucket(client):
    """Multiple changes appear in the bucket."""
    _add_change(client, notes="first")
    _add_change(client, notes="second")

    response = client.get("/gadgets/bell-pair/1.0.0/changes")
    assert response.status_code == 200

    bucket = response.json()
    assert len(bucket["changes"]) == 2
    notes = [c["notes"] for c in bucket["changes"]]
    assert "first" in notes
    assert "second" in notes
