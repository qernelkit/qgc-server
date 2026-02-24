def test_compile_simple_circuit(client):
    """POST /compile with a simple circuit returns compiled output and metrics."""
    request = {
        "initial_qasm": """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
""",
        "gadget_overrides": [],
        "mode": "baseline",
        "metrics": ["t_count", "cnot_count", "depth"],
    }

    response = client.post("/compile", json=request)
    assert response.status_code == 200

    result = response.json()
    assert "compiled_qasm" in result
    assert "report" in result
    assert "provenance" in result

    metrics = result["report"]["metrics"]
    assert "cnot_count" in metrics


def test_compile_with_gadget_override(client):
    """POST /compile replaces gadget markers with registered gadget QASM."""
    request = {
        "initial_qasm": """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;

// @gadget entangle
h q[0];
// @end_gadget

cx q[0], q[1];
""",
        "gadget_overrides": [
            {
                "target": "entangle",
                "replacement": {
                    "name": "bell-pair",
                    "version": "1.0.0",
                },
            }
        ],
        "mode": "baseline",
        "metrics": ["t_count", "cnot_count", "depth"],
    }

    response = client.post("/compile", json=request)
    assert response.status_code == 200

    result = response.json()
    assert "compiled_qasm" in result
    assert "entangle" in result["report"]["gadgets_used"]
    assert "compiled_at" in result["provenance"]
    assert "compiler_version" in result["provenance"]


def test_compile_with_inline_override(client):
    """POST /compile replaces gadget markers with inline QASM."""
    request = {
        "initial_qasm": """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;

// @gadget custom
h q[0];
// @end_gadget
""",
        "gadget_overrides": [
            {
                "target": "custom",
                "replacement": {
                    "qasm": "// Custom inline code\nx q[0];\ny q[1];",
                },
            }
        ],
        "mode": "baseline",
        "metrics": ["t_count"],
    }

    response = client.post("/compile", json=request)
    assert response.status_code == 200

    result = response.json()
    assert "Custom inline code" in result["compiled_qasm"]


def test_compile_missing_override(client):
    """POST /compile with unresolved gadget marker generates a warning."""
    request = {
        "initial_qasm": """OPENQASM 3.0;
// @gadget missing
h q[0];
// @end_gadget
""",
        "gadget_overrides": [],
        "mode": "baseline",
        "metrics": [],
    }

    response = client.post("/compile", json=request)
    assert response.status_code == 200

    result = response.json()
    assert len(result["report"]["warnings"]) > 0


def test_compile_not_under_gadgets_path(client):
    """POST /compile is at root, not under /gadgets."""
    response = client.post("/compile", json={
        "initial_qasm": "OPENQASM 3.0;",
        "gadget_overrides": [],
        "mode": "baseline",
        "metrics": [],
    })
    assert response.status_code == 200

    response = client.post("/gadgets/compile", json={
        "initial_qasm": "OPENQASM 3.0;",
        "gadget_overrides": [],
        "mode": "baseline",
        "metrics": [],
    })
    assert response.status_code in (404, 405)
