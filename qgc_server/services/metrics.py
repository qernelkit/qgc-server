"""Metrics computation for quantum circuits.

Follows Qiskit's approach:
- depth: Number of layers (gates that can execute in parallel)
- count_ops: Gate counts by type

See: https://docs.quantum.ibm.com/api/qiskit/circuit
"""

import re
from typing import List, Dict
from collections import defaultdict

from ..models import CompileMetrics


def count_ops(qasm: str) -> Dict[str, int]:
    """Count gate operations in QASM code.

    Similar to Qiskit's QuantumCircuit.count_ops().
    Returns dict mapping gate names to counts.
    """
    counts: Dict[str, int] = defaultdict(int)

    # Remove comments
    lines = []
    for line in qasm.split("\n"):
        if "//" in line:
            line = line[:line.index("//")]
        lines.append(line)
    qasm_clean = "\n".join(lines)

    # Keywords to skip (not gates)
    skip = {"openqasm", "include", "qubit", "bit", "creg", "qreg",
            "gate", "def", "if", "for", "while", "return", "measure",
            "reset", "barrier", "input", "output", "const", "let"}

    # Match: gate_name(optional_params) operands;
    pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*[^;]+;")

    for match in pattern.finditer(qasm_clean):
        gate = match.group(1).lower()
        if gate not in skip:
            counts[gate] += 1

    return dict(counts)


def depth(qasm: str) -> int:
    """Calculate circuit depth.

    Similar to Qiskit's QuantumCircuit.depth().
    Depth = number of layers where gates can execute in parallel.
    """
    # Remove comments
    lines = []
    for line in qasm.split("\n"):
        if "//" in line:
            line = line[:line.index("//")]
        lines.append(line)
    qasm_clean = "\n".join(lines)

    skip = {"openqasm", "include", "qubit", "bit", "creg", "qreg",
            "gate", "def", "if", "for", "while", "return", "measure",
            "reset", "barrier", "input", "output", "const", "let"}

    # Track layer per qubit
    qubit_layer: Dict[str, int] = {}
    max_depth = 0

    # Match gates with operands
    pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*([^;]+);")

    for match in pattern.finditer(qasm_clean):
        gate = match.group(1).lower()
        if gate in skip:
            continue

        operands = [op.strip() for op in match.group(2).split(",")]

        # Find max layer among operands
        max_op_layer = 0
        for op in operands:
            # Normalize q[0] -> q0
            op_clean = re.sub(r"\[(\d+)\]", r"\1", op)
            max_op_layer = max(max_op_layer, qubit_layer.get(op_clean, 0))

        # This gate is in the next layer
        gate_layer = max_op_layer + 1

        # Update all operands
        for op in operands:
            op_clean = re.sub(r"\[(\d+)\]", r"\1", op)
            qubit_layer[op_clean] = gate_layer

        max_depth = max(max_depth, gate_layer)

    return max_depth


def compute_metrics(qasm: str, requested: List[str]) -> CompileMetrics:
    """Compute requested metrics from QASM code."""
    metrics = CompileMetrics()
    ops = None

    if "t_count" in requested:
        if ops is None:
            ops = count_ops(qasm)
        # T gates and T-like rotations
        metrics.t_count = sum(ops.get(g, 0) for g in ["t", "tdg", "rz", "rx", "ry", "u1", "u2", "u3"])

    if "cnot_count" in requested:
        if ops is None:
            ops = count_ops(qasm)
        # Two-qubit gates
        metrics.cnot_count = sum(ops.get(g, 0) for g in ["cx", "cnot", "cz", "cy", "swap", "iswap", "ecr", "rzx", "cp"])

    if "depth" in requested:
        metrics.depth = depth(qasm)

    return metrics


# Keep class for backwards compatibility
class MetricsService:
    """Service for computing circuit metrics."""

    @classmethod
    def compute_metrics(cls, qasm: str, requested: List[str]) -> CompileMetrics:
        return compute_metrics(qasm, requested)
