"""MCP server for the Quantum Gadget Catalog.

Exposes qgc_server functionality as MCP tools for use with
Claude Desktop, Claude Code, and other MCP clients.

Uses STDIO transport (standard for Claude Desktop / Claude Code).
"""

import difflib
import hashlib
import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from qgc_server.models import (
    CompileMode,
    CompileRequest,
    GadgetOverride,
    GadgetRef,
    Hashes,
    InlineQasm,
    Manifest,
    MetricsHints,
    ResolvedGadget,
)
from qgc_server.services.compiler import get_compiler
from qgc_server.services.database import get_database
from qgc_server.services.metrics import compute_metrics
from qgc_server.services.registry import get_registry

mcp = FastMCP("qgc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_summary(s) -> str:
    """Format a GadgetSummary as readable text."""
    tags = ", ".join(s.tags) if s.tags else "none"
    desc = s.description or "No description"
    return f"  {s.name}@{s.version}  —  {desc}  [tags: {tags}]"


def _compute_diff(base: str, modified: str) -> str:
    """Compute unified diff between base and modified QASM."""
    base_lines = base.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        base_lines,
        modified_lines,
        fromfile="base.qasm",
        tofile="modified.qasm",
    )
    return "".join(diff)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_gadgets(query: str = "") -> str:
    """Search the quantum gadget catalog.

    Returns a list of gadgets matching the query. If query is empty,
    returns all gadgets in the catalog.

    Args:
        query: Search term to match against gadget names, descriptions, and tags.
    """
    registry = get_registry()
    results = registry.search(query or None)

    if not results:
        return "No gadgets found." + (f' (query: "{query}")' if query else "")

    lines = [f"Found {len(results)} gadget(s):" + (f'  (query: "{query}")' if query else "")]
    for s in results:
        lines.append(_format_summary(s))
    return "\n".join(lines)


@mcp.tool()
def get_gadget(name: str, version: str = "") -> str:
    """Get the manifest and QASM code for a specific gadget.

    Args:
        name: Gadget name (e.g. "bell-pair").
        version: Specific version (e.g. "1.0.0"). Leave empty for latest.
    """
    registry = get_registry()
    manifest = registry.get_manifest(name, version or None)

    if manifest is None:
        return f"Gadget not found: {name}" + (f"@{version}" if version else "")

    # Get QASM artifact
    qasm = None
    artifact = registry.get_artifact(name, manifest.version)
    if artifact:
        qasm = artifact.decode("utf-8")

    parts = [
        "## Manifest",
        json.dumps(manifest.model_dump(), indent=2, default=str),
    ]

    if qasm:
        parts.append("\n## QASM Code")
        parts.append(f"```qasm\n{qasm}\n```")

    return "\n".join(parts)


@mcp.tool()
def compile_circuit(
    initial_qasm: str,
    gadget_overrides_json: str = "[]",
    mode: str = "baseline",
) -> str:
    """Compile a quantum circuit with gadget substitutions.

    Replaces @gadget markers in the QASM with actual gadget implementations.

    Args:
        initial_qasm: OpenQASM 3 circuit code containing @gadget markers.
        gadget_overrides_json: JSON array of overrides. Each object has:
            - "target": gadget marker name to replace
            - "replacement": either {"name": "...", "version": "..."} for a registry gadget,
              or {"qasm": "..."} for inline QASM.
            Example: [{"target": "bell-pair", "replacement": {"name": "bell-pair", "version": "1.0.0"}}]
        mode: Compilation mode — "baseline" or "optimize".
    """
    # Parse overrides
    try:
        overrides_raw = json.loads(gadget_overrides_json)
    except json.JSONDecodeError as e:
        return f"Error parsing gadget_overrides_json: {e}"

    overrides = []
    for ov in overrides_raw:
        repl = ov.get("replacement", {})
        if "name" in repl and "version" in repl:
            replacement = GadgetRef(name=repl["name"], version=repl["version"])
        elif "qasm" in repl:
            replacement = InlineQasm(qasm=repl["qasm"])
        else:
            return f"Invalid replacement in override for target '{ov.get('target', '?')}': must have name+version or qasm."
        overrides.append(GadgetOverride(target=ov["target"], replacement=replacement))

    compile_mode = CompileMode.optimize if mode == "optimize" else CompileMode.baseline

    request = CompileRequest(
        initial_qasm=initial_qasm,
        gadget_overrides=overrides,
        mode=compile_mode,
        metrics=["t_count", "cnot_count", "depth"],
    )

    compiler = get_compiler()
    response = compiler.compile(request)

    # Format response
    parts = [
        "## Compiled QASM",
        f"```qasm\n{response.compiled_qasm}\n```",
        "\n## Metrics",
        json.dumps(response.report.metrics.model_dump(), indent=2),
    ]

    if response.report.warnings:
        parts.append("\n## Warnings")
        for w in response.report.warnings:
            parts.append(f"  - {w}")

    if response.report.gadgets_used:
        parts.append("\n## Gadgets Used")
        for g in response.report.gadgets_used:
            parts.append(f"  - {g}")

    return "\n".join(parts)


@mcp.tool()
def resolve_dependencies(gadgets_json: str) -> str:
    """Resolve gadget dependency tree.

    Given a list of gadget references, resolves all direct and transitive
    dependencies, returning the full list with versions and hashes.

    Args:
        gadgets_json: JSON array of gadget references.
            Example: [{"name": "bell-pair", "version": "1.0.0"}]
    """
    try:
        gadgets_raw = json.loads(gadgets_json)
    except json.JSONDecodeError as e:
        return f"Error parsing gadgets_json: {e}"

    registry = get_registry()
    resolved: list[ResolvedGadget] = []

    for gref in gadgets_raw:
        name = gref.get("name", "")
        version = gref.get("version", "")
        manifest = registry.get_manifest(name, version or None)

        if manifest is None:
            return f"Gadget not found: {name}@{version}"

        resolved.append(
            ResolvedGadget(
                name=manifest.name,
                version=manifest.version,
                sha256=manifest.hashes.sha256,
            )
        )

        # Resolve transitive dependencies
        for dep in manifest.depends_on:
            dep_manifest = registry.get_manifest(dep.gadget)
            if dep_manifest and not any(
                r.name == dep_manifest.name and r.version == dep_manifest.version
                for r in resolved
            ):
                resolved.append(
                    ResolvedGadget(
                        name=dep_manifest.name,
                        version=dep_manifest.version,
                        sha256=dep_manifest.hashes.sha256,
                    )
                )

    if not resolved:
        return "No gadgets to resolve."

    lines = [f"Resolved {len(resolved)} gadget(s):"]
    for r in resolved:
        lines.append(f"  {r.name}@{r.version}  sha256:{r.sha256[:16]}...")
    return "\n".join(lines)


@mcp.tool()
async def ingest_gadget(qasm: str, metadata_json: str = "{}") -> str:
    """Ingest a new quantum gadget from QASM code and optional metadata.

    Uses AI extraction (Ollama) to derive a structured manifest from the
    circuit and metadata. If Ollama is not configured, the extraction will
    fail — provide rich metadata to compensate.

    Args:
        qasm: OpenQASM 3 circuit code for the gadget.
        metadata_json: JSON object with freeform metadata (name, description, tags, etc.).
            Example: {"name": "my-gadget", "description": "A custom gadget", "tags": ["test"]}
    """
    from qgc_server.routes.ingest import extract_with_ollama, compute_sha256

    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as e:
        return f"Error parsing metadata_json: {e}"

    try:
        extracted = await extract_with_ollama(qasm, metadata)
    except Exception as e:
        return f"AI extraction failed: {e}. You may need to set OLLAMA_API_KEY."

    sha256 = compute_sha256(qasm.encode("utf-8"))

    from qgc_server.models import GadgetInterface

    manifest = Manifest(
        name=extracted.get("name", "unnamed"),
        version=extracted.get("version", "1.0.0"),
        ir="openqasm3",
        interface=GadgetInterface(
            input_qubits=extracted.get("input_qubits", 0),
            output_qubits=extracted.get("output_qubits", 0),
            classical_bits=extracted.get("classical_bits", 0),
        ),
        depends_on=[],
        metrics_hints=MetricsHints(
            t_count=extracted.get("t_count"),
            cnot_count=extracted.get("cnot_count"),
            depth=extracted.get("depth"),
        ),
        hashes=Hashes(sha256=sha256),
        description=extracted.get("description"),
        tags=extracted.get("tags", []),
    )

    registry = get_registry()
    original_input = {"input_mode": "mcp_ingest", "metadata": metadata}
    registry.insert_gadget(manifest, qasm, original_input)

    gadget_id = f"{manifest.name}@{manifest.version}"
    return (
        f"Gadget ingested successfully!\n\n"
        f"ID: {gadget_id}\n"
        f"## Manifest\n{json.dumps(manifest.model_dump(), indent=2, default=str)}\n\n"
        f"## AI Extraction\n{json.dumps(extracted, indent=2, default=str)}"
    )


@mcp.tool()
def list_changes(name: str, version: str) -> str:
    """List all changes (modifications) for a gadget version.

    Returns the change bucket including base QASM and all recorded changes
    with their diffs and metrics.

    Args:
        name: Gadget name (e.g. "bell-pair").
        version: Gadget version (e.g. "1.0.0").
    """
    db = get_database()
    bucket = db.get_change_bucket(name, version)

    if bucket is None:
        return f"Gadget not found: {name}@{version}"

    if not bucket.changes:
        return f"No changes recorded for {name}@{version}."

    lines = [
        f"Changes for {name}@{version}  ({len(bucket.changes)} change(s)):",
        f"Base QASM length: {len(bucket.base_qasm)} chars",
        "",
    ]

    for i, change in enumerate(bucket.changes, 1):
        lines.append(f"### Change {i}: {change.id}")
        lines.append(f"  Source: {change.source.value}")
        lines.append(f"  Created: {change.created_at.isoformat()}")
        if change.notes:
            lines.append(f"  Notes: {change.notes}")
        if change.metrics:
            lines.append(f"  Metrics: {json.dumps(change.metrics.model_dump(), indent=2)}")
        if change.diff_from_base:
            lines.append(f"  Diff:\n{change.diff_from_base}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def add_change(
    name: str,
    version: str,
    qasm: str,
    source: str = "manual",
    notes: str = "",
) -> str:
    """Add a QASM modification to an existing gadget's change bucket.

    Stores the full QASM snapshot, computes a diff from the base, and
    records circuit metrics.

    Args:
        name: Gadget name (e.g. "bell-pair").
        version: Gadget version (e.g. "1.0.0").
        qasm: New/modified OpenQASM 3 code (full snapshot).
        source: Change source — "manual", "ai", or "optimizer".
        notes: Optional notes describing the change.
    """
    from qgc_server.models import ChangeSource

    registry = get_registry()
    gadget = registry.get_gadget(name, version)
    if not gadget:
        return f"Gadget not found: {name}@{version}"

    # Compute diff
    diff = _compute_diff(gadget.qasm, qasm)

    # Compute metrics
    try:
        computed = compute_metrics(qasm, ["t_count", "cnot_count", "depth"])
        metrics = MetricsHints(
            t_count=computed.t_count,
            cnot_count=computed.cnot_count,
            depth=computed.depth,
        )
    except Exception:
        metrics = None

    # Parse source enum
    try:
        change_source = ChangeSource(source)
    except ValueError:
        change_source = ChangeSource.manual

    db = get_database()
    change_id = db.add_change(
        gadget_name=name,
        gadget_version=version,
        qasm=qasm,
        metrics=metrics,
        diff_from_base=diff,
        source=change_source,
        notes=notes or None,
    )

    parts = [
        f"Change added successfully!",
        f"Change ID: {change_id}",
    ]

    if metrics:
        parts.append(f"Metrics: {json.dumps(metrics.model_dump(), indent=2)}")

    if diff:
        parts.append(f"\nDiff from base:\n{diff}")
    else:
        parts.append("\nNo diff (QASM unchanged from base).")

    return "\n".join(parts)


@mcp.tool()
def promote_change(
    name: str,
    version: str,
    change_id: str,
    new_version: str,
    description: str = "",
) -> str:
    """Promote a change to become a new gadget version.

    Creates a new version of the gadget using the QASM from the specified
    change, preserving provenance information.

    Args:
        name: Gadget name (e.g. "bell-pair").
        version: Current gadget version the change belongs to (e.g. "1.0.0").
        change_id: UUID of the change to promote.
        new_version: Version string for the new gadget (e.g. "1.1.0").
        description: Optional description for the new version.
    """
    db = get_database()
    registry = get_registry()

    change = db.get_change(change_id)
    if change is None:
        return f"Change not found: {change_id}"

    if registry.gadget_exists(name, new_version):
        return f"Gadget {name}@{new_version} already exists. Choose a different version."

    original = registry.get_gadget(name, version)
    if not original:
        return f"Original gadget not found: {name}@{version}"

    sha256 = hashlib.sha256(change.qasm.encode("utf-8")).hexdigest()

    new_manifest = Manifest(
        name=name,
        version=new_version,
        ir=original.manifest.ir,
        interface=original.manifest.interface,
        depends_on=original.manifest.depends_on,
        metrics_hints=change.metrics or original.manifest.metrics_hints,
        hashes=Hashes(sha256=sha256),
        description=description or original.manifest.description,
        tags=original.manifest.tags,
    )

    provenance = {
        "promoted_from": f"{name}@{version}",
        "change_id": change_id,
        "original_metadata": change.metadata,
        "notes": change.notes,
    }

    registry.insert_gadget(new_manifest, change.qasm, provenance)

    new_gadget_id = f"{name}@{new_version}"
    return (
        f"Change promoted successfully!\n\n"
        f"New gadget: {new_gadget_id}\n"
        f"## Manifest\n{json.dumps(new_manifest.model_dump(), indent=2, default=str)}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Run the QGC MCP server on STDIO transport."""
    # Initialize the registry (loads builtins, runs file-to-DB migration)
    get_registry()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
