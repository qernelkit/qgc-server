"""MCP server for the Quantum Gadget Catalog.

Exposes qgc_server functionality as MCP tools for use with
Claude Desktop, Claude Code, and other MCP clients.

Uses STDIO transport (standard for Claude Desktop / Claude Code).
"""

import asyncio
import difflib
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
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
def ingest_gadget(qasm: str, manifest_json: str = "") -> str:
    """Ingest a new quantum gadget into the QGC catalog.

    YOU (the calling LLM) are the intelligence here. Analyze the QASM circuit,
    then construct and pass in a manifest. The tool validates, auto-computes
    what you leave out (sha256 hash, circuit metrics), and stores the gadget.

    ## Manifest Schema

    Pass a JSON object with these fields:

        {
          "name": "lowercase-hyphenated-name",   // REQUIRED. e.g. "zz-interaction"
          "version": "1.0.0",                     // semver, default "1.0.0"
          "description": "Human-readable summary", // what the circuit does
          "tags": ["qaoa", "ising", "2-qubit"],   // searchable tags
          "input_qubits": 2,                      // REQUIRED. count from the QASM
          "output_qubits": 2,                     // defaults to input_qubits
          "classical_bits": 0,                    // measurement bits, default 0
          "depends_on": [                         // other gadgets this one uses
            {"gadget": "bell-pair", "version_req": ">=1.0.0"}
          ],
          "t_count": 0,                           // auto-computed if omitted
          "cnot_count": 4,                        // auto-computed if omitted
          "depth": 6                              // auto-computed if omitted
        }

    ## Example (real gadget from catalog)

        manifest_json: {
          "name": "bell-pair",
          "version": "1.0.0",
          "description": "Create an entangled Bell pair |00> + |11>",
          "tags": ["entanglement", "basic"],
          "input_qubits": 2,
          "output_qubits": 2,
          "classical_bits": 0,
          "t_count": 0,
          "cnot_count": 1,
          "depth": 2
        }

    ## What you should do

    1. Read the QASM — count qubit registers, identify the circuit's purpose
    2. Pick a good name (lowercase, hyphenated, descriptive)
    3. Write a clear description of what the circuit implements
    4. Choose relevant tags for searchability
    5. Count or estimate qubits, gate metrics, depth
    6. Pass the manifest JSON — the tool handles hashing and storage

    Args:
        qasm: OpenQASM 3 circuit code for the gadget.
        manifest_json: JSON manifest object (see schema above).
    """
    # Parse manifest
    if not manifest_json:
        return (
            "No manifest provided. Analyze the QASM and construct a manifest JSON.\n\n"
            "At minimum provide: name, input_qubits, description, tags.\n"
            "See tool description for full schema and example."
        )

    try:
        m = json.loads(manifest_json)
    except json.JSONDecodeError as e:
        return f"Error parsing manifest_json: {e}"

    # Validate required fields
    if not m.get("name"):
        return "Manifest missing required field: `name`"
    if not isinstance(m.get("input_qubits"), int):
        return "Manifest missing required field: `input_qubits` (int)"

    # Auto-compute metrics from QASM if not provided
    t_count = m.get("t_count")
    cnot_count = m.get("cnot_count")
    depth = m.get("depth")
    if t_count is None or cnot_count is None or depth is None:
        try:
            computed = compute_metrics(qasm, ["t_count", "cnot_count", "depth"])
            if t_count is None:
                t_count = computed.t_count
            if cnot_count is None:
                cnot_count = computed.cnot_count
            if depth is None:
                depth = computed.depth
        except Exception:
            pass  # Metrics are optional hints

    # Auto-compute hash
    sha256 = hashlib.sha256(qasm.encode("utf-8")).hexdigest()

    # Build dependencies
    from qgc_server.models import GadgetInterface, GadgetDependency
    deps = []
    for dep in m.get("depends_on", []):
        if isinstance(dep, dict) and dep.get("gadget"):
            deps.append(GadgetDependency(
                gadget=dep["gadget"],
                version_req=dep.get("version_req", ">=0.0.0"),
            ))

    # Build manifest
    manifest = Manifest(
        name=m["name"],
        version=m.get("version", "1.0.0"),
        ir="openqasm3",
        interface=GadgetInterface(
            input_qubits=m["input_qubits"],
            output_qubits=m.get("output_qubits", m["input_qubits"]),
            classical_bits=m.get("classical_bits", 0),
        ),
        depends_on=deps,
        metrics_hints=MetricsHints(
            t_count=t_count,
            cnot_count=cnot_count,
            depth=depth,
        ),
        hashes=Hashes(sha256=sha256),
        description=m.get("description"),
        tags=m.get("tags", []),
    )

    # Check for duplicates
    registry = get_registry()
    if registry.gadget_exists(manifest.name, manifest.version):
        return (
            f"Gadget {manifest.name}@{manifest.version} already exists.\n"
            f"Use a different version (e.g. bump to next minor/patch) or "
            f"use `add_change` to track a modification to the existing version."
        )

    # Store
    provenance = {"input_mode": "mcp_ingest", "manifest_provided_by": "llm_caller"}
    registry.insert_gadget(manifest, qasm, provenance)

    gadget_id = f"{manifest.name}@{manifest.version}"
    return (
        f"Gadget ingested successfully!\n\n"
        f"**{gadget_id}**\n\n"
        f"## Manifest\n```json\n{json.dumps(manifest.model_dump(), indent=2, default=str)}\n```"
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
# Paper analysis tools
# ---------------------------------------------------------------------------

_PAPER_OUTPUT_BASE = Path("/tmp/qgc_papers")

# Quantum computing concepts that may appear in papers but are NOT (yet) gadgets.
# Used to detect "gaps" in the catalog.
_EXTRA_QUANTUM_KEYWORDS: dict[str, list[str]] = {
    "error_correction": ["error correction", "error-correction", "qec", "fault tolerant", "fault-tolerant"],
    "syndrome": ["syndrome extraction", "syndrome measurement", "syndrome decoding"],
    "stabilizer": ["stabilizer code", "stabilizer state", "stabilizer formalism"],
    "toffoli": ["toffoli", "ccx", "cc-not"],
    "grover": ["grover", "amplitude amplification"],
    "shor": ["shor's algorithm", "shor algorithm", "period finding"],
    "teleportation": ["teleportation", "quantum teleportation"],
    "swap": ["swap gate", "swap network", "fredkin"],
    "controlled_z": ["controlled-z", "cz gate", "controlled phase"],
    "surface_code": ["surface code", "planar code", "rotated surface code"],
    "magic_state": ["magic state", "magic state distillation", "t-state"],
    "variational": ["variational", "vqe", "qaoa", "ansatz"],
    "hamiltonian_simulation": ["hamiltonian simulation", "trotterization", "trotter"],
}


def _build_gadget_keyword_map(gadgets: list) -> dict[str, list[str]]:
    """Build keyword→gadget_id map from gadget manifests.

    For each gadget, keywords come from:
    - Name parts (split on '-')
    - Tags
    - Description words (3+ chars, lowercased)
    """
    keyword_map: dict[str, list[str]] = {}  # keyword → [gadget_ids]

    for g in gadgets:
        gadget_id = f"{g.name}@{g.version}"

        # Keywords from name parts
        name_parts = g.name.lower().replace("_", "-").split("-")
        # Also add full name as keyword
        keywords = set(name_parts + [g.name.lower()])

        # Keywords from tags
        if g.tags:
            for tag in g.tags:
                keywords.add(tag.lower())

        # Keywords from description
        if g.description:
            for word in re.findall(r"[a-zA-Z]{3,}", g.description.lower()):
                keywords.add(word)

        # Remove stop words
        stop = {"the", "and", "for", "with", "from", "that", "this", "qubit", "gate"}
        keywords -= stop

        for kw in keywords:
            keyword_map.setdefault(kw, [])
            if gadget_id not in keyword_map[kw]:
                keyword_map[kw].append(gadget_id)

    return keyword_map


def _scan_paper_for_matches(
    paper_text: str,
    keyword_map: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Scan paper text for gadget keyword matches.

    Returns gadget_id → [matched excerpts] mapping.
    """
    text_lower = paper_text.lower()
    paragraphs = re.split(r"\n\s*\n", paper_text)

    matches: dict[str, list[str]] = {}  # gadget_id → [excerpts]

    for kw, gadget_ids in keyword_map.items():
        if len(kw) < 3:
            continue
        # Use word boundary matching for short keywords to avoid false positives
        if len(kw) <= 4:
            pattern = rf"\b{re.escape(kw)}\b"
            if not re.search(pattern, text_lower):
                continue
        elif kw not in text_lower:
            continue

        # Find paragraphs containing this keyword
        for para in paragraphs:
            if kw in para.lower():
                excerpt = para.strip()[:200]
                for gid in gadget_ids:
                    matches.setdefault(gid, [])
                    if len(matches[gid]) < 5 and excerpt not in matches[gid]:
                        matches[gid].append(excerpt)

    return matches


def _detect_gaps(paper_text: str, matched_gadget_names: set[str]) -> list[dict]:
    """Detect quantum concepts in the paper not covered by matched gadgets."""
    text_lower = paper_text.lower()
    gaps = []

    for concept, phrases in _EXTRA_QUANTUM_KEYWORDS.items():
        found_phrases = [p for p in phrases if p in text_lower]
        if found_phrases:
            # Check it's not already covered by a matched gadget
            if not any(concept.replace("_", "-") in name for name in matched_gadget_names):
                gaps.append({
                    "concept": concept.replace("_", " "),
                    "matched_phrases": found_phrases,
                })

    return gaps


def _parse_content_list(content_list_path: Path) -> dict:
    """Parse a MineRU _content_list.json and extract tables, figures, and equations.

    Returns a dict with:
      - tables: list of {page, caption, html, img_path}
      - equations: list of {page, latex, img_path}
      - images: list of all img_paths referenced
    """
    result: dict = {"tables": [], "equations": [], "images": []}

    if not content_list_path.exists():
        return result

    try:
        data = json.loads(content_list_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return result

    auto_dir = content_list_path.parent

    for entry in data:
        entry_type = entry.get("type", "")
        img_path = entry.get("img_path", "")
        page_idx = entry.get("page_idx", 0)

        # Track all referenced images
        if img_path:
            abs_img = auto_dir / img_path
            if abs_img.exists():
                result["images"].append(str(abs_img))

        if entry_type == "table":
            table_info: dict = {
                "page": page_idx,
                "html": entry.get("table_body", ""),
                "caption": " ".join(entry.get("table_caption", [])),
                "footnote": " ".join(entry.get("table_footnote", [])),
            }
            if img_path:
                abs_img = auto_dir / img_path
                table_info["img_path"] = str(abs_img) if abs_img.exists() else img_path
            result["tables"].append(table_info)

        elif entry_type == "equation":
            eq_info: dict = {
                "page": page_idx,
                "latex": entry.get("text", ""),
            }
            if img_path:
                abs_img = auto_dir / img_path
                eq_info["img_path"] = str(abs_img) if abs_img.exists() else img_path
            result["equations"].append(eq_info)

    return result


def _build_extraction_result(
    stem: str,
    pdf_path: str,
    output_dir: Path,
    auto_dir: Path,
    md_path: Path,
    cached: bool = False,
) -> str:
    """Build the full extraction result including markdown, tables, and image inventory."""
    md_content = md_path.read_text(encoding="utf-8")
    truncated = len(md_content) > 50_000
    content = md_content[:50_000]

    # Parse content_list.json for structured data (tables, equations, images)
    content_list_path = auto_dir / f"{stem}_content_list.json"
    structured = _parse_content_list(content_list_path)

    # Also discover any images not referenced in content_list
    images_dir = auto_dir / "images"
    all_image_files: list[str] = []
    if images_dir.is_dir():
        all_image_files = sorted(str(p) for p in images_dir.iterdir() if p.is_file())

    # Build result
    lines = [f"## Extracted Paper: {stem}\n"]
    lines.append(f"**Source**: `{pdf_path}`")
    lines.append(f"**Output dir**: `{output_dir}`")
    lines.append(f"**Markdown path**: `{md_path}`")
    if cached:
        lines.append("**Cached**: yes (skipped re-extraction)")

    # Image inventory
    lines.append(f"\n### Extracted Assets")
    lines.append(f"- **Images**: {len(all_image_files)} file(s) in `{images_dir}`")
    lines.append(f"- **Tables**: {len(structured['tables'])}")
    lines.append(f"- **Equations**: {len(structured['equations'])}")

    # Tables section — include HTML body so LLM can reason about content
    if structured["tables"]:
        lines.append("\n### Tables\n")
        for i, tbl in enumerate(structured["tables"], 1):
            lines.append(f"**Table {i}** (page {tbl['page'] + 1})")
            if tbl.get("caption"):
                lines.append(f"  Caption: {tbl['caption']}")
            if tbl.get("img_path"):
                lines.append(f"  Image: `{tbl['img_path']}`")
            if tbl.get("html"):
                # Include HTML for text-based analysis; cap per-table size
                html_preview = tbl["html"][:2000]
                lines.append(f"  Content:\n```html\n{html_preview}\n```")
                if len(tbl["html"]) > 2000:
                    lines.append("  *(table HTML truncated)*")
            lines.append("")

    # Image file listing (so caller knows what's available)
    if all_image_files:
        lines.append("\n### Image Files\n")
        for img in all_image_files:
            lines.append(f"  - `{img}`")
        lines.append("")

    # Markdown content
    lines.append("\n---\n### Markdown Content\n")
    lines.append(content)
    if truncated:
        lines.append("\n\n---\n*[Markdown truncated at 50,000 characters]*")

    return "\n".join(lines)


async def _download_pdf(url: str, dest: Path) -> Optional[str]:
    """Download a PDF from a URL. Returns error string on failure, None on success."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not dest.suffix == ".pdf":
                return f"URL did not return a PDF (content-type: {content_type})"
            dest.write_bytes(resp.content)
    except httpx.HTTPStatusError as e:
        return f"HTTP {e.response.status_code} downloading {url}: {e}"
    except httpx.ConnectError as e:
        return f"Connection failed for {url}: {e}"
    except Exception as e:
        return f"Failed to download {url}: {e}"
    return None


# Track background MineRU jobs: stem → {"process": Popen, "pdf": str, "source": str}
_extraction_jobs: dict[str, dict] = {}


def _resolve_pdf_stem(pdf_path: str) -> str:
    """Get the output stem for a PDF path."""
    return Path(pdf_path).stem


def _start_mineru_background(pdf_path: str, output_dir: Path) -> subprocess.Popen:
    """Start MineRU as a background process."""
    log_file = output_dir / "_mineru.log"
    log_handle = open(log_file, "w")
    proc = subprocess.Popen(
        ["mineru", "-p", pdf_path, "-o", str(output_dir), "-b", "pipeline"],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return proc


@mcp.tool()
async def extract_paper(pdf_path: str = "", url: str = "") -> str:
    """Extract content from a quantum computing paper (PDF) using MineRU.

    Accepts either a local file path or a URL. If a URL is provided, the PDF
    is downloaded first (the MCP server has network access even when the
    calling environment does not).

    MineRU extraction is **non-blocking**: it starts in the background and
    returns immediately. Call this tool again with the same arguments to check
    if extraction is complete. Once done, returns the full parsed content.

    ## Lifecycle

    1. First call → downloads PDF (if URL), starts MineRU, returns "extraction started"
    2. Subsequent calls → checks if MineRU finished:
       - Still running → returns progress status
       - Done → returns full markdown + tables + images
    3. Already cached → returns results instantly (no re-extraction)

    The output (when ready) includes:
    - Full markdown text of the paper with LaTeX formulas
    - Table inventory with HTML content and image paths
    - Image file listing (extracted figures, table renders, equation renders)
    - Equation inventory with LaTeX source

    Args:
        pdf_path: Absolute path to a local PDF file.
        url: URL to a PDF paper (e.g. arXiv, journal site). Will be downloaded
             to a temp directory automatically.
    """
    # --- Resolve PDF source ---
    if url and not pdf_path:
        # Derive a filename from the URL
        from urllib.parse import urlparse, unquote
        parsed = urlparse(url)
        url_filename = Path(unquote(parsed.path)).name
        if not url_filename.endswith(".pdf"):
            # Use a hash-based name for URLs without .pdf extension
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
            url_filename = f"paper_{url_hash}.pdf"

        download_dest = _PAPER_OUTPUT_BASE / "_downloads" / url_filename
        # Check if already downloaded
        if not download_dest.exists():
            err = await _download_pdf(url, download_dest)
            if err:
                return (
                    f"Failed to download PDF from URL.\n\n"
                    f"**Error**: {err}\n\n"
                    f"**Tip**: If the URL requires authentication or special access, "
                    f"download the PDF manually and pass the local path via `pdf_path` instead."
                )
        pdf_path = str(download_dest)

    if not pdf_path:
        return "Provide either `pdf_path` (local file) or `url` (remote PDF)."

    pdf = Path(pdf_path)
    if not pdf.exists():
        return f"PDF not found: {pdf_path}"
    if not pdf.suffix.lower() == ".pdf":
        return f"File does not appear to be a PDF: {pdf_path}"

    stem = pdf.stem
    output_dir = _PAPER_OUTPUT_BASE / stem
    auto_dir = output_dir / stem / "auto"
    md_path = auto_dir / f"{stem}.md"

    source_label = url or pdf_path

    # --- Check cache: if already extracted, return immediately ---
    if md_path.exists():
        # Clean up any stale job reference
        _extraction_jobs.pop(stem, None)
        return _build_extraction_result(stem, source_label, output_dir, auto_dir, md_path, cached=True)

    # --- Check if a background job is already running for this paper ---
    if stem in _extraction_jobs:
        job = _extraction_jobs[stem]
        proc: subprocess.Popen = job["process"]
        rc = proc.poll()

        if rc is None:
            # Still running
            log_file = output_dir / "_mineru.log"
            log_tail = ""
            if log_file.exists():
                try:
                    log_text = log_file.read_text(encoding="utf-8", errors="replace")
                    log_lines = log_text.strip().splitlines()
                    log_tail = "\n".join(log_lines[-5:])
                except OSError:
                    pass
            return (
                f"MineRU extraction is **in progress** for `{stem}`.\n\n"
                f"Call `extract_paper` again with the same arguments to check status.\n\n"
                f"**Recent log output**:\n```\n{log_tail or '(no output yet)'}\n```"
            )

        # Process finished
        _extraction_jobs.pop(stem)

        if rc != 0:
            log_file = output_dir / "_mineru.log"
            log_content = ""
            if log_file.exists():
                try:
                    log_content = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
                except OSError:
                    pass
            return (
                f"MineRU extraction **failed** (exit code {rc}).\n\n"
                f"**Log**:\n```\n{log_content or '(no log)'}\n```"
            )

        # Success — find the markdown
        if md_path.exists():
            return _build_extraction_result(stem, source_label, output_dir, auto_dir, md_path)

        md_files = list(output_dir.rglob("*.md"))
        if md_files:
            md_path = md_files[0]
            auto_dir = md_path.parent
            return _build_extraction_result(stem, source_label, output_dir, auto_dir, md_path)

        return (
            f"MineRU completed but no markdown file found.\n"
            f"Output directory: `{output_dir}`"
        )

    # --- No cache, no running job: start a new extraction ---

    # Check that mineru CLI is on PATH (instant check, no subprocess)
    if shutil.which("mineru") is None:
        return (
            "MineRU is not installed (not found on PATH). To install it:\n\n"
            "```bash\n"
            "uv pip install 'mineru[all]'\n"
            "```\n\n"
            "Or add it to the qgc_server environment:\n"
            "```bash\n"
            "cd qgc_server && uv sync\n"
            "```\n\n"
            "Note: MineRU includes PyTorch and ML model dependencies (~2GB download)."
        )

    # Create output directory and start background extraction
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = _start_mineru_background(pdf_path, output_dir)

    _extraction_jobs[stem] = {
        "process": proc,
        "pdf": pdf_path,
        "source": source_label,
    }

    return (
        f"MineRU extraction **started** for `{stem}`.\n\n"
        f"- PDF: `{pdf_path}`\n"
        f"- Output dir: `{output_dir}`\n"
        f"- PID: {proc.pid}\n\n"
        f"MineRU loads PyTorch + ML models, so first extraction may take 2-5 minutes.\n"
        f"**Call `extract_paper` again with the same arguments to check progress and get results.**"
    )


@mcp.tool()
def analyze_paper_for_gadgets(paper_text: str = "", paper_path: str = "") -> str:
    """Analyze extracted paper content and match against QGC gadget catalog.

    Scans the paper for quantum computing concepts — including text, tables,
    and figure captions — and identifies which could be implemented using
    existing QGC primitives. Also identifies concepts NOT yet in the catalog
    (gaps).

    Provide either paper_text directly or paper_path pointing to:
    - A MineRU-extracted .md file
    - A MineRU auto/ output directory (will read .md + _content_list.json)
    - Just the paper stem name (will look in /tmp/qgc_papers/<stem>/)

    Args:
        paper_text: The extracted paper content (markdown text).
        paper_path: Path to extracted .md file, auto/ directory, or paper stem name.
    """
    table_html_texts: list[str] = []
    table_images: list[dict] = []
    content_list_path: Optional[Path] = None

    # Resolve paper text and structured content
    if not paper_text and paper_path:
        p = Path(paper_path)

        # If it's a directory (auto/ dir), find the .md and content_list inside
        if p.is_dir():
            md_files = list(p.glob("*.md"))
            if md_files:
                paper_text = md_files[0].read_text(encoding="utf-8")
                stem = md_files[0].stem
                content_list_path = p / f"{stem}_content_list.json"
            else:
                return f"No .md file found in directory: {paper_path}"
        elif p.exists() and p.suffix == ".md":
            paper_text = p.read_text(encoding="utf-8")
            # Look for sibling content_list.json
            stem = p.stem
            content_list_path = p.parent / f"{stem}_content_list.json"
        else:
            # Try as a stem name under the cache dir
            stem = p.stem
            auto_dir = _PAPER_OUTPUT_BASE / stem / stem / "auto"
            cached_md = auto_dir / f"{stem}.md"
            if cached_md.exists():
                paper_text = cached_md.read_text(encoding="utf-8")
                content_list_path = auto_dir / f"{stem}_content_list.json"
            else:
                return f"Paper not found: {paper_path}"

    if not paper_text:
        return (
            "No paper content provided. Either pass paper_text directly, "
            "or provide paper_path pointing to a MineRU-extracted .md file "
            "or auto/ output directory."
        )

    # Parse structured content (tables, equations) if content_list available
    if content_list_path and content_list_path.exists():
        structured = _parse_content_list(content_list_path)
        for tbl in structured["tables"]:
            if tbl.get("html"):
                table_html_texts.append(tbl["html"])
            table_images.append({
                "page": tbl["page"],
                "caption": tbl.get("caption", ""),
                "img_path": tbl.get("img_path", ""),
            })

    # Fetch all gadgets from registry
    registry = get_registry()
    all_gadgets = registry.search(None)

    if not all_gadgets:
        return "No gadgets found in the QGC catalog. Ingest some gadgets first."

    # Build keyword map and scan — include table HTML in the searchable text
    combined_text = paper_text
    if table_html_texts:
        # Strip HTML tags to get plain text for keyword matching
        table_plain = "\n\n".join(
            re.sub(r"<[^>]+>", " ", html) for html in table_html_texts
        )
        combined_text = paper_text + "\n\n" + table_plain

    keyword_map = _build_gadget_keyword_map(all_gadgets)
    matches = _scan_paper_for_matches(combined_text, keyword_map)
    matched_names = {gid.split("@")[0] for gid in matches}
    gaps = _detect_gaps(combined_text, matched_names)

    # Build report
    lines = ["# Paper <> QGC Gadget Analysis\n"]

    # Summary
    lines.append(f"**Catalog size**: {len(all_gadgets)} gadget(s)")
    lines.append(f"**Matches found**: {len(matches)} gadget(s)")
    lines.append(f"**Concept gaps**: {len(gaps)}")
    if table_html_texts:
        lines.append(f"**Tables scanned**: {len(table_html_texts)}")
    lines.append("")

    # Matched gadgets
    if matches:
        lines.append("## Matching Gadgets\n")
        for gadget_id, excerpts in sorted(matches.items()):
            lines.append(f"### {gadget_id}")
            lines.append(f"*{len(excerpts)} relevant section(s) found*\n")
            for i, excerpt in enumerate(excerpts, 1):
                # Clean up excerpt for display
                clean = excerpt.replace("\n", " ").strip()
                if len(clean) > 150:
                    clean = clean[:150] + "..."
                lines.append(f"  {i}. \"{clean}\"\n")
    else:
        lines.append("## No Matching Gadgets\n")
        lines.append("No direct matches found between paper content and catalog gadgets.\n")

    # Unmatched gadgets (in catalog but not referenced in paper)
    all_gadget_ids = {f"{g.name}@{g.version}" for g in all_gadgets}
    unmatched = sorted(all_gadget_ids - set(matches.keys()))
    if unmatched:
        lines.append("## Unused Catalog Gadgets\n")
        lines.append("These gadgets are in the catalog but not referenced in the paper:\n")
        for gid in unmatched:
            lines.append(f"  - {gid}")
        lines.append("")

    # Tables with relevant images
    if table_images:
        lines.append("## Paper Tables\n")
        lines.append("Tables extracted from the paper (may contain circuit parameters, gate counts, etc.):\n")
        for i, tbl in enumerate(table_images, 1):
            caption = tbl["caption"] or "(no caption)"
            lines.append(f"  {i}. Page {tbl['page'] + 1}: {caption}")
            if tbl.get("img_path"):
                lines.append(f"     Image: `{tbl['img_path']}`")
        lines.append("")

    # Gaps — concepts in paper not covered by catalog
    if gaps:
        lines.append("## Concept Gaps (not in catalog)\n")
        lines.append("These quantum concepts appear in the paper but have no matching gadget:\n")
        for gap in gaps:
            phrases = ", ".join(f'"{p}"' for p in gap["matched_phrases"])
            lines.append(f"  - **{gap['concept']}** — detected phrases: {phrases}")
        lines.append("")

    # Suggestions
    lines.append("## Suggested Next Steps\n")
    if matches:
        lines.append("1. Use `get_gadget` to retrieve QASM for matched gadgets")
        lines.append("2. Use `compile_circuit` to assemble matched primitives into a larger circuit")
    if gaps:
        lines.append("3. Consider implementing missing concepts as new gadgets via `ingest_gadget`")
    if table_images:
        lines.append("4. Review table images for circuit diagrams or gate parameters")
    if not matches and not gaps:
        lines.append("The paper may not describe implementable quantum circuits, "
                      "or the concepts use different terminology than the catalog.")

    return "\n".join(lines)


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
