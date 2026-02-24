"""Gadget ingestion endpoint with AI-powered schema extraction.

POST /ingest

Accepts multiple input formats (auto-detected):

1. QASM only:
   { "qasm": "OPENQASM 3.0; ..." }

2. QASM + metadata:
   { "qasm": "OPENQASM 3.0; ...", "metadata": { ... } }

3. Single JSON blob with embedded QASM:
   { "blob": { "experiment": "...", "circuit": { "qasm": "OPENQASM 3.0; ..." }, ... } }
   Optionally specify: "qasm_path": "circuit.qasm" (dot-notation)

Uses Ollama Cloud to extract structured manifest from freeform metadata.
"""

import hashlib
import json
import os
import httpx
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..models import Manifest, GadgetInterface, MetricsHints, Hashes
from ..services.registry import get_registry

router = APIRouter()

# Ollama Cloud config
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "https://ollama.com/api/chat")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# The target schema for gadget.json v1
GADGET_SCHEMA = """
{
  "name": "string (lowercase, hyphenated, e.g., 'qft-4q')",
  "version": "string (semver, e.g., '1.0.0')",
  "description": "string (human-readable description)",
  "tags": ["string"] (list of search tags),
  "input_qubits": "int (number of input qubits)",
  "output_qubits": "int (number of output qubits)",
  "classical_bits": "int (number of classical bits)",
  "t_count": "int (number of T gates)",
  "cnot_count": "int (number of CNOT/CX gates)",
  "depth": "int (circuit depth estimate)"
}
"""

# Common field names where QASM might be found
QASM_FIELD_CANDIDATES = [
    "qasm", "openqasm", "circuit", "code", "qasm_code", "circuit_qasm",
    "qasm_str", "qasm_string", "openqasm3", "qasm3"
]


class IngestRequest(BaseModel):
    """Flexible request - provide qasm directly, or blob with embedded qasm."""
    # Option 1: Direct QASM
    qasm: Optional[str] = Field(None, description="OpenQASM circuit code (direct)")

    # Option 2: Metadata to accompany direct QASM
    metadata: Optional[dict[str, Any]] = Field(None, description="Freeform metadata")

    # Option 3: Single blob with QASM embedded somewhere
    blob: Optional[dict[str, Any]] = Field(None, description="Full JSON blob with embedded QASM")

    # Optional: specify where QASM is in the blob (dot notation: "circuit.qasm")
    qasm_path: Optional[str] = Field(None, description="Dot-notation path to QASM field in blob")


class IngestResponse(BaseModel):
    """Response from gadget ingestion."""
    success: bool
    gadget_id: str
    manifest: Manifest
    storage_path: str
    ai_extraction: dict
    input_mode: str  # "qasm_only", "qasm_with_metadata", "blob"


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_nested_value(obj: dict, path: str) -> Any:
    """Get value from nested dict using dot notation: 'a.b.c' -> obj['a']['b']['c']"""
    keys = path.split('.')
    value = obj
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def find_qasm_in_blob(blob: dict, path_hint: Optional[str] = None) -> tuple[Optional[str], dict]:
    """
    Find QASM string in a blob and return (qasm, remaining_metadata).

    If path_hint is provided, use it directly.
    Otherwise, search common field names recursively.
    """
    if path_hint:
        qasm = get_nested_value(blob, path_hint)
        if qasm and isinstance(qasm, str) and _looks_like_qasm(qasm):
            # Remove the qasm from metadata (shallow copy)
            metadata = _remove_nested_key(blob, path_hint)
            return qasm, metadata

    # Auto-detect: search for QASM in common locations
    def search_recursive(obj: dict, current_path: str = "") -> Optional[tuple[str, str]]:
        for key, value in obj.items():
            full_path = f"{current_path}.{key}" if current_path else key

            # Check if this key is a QASM candidate
            if key.lower() in QASM_FIELD_CANDIDATES:
                if isinstance(value, str) and _looks_like_qasm(value):
                    return value, full_path
                elif isinstance(value, dict):
                    # Check nested (e.g., circuit.qasm)
                    for subkey, subval in value.items():
                        if subkey.lower() in QASM_FIELD_CANDIDATES:
                            if isinstance(subval, str) and _looks_like_qasm(subval):
                                return subval, f"{full_path}.{subkey}"

            # Recurse into dicts
            if isinstance(value, dict):
                result = search_recursive(value, full_path)
                if result:
                    return result

        return None

    result = search_recursive(blob)
    if result:
        qasm, found_path = result
        metadata = _remove_nested_key(blob, found_path)
        return qasm, metadata

    return None, blob


def _looks_like_qasm(s: str) -> bool:
    """Check if a string looks like QASM code."""
    s_lower = s.lower().strip()
    return (
        s_lower.startswith('openqasm') or
        'qreg' in s_lower or
        'qubit[' in s_lower or
        ('include' in s_lower and 'stdgates' in s_lower)
    )


def _remove_nested_key(obj: dict, path: str) -> dict:
    """Return a copy of obj with the nested key at path removed."""
    import copy
    result = copy.deepcopy(obj)

    keys = path.split('.')
    if len(keys) == 1:
        result.pop(keys[0], None)
    else:
        # Navigate to parent and remove the final key
        parent = result
        for key in keys[:-1]:
            if key in parent and isinstance(parent[key], dict):
                parent = parent[key]
            else:
                return result  # Path doesn't exist, return unchanged
        parent.pop(keys[-1], None)

    return result


async def extract_with_ollama(qasm: str, metadata: dict) -> dict:
    """Use Ollama Cloud to extract structured manifest from QASM + metadata."""
    from fastapi import HTTPException

    if not OLLAMA_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OLLAMA_API_KEY not configured. Set the environment variable to enable AI extraction."
        )

    prompt = f"""You are a quantum computing expert. Extract structured metadata from the following quantum circuit and freeform metadata.

## Target Schema
{GADGET_SCHEMA}

## Input QASM Circuit
```qasm
{qasm[:2000]}
```

## Input Metadata (freeform JSON)
```json
{json.dumps(metadata, indent=2, default=str)[:3000]}
```

## Instructions
1. Analyze the QASM to count qubits, gates (especially T and CNOT), estimate depth
2. Extract name, description, tags from the metadata (or infer from QASM if not provided)
3. Return ONLY valid JSON matching the target schema - no markdown, no explanation

## Output (JSON only)
"""

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                OLLAMA_API_URL,
                headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
            )
            response.raise_for_status()
            result = response.json()

            content = result.get("message", {}).get("content", "")

            # Parse JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama extraction failed: {e}"
            )


@router.post("", response_model=IngestResponse)
async def ingest_gadget(request: IngestRequest):
    """
    Ingest a gadget - auto-detects input format.

    POST /ingest

    **Format 1: Direct QASM**
    ```json
    { "qasm": "OPENQASM 3.0; ..." }
    ```

    **Format 2: QASM + metadata**
    ```json
    {
        "qasm": "OPENQASM 3.0; ...",
        "metadata": { "name": "my-gadget", "tags": ["test"] }
    }
    ```

    **Format 3: Single blob with embedded QASM**
    ```json
    {
        "blob": {
            "experiment": "bell-test",
            "circuit": { "qasm": "OPENQASM 3.0; ..." },
            "results": { "fidelity": 0.98 }
        }
    }
    ```
    Optionally add `"qasm_path": "circuit.qasm"` to specify location.
    """
    qasm: str
    metadata: dict
    input_mode: str

    # Determine input mode
    if request.qasm:
        # Mode 1 or 2: Direct QASM provided
        qasm = request.qasm
        metadata = request.metadata or {}
        input_mode = "qasm_with_metadata" if request.metadata else "qasm_only"

    elif request.blob:
        # Mode 3: Blob with embedded QASM
        found_qasm, remaining_metadata = find_qasm_in_blob(request.blob, request.qasm_path)

        if not found_qasm:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail="Could not find QASM in blob. Provide 'qasm_path' or ensure QASM is in a standard field."
            )

        qasm = found_qasm
        metadata = remaining_metadata
        input_mode = "blob"

    else:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Provide either 'qasm' (with optional 'metadata') or 'blob' containing embedded QASM."
        )

    # Extract with AI (or fallback)
    extracted = await extract_with_ollama(qasm, metadata)

    # Compute hash
    sha256 = compute_sha256(qasm.encode('utf-8'))

    # Build manifest
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

    # Store original input for provenance
    original_input = {
        "input_mode": input_mode,
        "metadata": metadata,
    }
    if request.blob:
        original_input["original_blob"] = request.blob

    # Insert into database via registry
    registry = get_registry()
    registry.insert_gadget(manifest, qasm, original_input)

    return IngestResponse(
        success=True,
        gadget_id=f"{manifest.name}@{manifest.version}",
        manifest=manifest,
        storage_path=f"database:gadgets/{manifest.name}/{manifest.version}",
        ai_extraction=extracted,
        input_mode=input_mode,
    )
