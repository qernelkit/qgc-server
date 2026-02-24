"""ChangeBucket endpoints for tracking gadget modifications.

Provides a way to store and cycle through modifications to a gadget's QASM,
with full snapshots and diffs, and the ability to promote changes to new versions.
"""

import difflib
import hashlib

from fastapi import APIRouter, HTTPException

from ..models import (
    Change,
    ChangeBucket,
    AddChangeRequest,
    AddChangeResponse,
    PromoteChangeRequest,
    PromoteChangeResponse,
    MetricsHints,
    Manifest,
    Hashes,
)
from ..services.registry import get_registry
from ..services.database import get_database
from ..services.metrics import compute_metrics

router = APIRouter()


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


@router.get("/{name}/{version}/changes", response_model=ChangeBucket)
async def get_changes(name: str, version: str):
    """
    Get all changes for a gadget.

    Returns the full ChangeBucket including base QASM and all modifications.
    """
    db = get_database()
    bucket = db.get_change_bucket(name, version)

    if bucket is None:
        raise HTTPException(status_code=404, detail=f"Gadget {name}@{version} not found")

    return bucket


@router.post("/{name}/{version}/changes", response_model=AddChangeResponse)
async def add_change(name: str, version: str, request: AddChangeRequest):
    """
    Add a new change to the gadget's change bucket.

    Stores full QASM snapshot and computes diff from base.
    """
    db = get_database()
    registry = get_registry()

    gadget = registry.get_gadget(name, version)
    if not gadget:
        raise HTTPException(status_code=404, detail=f"Gadget {name}@{version} not found")

    # Compute diff from base
    diff = _compute_diff(gadget.qasm, request.qasm)

    # Compute metrics for the new QASM
    try:
        computed = compute_metrics(request.qasm, ["t_count", "cnot_count", "depth"])
        metrics = MetricsHints(
            t_count=computed.t_count,
            cnot_count=computed.cnot_count,
            depth=computed.depth,
        )
    except Exception:
        metrics = None

    # Add to database
    change_id = db.add_change(
        gadget_name=name,
        gadget_version=version,
        qasm=request.qasm,
        metadata=request.metadata,
        metrics=metrics,
        diff_from_base=diff,
        source=request.source,
        notes=request.notes,
    )

    return AddChangeResponse(
        change_id=change_id,
        diff_from_base=diff,
        metrics=metrics,
    )


@router.get("/{name}/{version}/changes/{change_id}", response_model=Change)
async def get_change(name: str, version: str, change_id: str):
    """
    Get a specific change by ID.
    """
    db = get_database()
    change = db.get_change(change_id)

    if change is None:
        raise HTTPException(status_code=404, detail=f"Change {change_id} not found")

    return change


@router.delete("/{name}/{version}/changes/{change_id}")
async def delete_change(name: str, version: str, change_id: str):
    """
    Delete a specific change from the bucket.
    """
    db = get_database()
    deleted = db.delete_change(change_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Change {change_id} not found")

    return {"success": True, "deleted": change_id}


@router.post("/{name}/{version}/changes/{change_id}/promote", response_model=PromoteChangeResponse)
async def promote_change(name: str, version: str, change_id: str, request: PromoteChangeRequest):
    """
    Promote a change to become a new version of the gadget.

    Creates a new gadget version with the change's QASM.
    """
    db = get_database()
    registry = get_registry()

    # Get the change
    change = db.get_change(change_id)
    if change is None:
        raise HTTPException(status_code=404, detail=f"Change {change_id} not found")

    # Check if new version already exists
    if registry.gadget_exists(name, request.new_version):
        raise HTTPException(
            status_code=409,
            detail=f"Gadget {name}@{request.new_version} already exists"
        )

    # Get original gadget as template
    original = registry.get_gadget(name, version)
    if not original:
        raise HTTPException(status_code=404, detail=f"Original gadget {name}@{version} not found")

    # Compute hash of new QASM
    sha256 = hashlib.sha256(change.qasm.encode("utf-8")).hexdigest()

    # Build new manifest
    new_manifest = Manifest(
        name=name,
        version=request.new_version,
        ir=original.manifest.ir,
        interface=original.manifest.interface,
        depends_on=original.manifest.depends_on,
        metrics_hints=change.metrics or original.manifest.metrics_hints,
        hashes=Hashes(sha256=sha256),
        description=request.description or original.manifest.description,
        tags=original.manifest.tags,
    )

    # Store provenance info
    provenance = {
        "promoted_from": f"{name}@{version}",
        "change_id": change_id,
        "original_metadata": change.metadata,
        "notes": change.notes,
    }

    # Insert new gadget
    registry.insert_gadget(new_manifest, change.qasm, provenance)

    return PromoteChangeResponse(
        success=True,
        new_gadget_id=f"{name}@{request.new_version}",
        storage_path=f"database:gadgets/{name}/{request.new_version}",
    )


@router.get("/{name}/{version}/changes/compare/{change_id_a}/{change_id_b}")
async def compare_changes(name: str, version: str, change_id_a: str, change_id_b: str):
    """
    Compare two changes, showing the diff between them.
    """
    db = get_database()

    change_a = db.get_change(change_id_a)
    change_b = db.get_change(change_id_b)

    if change_a is None:
        raise HTTPException(status_code=404, detail=f"Change {change_id_a} not found")
    if change_b is None:
        raise HTTPException(status_code=404, detail=f"Change {change_id_b} not found")

    diff = _compute_diff(change_a.qasm, change_b.qasm)

    return {
        "change_a": change_id_a,
        "change_b": change_id_b,
        "diff": diff,
        "metrics_a": change_a.metrics,
        "metrics_b": change_b.metrics,
    }
