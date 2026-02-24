"""Resolve endpoint."""

from typing import List

from fastapi import APIRouter, HTTPException

from ..models import ResolveRequest, ResolveResponse, ResolvedGadget
from ..services.registry import get_registry

router = APIRouter()


@router.post("/resolve", response_model=ResolveResponse)
async def resolve_dependencies(request: ResolveRequest):
    """Resolve gadget dependencies.

    POST /resolve - Resolve dependencies (NOT /gadgets/resolve)
    """
    registry = get_registry()
    resolved: List[ResolvedGadget] = []

    for gadget_ref in request.gadgets:
        manifest = registry.get_manifest(gadget_ref.name, gadget_ref.version)

        if manifest is None:
            raise HTTPException(
                status_code=404,
                detail=f"Gadget not found: {gadget_ref.name}@{gadget_ref.version}",
            )

        resolved.append(
            ResolvedGadget(
                name=manifest.name,
                version=manifest.version,
                sha256=manifest.hashes.sha256,
            )
        )

        # Also resolve transitive dependencies
        for dep in manifest.depends_on:
            # For now, use exact version matching
            dep_manifest = registry.get_manifest(dep.gadget)
            if dep_manifest is not None:
                # Check if already in resolved list
                if not any(
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

    return ResolveResponse(resolved=resolved)
