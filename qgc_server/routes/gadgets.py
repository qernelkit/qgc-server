"""Gadget registry endpoints."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Response

from ..models import GadgetSummary, Manifest
from ..services.registry import get_registry

router = APIRouter()


@router.get("", response_model=List[GadgetSummary])
async def search_gadgets(q: Optional[str] = Query(None, description="Search query")):
    """Search for gadgets.

    GET /gadgets - List all gadgets
    GET /gadgets?q=query - Search with query
    """
    registry = get_registry()
    return registry.search(q)


@router.get("/{name}", response_model=Manifest)
async def get_gadget_latest(name: str):
    """Get the latest manifest for a gadget.

    GET /gadgets/{name} - Get latest version
    """
    registry = get_registry()
    manifest = registry.get_manifest(name)

    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Gadget not found: {name}")

    return manifest


@router.get("/{name}/{version}", response_model=Manifest)
async def get_gadget_version(name: str, version: str):
    """Get manifest for a specific gadget version.

    GET /gadgets/{name}/{version} - Get specific version
    """
    registry = get_registry()
    manifest = registry.get_manifest(name, version)

    if manifest is None:
        raise HTTPException(
            status_code=404, detail=f"Gadget not found: {name}@{version}"
        )

    return manifest


@router.get("/{name}/{version}/artifact")
async def download_artifact(name: str, version: str):
    """Download the OpenQASM 3 artifact for a gadget.

    GET /gadgets/{name}/{version}/artifact - Download QASM
    """
    registry = get_registry()
    artifact = registry.get_artifact(name, version)

    if artifact is None:
        raise HTTPException(
            status_code=404, detail=f"Artifact not found: {name}@{version}"
        )

    return Response(
        content=artifact,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{name}-{version}.qasm"'},
    )
