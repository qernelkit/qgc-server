"""Compile endpoint."""

from fastapi import APIRouter

from ..models import CompileRequest, CompileResponse
from ..services.compiler import get_compiler

router = APIRouter()


@router.post("/compile", response_model=CompileResponse)
async def compile_circuit(request: CompileRequest):
    """Compile a circuit with gadget overrides.

    POST /compile - Compile circuit (NOT /gadgets/compile)
    """
    compiler = get_compiler()
    return compiler.compile(request)
