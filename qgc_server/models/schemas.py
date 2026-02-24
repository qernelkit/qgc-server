from datetime import datetime
from enum import Enum
from typing import Annotated, List, Optional, Union

from pydantic import BaseModel, Field


class GadgetInterface(BaseModel):
    """Gadget interface specification."""
    input_qubits: int
    output_qubits: int
    classical_bits: int = 0


class GadgetDependency(BaseModel):
    """Gadget dependency specification."""
    gadget: str
    version_req: str


class MetricsHints(BaseModel):
    """Hints for circuit metrics."""
    t_count: Optional[int] = None
    cnot_count: Optional[int] = None
    depth: Optional[int] = None


class Hashes(BaseModel):
    """Hash values for artifact verification."""
    sha256: str


class Manifest(BaseModel):
    """Full gadget manifest."""
    name: str
    version: str
    ir: str = "openqasm3"
    interface: GadgetInterface
    depends_on: List[GadgetDependency] = Field(default_factory=list)
    metrics_hints: MetricsHints = Field(default_factory=MetricsHints)
    hashes: Hashes
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class GadgetSummary(BaseModel):
    """Summary of a gadget for search results."""
    name: str
    version: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


# Compile models

class GadgetRef(BaseModel):
    """Reference to a gadget (by ID)."""
    name: str
    version: str


class InlineQasm(BaseModel):
    """Inline OpenQASM replacement."""
    qasm: str


# Replacement is an untagged union - Pydantic will try GadgetRef first, then InlineQasm
Replacement = Annotated[Union[GadgetRef, InlineQasm], Field(discriminator=None)]


class GadgetOverride(BaseModel):
    """A single gadget override in a compile request."""
    target: str
    replacement: Replacement


class CompileMode(str, Enum):
    """Compilation mode."""
    baseline = "baseline"
    optimize = "optimize"


class CompileRequest(BaseModel):
    """Compilation request."""
    initial_qasm: str
    gadget_overrides: List[GadgetOverride] = Field(default_factory=list)
    mode: CompileMode = CompileMode.baseline
    metrics: List[str] = Field(default_factory=lambda: ["t_count", "cnot_count", "depth"])


class CompileMetrics(BaseModel):
    """Metrics from compilation."""
    t_count: Optional[int] = None
    cnot_count: Optional[int] = None
    depth: Optional[int] = None


class CompileReport(BaseModel):
    """Compilation report."""
    metrics: CompileMetrics
    warnings: List[str] = Field(default_factory=list)
    gadgets_used: List[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """Provenance information for compiled output."""
    registry_url: str
    compiled_at: datetime
    compiler_version: str


class CompileResponse(BaseModel):
    """Compilation response."""
    compiled_qasm: str
    report: CompileReport
    provenance: Provenance


# Resolve models

class ResolveRequest(BaseModel):
    """Resolve request - for dependency resolution."""
    gadgets: List[GadgetRef]


class ResolvedGadget(BaseModel):
    """Resolved gadget with full details."""
    name: str
    version: str
    sha256: str


class ResolveResponse(BaseModel):
    """Resolve response."""
    resolved: List[ResolvedGadget]


# ChangeBucket models

class ChangeSource(str, Enum):
    """Source of a gadget change."""
    manual = "manual"
    ai = "ai"
    optimizer = "optimizer"


class Change(BaseModel):
    """A single change/modification to a gadget."""
    id: str  # UUID
    qasm: str  # Full QASM snapshot
    metadata: dict = Field(default_factory=dict)  # Associated metadata
    metrics: Optional[MetricsHints] = None  # Computed metrics if available
    diff_from_base: Optional[str] = None  # Unified diff from original
    source: ChangeSource = ChangeSource.manual
    notes: Optional[str] = None  # User notes
    created_at: datetime


class ChangeBucket(BaseModel):
    """Collection of changes for a gadget."""
    gadget_name: str
    gadget_version: str
    base_qasm: str  # Original QASM (immutable reference)
    base_metadata: dict = Field(default_factory=dict)  # Original metadata
    changes: List[Change] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class AddChangeRequest(BaseModel):
    """Request to add a new change to the bucket."""
    qasm: str  # New QASM version
    metadata: dict = Field(default_factory=dict)
    notes: Optional[str] = None
    source: ChangeSource = ChangeSource.manual


class AddChangeResponse(BaseModel):
    """Response after adding a change."""
    change_id: str
    diff_from_base: str
    metrics: Optional[MetricsHints] = None


class PromoteChangeRequest(BaseModel):
    """Request to promote a change to a new gadget version."""
    new_version: str  # e.g., "1.1.0"
    description: Optional[str] = None


class PromoteChangeResponse(BaseModel):
    """Response after promoting a change."""
    success: bool
    new_gadget_id: str  # "name@new_version"
    storage_path: str
