"""Compiler service for circuit compilation with gadget replacement."""

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ..models import (
    CompileReport,
    CompileRequest,
    CompileResponse,
    GadgetOverride,
    GadgetRef,
    InlineQasm,
    Provenance,
)
from ..config import settings
from .registry import get_registry
from .metrics import MetricsService


class CompilerService:
    """Service for compiling quantum circuits with gadget replacement."""

    # Pattern to match gadget markers: // @gadget <name> ... // @end_gadget
    GADGET_START_PATTERN = re.compile(r"//\s*@gadget\s+(\S+)")
    GADGET_END_PATTERN = re.compile(r"//\s*@end_gadget")

    def compile(self, request: CompileRequest) -> CompileResponse:
        """Compile a circuit with gadget overrides."""
        warnings: List[str] = []
        gadgets_used: List[str] = []

        # Build override map: target -> replacement
        override_map: Dict[str, GadgetOverride] = {}
        for override in request.gadget_overrides:
            override_map[override.target] = override

        # Process the QASM code
        compiled_qasm, used_gadgets, compile_warnings = self._process_qasm(
            request.initial_qasm, override_map
        )

        gadgets_used.extend(used_gadgets)
        warnings.extend(compile_warnings)

        # Compute metrics
        metrics = MetricsService.compute_metrics(compiled_qasm, request.metrics)

        # Build response
        report = CompileReport(
            metrics=metrics,
            warnings=warnings,
            gadgets_used=gadgets_used,
        )

        provenance = Provenance(
            registry_url=settings.registry_url,
            compiled_at=datetime.now(timezone.utc),
            compiler_version=settings.compiler_version,
        )

        return CompileResponse(
            compiled_qasm=compiled_qasm,
            report=report,
            provenance=provenance,
        )

    def _process_qasm(
        self,
        qasm: str,
        override_map: Dict[str, GadgetOverride],
    ) -> Tuple[str, List[str], List[str]]:
        """Process QASM code, replacing gadget markers with gadget code.

        Returns (processed_qasm, gadgets_used, warnings).
        """
        gadgets_used: List[str] = []
        warnings: List[str] = []

        lines = qasm.split("\n")
        result_lines: List[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for gadget start marker
            start_match = self.GADGET_START_PATTERN.search(line)
            if start_match:
                gadget_target = start_match.group(1)

                # Find the end marker
                end_index = None
                for j in range(i + 1, len(lines)):
                    if self.GADGET_END_PATTERN.search(lines[j]):
                        end_index = j
                        break

                if end_index is None:
                    warnings.append(f"Missing @end_gadget for @gadget {gadget_target}")
                    result_lines.append(line)
                    i += 1
                    continue

                # Get replacement code
                replacement_code = self._get_replacement(
                    gadget_target, override_map, warnings
                )

                if replacement_code is not None:
                    gadgets_used.append(gadget_target)
                    # Add comment showing what was replaced
                    result_lines.append(f"// Replaced gadget: {gadget_target}")
                    result_lines.append(replacement_code.rstrip())
                    result_lines.append(f"// End replaced gadget: {gadget_target}")
                else:
                    # Keep original code if no replacement found
                    for k in range(i, end_index + 1):
                        result_lines.append(lines[k])

                i = end_index + 1
            else:
                result_lines.append(line)
                i += 1

        return "\n".join(result_lines), gadgets_used, warnings

    def _get_replacement(
        self,
        target: str,
        override_map: Dict[str, GadgetOverride],
        warnings: List[str],
    ) -> Optional[str]:
        """Get replacement code for a gadget target."""
        if target not in override_map:
            warnings.append(f"No override specified for gadget target: {target}")
            return None

        override = override_map[target]
        replacement = override.replacement

        # Handle GadgetRef - fetch from registry
        if isinstance(replacement, GadgetRef):
            registry = get_registry()
            artifact = registry.get_artifact(replacement.name, replacement.version)
            if artifact is None:
                warnings.append(
                    f"Gadget not found: {replacement.name}@{replacement.version}"
                )
                return None
            return artifact.decode("utf-8")

        # Handle InlineQasm
        if isinstance(replacement, InlineQasm):
            return replacement.qasm

        # Should not reach here, but handle gracefully
        warnings.append(f"Unknown replacement type for target: {target}")
        return None


# Global singleton
_compiler: Optional[CompilerService] = None


def get_compiler() -> CompilerService:
    """Get the global compiler service instance."""
    global _compiler
    if _compiler is None:
        _compiler = CompilerService()
    return _compiler
