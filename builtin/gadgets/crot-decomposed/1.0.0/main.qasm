// Controlled rotation decomposition gadget
// Decomposes controlled rotation into CNOT and single-qubit gates
OPENQASM 3.0;
include "stdgates.inc";

// Controlled-Rz decomposition
// CRz(theta) = CNOT . Rz(-theta/2) . CNOT . Rz(theta/2)
gate crot_z control, target {
    // Rz(theta/2) on target
    rz(pi/4) target;

    // CNOT
    cx control, target;

    // Rz(-theta/2) on target
    rz(-pi/4) target;

    // CNOT
    cx control, target;
}

// Controlled-Ry decomposition
gate crot_y control, target {
    // Ry(theta/2) on target
    ry(pi/4) target;

    // CNOT
    cx control, target;

    // Ry(-theta/2) on target
    ry(-pi/4) target;

    // CNOT
    cx control, target;
}
