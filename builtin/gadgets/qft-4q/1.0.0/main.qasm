// Quantum Fourier Transform - 4 qubit
// From FTCircuitBench, converted to OpenQASM 3.0
OPENQASM 3.0;
include "stdgates.inc";

// QFT gate definition for 4 qubits
gate qft_4q q0, q1, q2, q3 {
    h q0;
    cp(pi/2) q1, q0;
    h q1;
    cp(pi/4) q2, q0;
    cp(pi/2) q2, q1;
    h q2;
    cp(pi/8) q3, q0;
    cp(pi/4) q3, q1;
    cp(pi/2) q3, q2;
    h q3;
}
