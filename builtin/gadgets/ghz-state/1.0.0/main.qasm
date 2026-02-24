// GHZ state gadget
// Creates |000> + |111> across 3 qubits
OPENQASM 3.0;
include "stdgates.inc";

gate ghz_state q0, q1, q2 {
    h q0;
    cx q0, q1;
    cx q1, q2;
}
