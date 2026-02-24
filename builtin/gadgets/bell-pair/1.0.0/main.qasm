// Bell pair gadget
// Creates entangled state |00> + |11>
OPENQASM 3.0;
include "stdgates.inc";

gate bell_pair q0, q1 {
    h q0;
    cx q0, q1;
}
