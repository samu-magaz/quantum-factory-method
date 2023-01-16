# -*- coding: utf-8 -*-

from examples.openqasm import QiskitPlatform, CirqPlatform

import unittest


class OpenQASMTestSuite(unittest.TestCase):
    """Tests of OpenQASM example"""

    def test_openqasm(self):
        bell_state = """
            OPENQASM 2.0;
            include "qelib1.inc";

            qreg q[2];	// 2 qubits
            creg c[2]; 	// 2 bits
            h q[0];  	    // Hadamard gate on qubit
            cx q[0],q[1]; // CNOT gate on qubits
            
            measure q[0] -> c[0];
            measure q[1] -> c[1];
        """

        platforms = [QiskitPlatform, CirqPlatform]

        results = list(map(lambda platform: platform.execute(platform.build(bell_state)), platforms))

        [print(result.values) for result in results]
