# -*- coding: utf-8 -*-

from examples.qrbs import *
from examples.inferential_circuit import *

import unittest


class QRBSTestSuite(unittest.TestCase):
    """Tests of QRBS example"""

    def test_qrbs(self):
        inferential_circuit = AndOperator(
            'I', 0.90,
            OrOperator(
                'E', 0.67,
                AndOperator(
                    'C', 0.75,
                    Fact('A', 1.00),
                    Fact('B', 0.40)
                ),
                Fact('D', 0.42)
            ),
            OrOperator(
                'H', 0.76,
                Fact('F', 0.33),
                Fact('G', 0.85)
            )
        )

        platforms = [QiskitPlatform, CirqPlatform]

        results = list(map(lambda platform: platform.execute(inferential_circuit.accept(platform)), platforms))

        [print(result.values) for result in results]
