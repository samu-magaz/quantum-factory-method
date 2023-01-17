'''QRBS example module
'''

from __future__ import annotations
import abc
import re
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from inferential_circuit import Fact, NotOperator, AndOperator, OrOperator
import numpy as np

import qiskit
from qiskit.quantum_info.operators import Operator
from qiskit_aer import Aer

import cirq
from cirq import LineQubit, CircuitOperation
from cirq import X
from cirq import CNOT
from cirq import CCNOT


class Circuit(abc.ABC):
    '''Base class for Circuit classes
    '''
    def __init__(self, tags: dict[str,int]) -> None:
        self.tags = tags


class QiskitCircuit(Circuit):
    '''Circuit class for Qiskit
    '''
    def __init__(self, tags: dict[str,int], built_circuit: qiskit.QuantumCircuit) -> None:
        super().__init__(tags)
        self.built_circuit = built_circuit


class CirqCircuit(Circuit):
    '''Circuit class for Cirq
    '''
    def __init__(self, tags: dict[str,int], built_circuit: cirq.Circuit) -> None:
        super().__init__(tags)
        self.built_circuit = built_circuit


class Result:
    '''Class to store the values after executing a circuit
    '''
    def __init__(self, values: dict[str,float]) -> None:
        self.values = values


class Platform(abc.ABC):
    '''Base class for Platform classes
    '''
    @staticmethod
    @abc.abstractmethod
    def build_fact(fact: Fact) -> Circuit:
        '''Abstract method for building the corresponding quantum circuit of a fact; part of Visitor design pattern

        :param fact: The Fact object whose quantum circuit will be built
        :type fact: Fact
        :return: The Circuit object with the quantum circuit
        :rtype: Circuit
        '''
        pass
    
    @staticmethod
    @abc.abstractmethod
    def build_not(notOperator: NotOperator) -> Circuit:
        '''Abstract method for building the corresponding quantum circuit of a not operator; part of Visitor design pattern

        :param notOperator: The NotOperator whose quantum circuit will be built
        :type notOperator: NotOperator
        :return: The Circuit object with the quantum circuit
        :rtype: Circuit
        '''
        pass
    
    @staticmethod
    @abc.abstractmethod
    def build_and(andOperator: AndOperator) -> Circuit:
        '''Abstract method for building the corresponding quantum circuit of an and operator; part of Visitor design pattern

        :param andOperator: The AndOperator whose quantum circuit will be built
        :type andOperator: AndOperator
        :return: The Circuit object with the quantum circuit
        :rtype: Circuit
        '''
        pass
    
    @staticmethod
    @abc.abstractmethod
    def build_or(orOperator: OrOperator) -> Circuit:
        '''Abstract method for building the corresponding quantum circuit of an or operator; part of Visitor design pattern

        :param orOperator: The OrOperator whose quantum circuit will be built
        :type orOperator: OrOperator
        :return: The Circuit object with the quantum circuit
        :rtype: Circuit
        '''
        pass
    
    @staticmethod
    @abc.abstractmethod
    def execute(circuit: Circuit) -> Result:
        '''Abstract method for executing a previously built quantum circuit

        :param Circuit: The Circuit object containing the quantum circuit that will be executed
        :type Circuit: Circuit
        :return: The Result object with the values of the execution
        :rtype: Result
        '''
        pass
    
    
class QiskitPlatform(Platform):
    '''Platform class for Qiskit
    '''
    def _build_m_operator(certainty: float) -> Operator:
        '''Private method to build the corresponding quantum operator of the M gate

        :param certainty: The certainty associated with the element
        :type certainty: float
        :return: The Operator that will be applied to the quantum circuit
        :rtype: Operator
        '''
        alpha = certainty * np.pi/2
        return Operator([
            [np.cos(alpha), np.sin(alpha)],
            [np.sin(alpha), -np.cos(alpha)]
        ])

    @staticmethod
    @abc.abstractmethod
    def build_fact(fact: Fact) -> QiskitCircuit:
        '''Builds the corresponding quantum circuit of a fact; part of Visitor design pattern

        :param fact: The Fact object whose quantum circuit will be built
        :type fact: Fact
        :return: The QiskitCircuit object with the quantum circuit
        :rtype: QiskitCircuit
        '''
        tags = {fact.tag: 0}

        qreg = qiskit.QuantumRegister(1)
        circ = qiskit.QuantumCircuit(qreg)
        M = QiskitPlatform._build_m_operator(fact.certainty)

        circ.append(M, [qreg[0]])

        return QiskitCircuit(tags, circ)
    
    @staticmethod
    @abc.abstractmethod
    def build_not(notOperator: NotOperator) -> QiskitCircuit:
        '''Builds the corresponding quantum circuit of a not operator; part of Visitor design pattern

        :param notOperator: The NotOperator object whose quantum circuit will be built
        :type notOperator: NotOperator
        :return: The QiskitCircuit object with the quantum circuit
        :rtype: QiskitCircuit
        '''
        child_circ = notOperator.child.accept(QiskitPlatform)

        height = child_circ.built_circuit.num_qubits + 2

        tags = {**child_circ.tags}
        del tags[notOperator.child.tag]
        tags[notOperator.tag] = height - 1

        qreg = qiskit.QuantumRegister(height)
        circ = qiskit.QuantumCircuit(qreg)
        M = QiskitPlatform._build_m_operator(notOperator.certainty)

        circ.compose(child_circ.built_circuit, qreg[0 : child_circ.built_circuit.num_qubits], inplace=True)
        circ.x(qreg[-3])
        circ.append(M, [qreg[-2]])
        circ.ccx(qreg[-3], qreg[-2], qreg[-1])

        return QiskitCircuit(tags, circ)
    
    @staticmethod
    @abc.abstractmethod
    def build_and(andOperator: AndOperator) -> QiskitCircuit:
        '''Builds the corresponding quantum circuit of an and operator; part of Visitor design pattern

        :param andOperator: The AndOperator object whose quantum circuit will be built
        :type andOperator: AndOperator
        :return: The QiskitCircuit object with the quantum circuit
        :rtype: QiskitCircuit
        '''
        left_child_circ = andOperator.left_child.accept(QiskitPlatform)
        right_child_circ = andOperator.right_child.accept(QiskitPlatform)

        height = left_child_circ.built_circuit.num_qubits + right_child_circ.built_circuit.num_qubits + 3

        tags = {**left_child_circ.tags, **{k: (v + left_child_circ.built_circuit.num_qubits) for k,v in right_child_circ.tags.items()}}
        tags[andOperator.left_child.tag] = left_child_circ.built_circuit.num_qubits - 1
        tags[andOperator.right_child.tag] = left_child_circ.built_circuit.num_qubits + right_child_circ.built_circuit.num_qubits - 1
        tags[andOperator.tag] = height - 1

        qreg = qiskit.QuantumRegister(height)
        circ = qiskit.QuantumCircuit(qreg)
        M = QiskitPlatform._build_m_operator(andOperator.certainty)

        circ.compose(left_child_circ.built_circuit, qreg[0 : left_child_circ.built_circuit.num_qubits], inplace=True)
        circ.compose(right_child_circ.built_circuit, qreg[left_child_circ.built_circuit.num_qubits : left_child_circ.built_circuit.num_qubits + right_child_circ.built_circuit.num_qubits], inplace=True)
        circ.ccx(qreg[left_child_circ.built_circuit.num_qubits - 1], qreg[left_child_circ.built_circuit.num_qubits + right_child_circ.built_circuit.num_qubits - 1], qreg[-3])
        circ.append(M, [qreg[-2]])
        circ.ccx(qreg[-3], qreg[-2], qreg[-1])

        return QiskitCircuit(tags, circ)
    
    @staticmethod
    @abc.abstractmethod
    def build_or(orOperator: OrOperator) -> QiskitCircuit:
        '''Builds the corresponding quantum circuit of an or operator; part of Visitor design pattern

        :param orOperator: The OrOperator object whose quantum circuit will be built
        :type orOperator: OrOperator
        :return: The QiskitCircuit object with the quantum circuit
        :rtype: QiskitCircuit
        '''
        left_child_circ = orOperator.left_child.accept(QiskitPlatform)
        right_child_circ = orOperator.right_child.accept(QiskitPlatform)

        height = left_child_circ.built_circuit.num_qubits + right_child_circ.built_circuit.num_qubits + 3

        tags = {**left_child_circ.tags, **{k: (v + left_child_circ.built_circuit.num_qubits) for k,v in right_child_circ.tags.items()}}
        tags[orOperator.left_child.tag] = left_child_circ.built_circuit.num_qubits - 1
        tags[orOperator.right_child.tag] = left_child_circ.built_circuit.num_qubits + right_child_circ.built_circuit.num_qubits - 1
        tags[orOperator.tag] = height - 1

        qreg = qiskit.QuantumRegister(height)
        circ = qiskit.QuantumCircuit(qreg)
        M = QiskitPlatform._build_m_operator(orOperator.certainty)

        circ.compose(left_child_circ.built_circuit, qreg[0 : left_child_circ.built_circuit.num_qubits], inplace=True)
        circ.compose(right_child_circ.built_circuit, qreg[left_child_circ.built_circuit.num_qubits : left_child_circ.built_circuit.num_qubits + right_child_circ.built_circuit.num_qubits], inplace=True)
        circ.ccx(qreg[left_child_circ.built_circuit.num_qubits - 1], qreg[left_child_circ.built_circuit.num_qubits + right_child_circ.built_circuit.num_qubits - 1], qreg[-3])
        circ.cx(qreg[left_child_circ.built_circuit.num_qubits - 1], qreg[-3])
        circ.cx(qreg[left_child_circ.built_circuit.num_qubits + right_child_circ.built_circuit.num_qubits - 1], qreg[-3])
        circ.append(M, [qreg[-2]])
        circ.ccx(qreg[-3], qreg[-2], qreg[-1])

        return QiskitCircuit(tags, circ)
    
    @staticmethod
    @abc.abstractmethod
    def execute(quantum_circuit: QiskitCircuit) -> Result:
        '''Executes a previously built quantum circuit

        :param quantum_circuit: The QiskitCircuit object containing the quantum circuit that will be executed
        :type quantum_circuit: QiskitCircuit
        :return: The Result object with the values of the execution
        :rtype: Result
        '''
        circ = quantum_circuit.built_circuit
        circ.measure_all()

        simulator = Aer.get_backend('aer_simulator')
        circ = qiskit.transpile(circ, simulator)
        result = simulator.run(circ).result()

        counts = result.get_counts()
        counts = {k[::-1]: v for k,v in counts.items()}

        values = []
        for kq,vq in quantum_circuit.tags.items():
            temp = 0
            for kc,vc in counts.items():
                if kc[vq] == '1':
                    temp += vc
            values.append({'tag': kq, 'measure': temp/1024})
        return Result(values)
    
    
class CirqPlatform(Platform):
    '''Platform class for Cirq
    '''

    class MGate(cirq.Gate):
        '''Class for the M gate implementation in Cirq
        '''
        def __init__(self, certainty):
            super()
            self.alpha = certainty * np.pi/2

        def _num_qubits_(self):
            return 1

        def _unitary_(self):
            return np.array([
                [np.cos(self.alpha), np.sin(self.alpha)],
                [np.sin(self.alpha), -np.cos(self.alpha)]
            ]) / np.sqrt(2)

        def _circuit_diagram_info_(self, args):
            return f"M({self.alpha})"

    @staticmethod
    @abc.abstractmethod
    def build_fact(fact: Fact) -> CirqCircuit:
        '''Builds the corresponding quantum circuit of a fact; part of Visitor design pattern

        :param fact: The Fact object whose quantum circuit will be built
        :type fact: Fact
        :return: The CirqCircuit object with the quantum circuit
        :rtype: CirqCircuit
        '''
        tags = {fact.tag: 0}

        tags[fact.tag] = 0

        circ = cirq.Circuit(CirqPlatform.MGate(fact.certainty).on(LineQubit(0)))

        return CirqCircuit(tags, circ)
    
    @staticmethod
    @abc.abstractmethod
    def build_not(notOperator: NotOperator) -> CirqCircuit:
        '''Builds the corresponding quantum circuit of a not operator; part of Visitor design pattern

        :param notOperator: The NotOperator object whose quantum circuit will be built
        :type notOperator: NotOperator
        :return: The CirqCircuit object with the quantum circuit
        :rtype: CirqCircuit
        '''
        child_circ = notOperator.child.accept(CirqPlatform)

        height = len(child_circ.built_circuit.all_qubits()) + 2

        tags = {**child_circ.tags}
        del tags[notOperator.child.tag]
        tags[notOperator.tag] = height - 1

        qubits = LineQubit.range(height)
        child_op = CircuitOperation(child_circ.built_circuit.freeze())

        qubit_map = {}
        for i, child_qubit in enumerate(child_op.qubits):
            qubit_map[child_qubit] = qubits[i]

        circ = cirq.Circuit(
            child_op.with_qubit_mapping(qubit_map),
            X(qubits[-3]), 
            CirqPlatform.MGate(notOperator.certainty).on(qubits[-2]), 
            CCNOT(qubits[-3], qubits[-2], qubits[-1])
        )

        return CirqCircuit(tags, circ)
    
    @staticmethod
    @abc.abstractmethod
    def build_and(andOperator: AndOperator) -> CirqCircuit:
        '''Builds the corresponding quantum circuit of an and operator; part of Visitor design pattern

        :param andOperator: The AndOperator object whose quantum circuit will be built
        :type andOperator: AndOperator
        :return: The CirqCircuit object with the quantum circuit
        :rtype: CirqCircuit
        '''
        left_child_circ = andOperator.left_child.accept(CirqPlatform)
        right_child_circ = andOperator.right_child.accept(CirqPlatform)

        height = len(left_child_circ.built_circuit.all_qubits()) + len(right_child_circ.built_circuit.all_qubits()) + 3

        tags = {**left_child_circ.tags, **{k: (v + len(left_child_circ.built_circuit.all_qubits())) for k,v in right_child_circ.tags.items()}}
        tags[andOperator.left_child.tag] = len(left_child_circ.built_circuit.all_qubits()) - 1
        tags[andOperator.right_child.tag] = len(left_child_circ.built_circuit.all_qubits()) + len(right_child_circ.built_circuit.all_qubits()) - 1
        tags[andOperator.tag] = height - 1

        qubits = LineQubit.range(height)
        left_child_op = CircuitOperation(left_child_circ.built_circuit.freeze())
        right_child_op = CircuitOperation(right_child_circ.built_circuit.freeze())

        left_qubit_map = {}
        for i, child_qubit in enumerate(left_child_op.qubits):
            left_qubit_map[child_qubit] = qubits[i]

        right_qubit_map = {}
        for i, child_qubit in enumerate(right_child_op.qubits):
            right_qubit_map[child_qubit] = qubits[i + len(left_child_circ.built_circuit.all_qubits())]

        circ = cirq.Circuit(
            left_child_op.with_qubit_mapping(left_qubit_map),
            right_child_op.with_qubit_mapping(right_qubit_map),
            CCNOT(qubits[len(left_child_circ.built_circuit.all_qubits()) - 1], qubits[len(left_child_circ.built_circuit.all_qubits()) + len(right_child_circ.built_circuit.all_qubits()) - 1], qubits[-3]),
            CirqPlatform.MGate(andOperator.certainty).on(qubits[-2]), 
            CCNOT(qubits[-3], qubits[-2], qubits[-1])
        )

        return CirqCircuit(tags, circ)
    
    @staticmethod
    @abc.abstractmethod
    def build_or(orOperator: OrOperator) -> CirqCircuit:
        '''Builds the corresponding quantum circuit of an or operator; part of Visitor design pattern

        :param orOperator: The OrOperator object whose quantum circuit will be built
        :type orOperator: OrOperator
        :return: The CirqCircuit object with the quantum circuit
        :rtype: CirqCircuit
        '''
        left_child_circ = orOperator.left_child.accept(CirqPlatform)
        right_child_circ = orOperator.right_child.accept(CirqPlatform)

        height = len(left_child_circ.built_circuit.all_qubits()) + len(right_child_circ.built_circuit.all_qubits()) + 3

        tags = {**left_child_circ.tags, **{k: (v + len(left_child_circ.built_circuit.all_qubits())) for k,v in right_child_circ.tags.items()}}
        tags[orOperator.left_child.tag] = len(left_child_circ.built_circuit.all_qubits()) - 1
        tags[orOperator.right_child.tag] = len(left_child_circ.built_circuit.all_qubits()) + len(right_child_circ.built_circuit.all_qubits()) - 1
        tags[orOperator.tag] = height - 1

        qubits = LineQubit.range(height)
        left_child_op = CircuitOperation(left_child_circ.built_circuit.freeze())
        right_child_op = CircuitOperation(right_child_circ.built_circuit.freeze())

        left_qubit_map = {}
        for i, child_qubit in enumerate(left_child_op.qubits):
            left_qubit_map[child_qubit] = qubits[i]

        right_qubit_map = {}
        for i, child_qubit in enumerate(right_child_op.qubits):
            right_qubit_map[child_qubit] = qubits[i + len(left_child_circ.built_circuit.all_qubits())]

        circ = cirq.Circuit(
            left_child_op.with_qubit_mapping(left_qubit_map),
            right_child_op.with_qubit_mapping(right_qubit_map),
            CCNOT(qubits[len(left_child_circ.built_circuit.all_qubits()) - 1], qubits[len(left_child_circ.built_circuit.all_qubits()) + len(right_child_circ.built_circuit.all_qubits()) - 1], qubits[-3]),
            CNOT(qubits[len(left_child_circ.built_circuit.all_qubits()) - 1], qubits[-3]),
            CNOT(qubits[len(left_child_circ.built_circuit.all_qubits()) + len(right_child_circ.built_circuit.all_qubits()) - 1], qubits[-3]),
            CirqPlatform.MGate(orOperator.certainty).on(qubits[-2]), 
            CCNOT(qubits[-3], qubits[-2], qubits[-1])
        )

        return CirqCircuit(tags, circ)
    
    @staticmethod
    @abc.abstractmethod
    def execute(quantum_circuit: CirqCircuit) -> Result:
        '''Executes a previously built quantum circuit

        :param quantum_circuit: The CirqCircuit object containing the quantum circuit that will be executed
        :type quantum_circuit: CirqCircuit
        :return: The Result object with the values of the execution
        :rtype: Result
        '''
        circ = quantum_circuit.built_circuit
        circ.append(cirq.measure(*circ.all_qubits()))
        result = cirq.Simulator().run(circ, repetitions=1024)

        keys = [k for k,_ in result.measurements.items()]
        measures = [m for _,m in result.measurements.items()]
        [measures] = measures
        keys = list(map(lambda k: int(k), re.findall("[0-9]+", keys[0])))

        values = []
        for kq,vq in quantum_circuit.tags.items():
            temp = 0
            for value in measures:
                temp += value[keys.index(vq)]
            values.append({'tag': kq, 'measure': temp/1024})
        
        return Result(values)
