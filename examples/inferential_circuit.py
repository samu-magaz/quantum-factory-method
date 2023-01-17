'''Inferential circuit module
'''

from __future__ import annotations
import abc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from qrbs import Platform, Circuit


class InferentialCircuit(abc.ABC):
    '''Base class for InferentialCircuit classes; part of Composite design pattern 
    '''
    def __init__(self, tag: str, certainty: float) -> None:
        '''InferentialCircuit constructor

        :param tag: The tag that idefntifies the element
        :type tag: str
        :param certainty: The certainty associated with the element
        :type certainty: float
        '''
        self.tag = tag
        self.certainty = certainty

    @abc.abstractmethod
    def accept(platform: Platform) -> Circuit:
        '''Abstract method for accepting a Platform and calling the building method; part of Visitor design pattern 

        :param platform: The Platform that will build the quantum circuit
        :type platform: qrbs.Platform
        :return: The Circuit object with the quantum circuit
        :rtype: qrbs.Circuit
        '''
        pass


class Fact(InferentialCircuit):
    '''InferentialCircuit class for Fact; part of Composite design pattern
    '''
    def __init__(self, tag: str, certainty: float) -> None:
        '''Fact constructor

        :param tag: The tag that identifies the fact
        :type tag: str
        :param certainty: The certainty associated with the fact
        :type certainty: float
        '''
        super().__init__(tag, certainty)

    def accept(self, platform: Platform) -> Circuit:
        '''Accepts a Platform and calls the building method; part of Visitor design pattern 

        :param platform: The Platform that will build the quantum circuit
        :type platform: qrbs.Platform
        :return: The Circuit object with the quantum circuit
        :rtype: qrbs.Circuit
        '''
        return platform.build_fact(self)


class NotOperator(InferentialCircuit):
    '''InferentialCircuit class for NotOperator; part of Composite design pattern
    '''
    def __init__(self, tag: str, certainty: float, child: InferentialCircuit) -> None:
        '''NotOperator constructor

        :param tag: The tag that identifies the not operator
        :type tag: str
        :param certainty: The certainty associated with the not operator
        :type certainty: float
        :param child: The child of the not operator
        :type child: InferentialCircuit
        '''
        super().__init__(tag, certainty)
        self.child = child

    def accept(self, platform: Platform) -> Circuit:
        '''Accepts a Platform and calls the building method; part of Visitor design pattern 

        :param platform: The Platform that will build the quantum circuit
        :type platform: qrbs.Platform
        :return: The Circuit object with the quantum circuit
        :rtype: qrbs.Circuit
        '''
        return platform.build_not(self)


class AndOperator(InferentialCircuit):
    '''InferentialCircuit class for AndOperator; part of Composite design pattern
    '''
    def __init__(self, tag: str, certainty: float, left_child: InferentialCircuit, right_child: InferentialCircuit) -> None:
        '''AndOperator constructor

        :param tag: The tag that identifies the and operator
        :type tag: str
        :param certainty: The certainty associated with the and operator
        :type certainty: float
        :param left_child: The left child of the and operator
        :type left_child: InferentialCircuit
        :param right_child: The right child of the and operator
        :type right_child: InferentialCircuit
        '''
        super().__init__(tag, certainty)
        self.left_child = left_child
        self.right_child = right_child

    def accept(self, platform: Platform) -> Circuit:
        '''Accepts a Platform and calls the building method; part of Visitor design pattern 

        :param platform: The Platform that will build the quantum circuit
        :type platform: qrbs.Platform
        :return: The Circuit object with the quantum circuit
        :rtype: qrbs.Circuit
        '''
        return platform.build_and(self)


class OrOperator(InferentialCircuit):
    '''InferentialCircuit class for OrOperator; part of Composite design pattern
    '''
    def __init__(self, tag: str, certainty: float, left_child: InferentialCircuit, right_child: InferentialCircuit) -> None:
        '''OrOperator constructor

        :param tag: The tag that identifies the or operator
        :type tag: str
        :param certainty: The certainty associated with the or operator
        :type certainty: float
        :param left_child: The left child of the or operator
        :type left_child: InferentialCircuit
        :param right_child: The right child of the or operator
        :type right_child: InferentialCircuit
        '''
        super().__init__(tag, certainty)
        self.left_child = left_child
        self.right_child = right_child

    def accept(self, platform: Platform) -> Circuit:
        '''Accepts a Platform and calls the building method; part of Visitor design pattern 

        :param platform: The Platform that will build the quantum circuit
        :type platform: qrbs.Platform
        :return: The Circuit object with the quantum circuit
        :rtype: qrbs.Circuit
        '''
        return platform.build_or(self)