from __future__ import annotations
import abc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from qrbs import Platform, Circuit


class InferentialCircuit():
    def __init__(self, tag: str, certainty: float) -> None:
        self.tag = tag
        self.certainty = certainty

    @abc.abstractmethod
    def accept(platform: Platform) -> Circuit:
        pass


class Fact(InferentialCircuit):
    def accept(self, platform: Platform) -> Circuit:
        return platform.build_fact(self)


class NotOperator(InferentialCircuit):
    def __init__(self, tag: str, certainty: float, child: InferentialCircuit) -> None:
        super().__init__(tag, certainty)
        self.child = child

    def accept(self, platform: Platform) -> Circuit:
        return platform.build_not(self)


class AndOperator(InferentialCircuit):
    def __init__(self, tag: str, certainty: float, left_child: InferentialCircuit, right_child: InferentialCircuit) -> None:
        super().__init__(tag, certainty)
        self.left_child = left_child
        self.right_child = right_child

    def accept(self, platform: Platform) -> Circuit:
        return platform.build_and(self)


class OrOperator(InferentialCircuit):
    def __init__(self, tag: str, certainty: float, left_child: InferentialCircuit, right_child: InferentialCircuit) -> None:
        super().__init__(tag, certainty)
        self.left_child = left_child
        self.right_child = right_child

    def accept(self, platform: Platform) -> Circuit:
        return platform.build_or(self)