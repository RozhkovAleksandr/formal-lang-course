import numpy as np

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy.sparse import csr_matrix, kron
from typing import Iterable
from itertools import product


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton = None):
        if automaton is None:
            self.number_of_states = 0
            self.states = dict()
            self.start_states = set()
            self.final_states = set()
            self.decomposition = dict()
            return

        graph = automaton.to_networkx()
        self.number_of_states = graph.number_of_nodes()
        self.states = dict(zip(graph.nodes, range(self.number_of_states)))
        self.start_states = set(self.states.keys()).intersection(automaton.start_states)
        self.final_states = set(self.states.keys()).intersection(automaton.final_states)

        transitions = {}
        for symbol in automaton.symbols:
            transitions[symbol] = np.zeros(
                (self.number_of_states, self.number_of_states), dtype=bool
            )

        for s1, s2, label in graph.edges(data="label"):
            if label:
                state1 = self.states[s1]
                state2 = self.states[s2]
                transitions[label][state1, state2] = 1

        self.decomposition = dict(
            zip(
                transitions.keys(),
                [csr_matrix(matrix) for matrix in transitions.values()],
            )
        )

    def accepts(self, word: Iterable[Symbol]) -> bool:
        states = set(self.start_states)

        for letter in word:
            if self.decomposition.get(letter) is None:
                return False

            for s1, s2 in product(states, self.states.keys()):
                if self.decomposition[letter][self.states[s1], self.states[s2]]:
                    states.add(s2)

        if states.intersection(self.final_states):
            return True

        return False

    def transitive_closure(self) -> np.ndarray:
        A = np.eye(self.number_of_states, dtype=bool)

        for dec in self.decomposition.values():
            A |= dec.toarray()

        transitive_closure = np.linalg.matrix_power(A, self.number_of_states).astype(
            bool
        )
        return transitive_closure

    def is_empty(self) -> bool:
        transitive_closure = self.transitive_closure()
        for start_state in self.start_states:
            for final_state in self.final_states:
                if transitive_closure[
                    self.states[start_state], self.states[final_state]
                ]:
                    return False

        return True
