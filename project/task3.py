import numpy as np
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy.sparse import csr_matrix, kron
from typing import Iterable
from itertools import product
import networkx as nx


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


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    intersection_matrix = AdjacencyMatrixFA()

    intersection_matrix.number_of_states = (
        automaton1.number_of_states * automaton2.number_of_states
    )

    intersection_matrix.states = {}
    for s1 in automaton1.states.keys():
        for s2 in automaton2.states.keys():
            intersection_matrix.states[(s1, s2)] = (
                automaton1.states[s1] * automaton2.number_of_states
                + automaton2.states[s2]
            )

    intersection_matrix.start_states = set(
        intersection_matrix.states.keys()
    ).intersection(product(automaton1.start_states, automaton2.start_states))

    intersection_matrix.final_states = set(
        intersection_matrix.states.keys()
    ).intersection(product(automaton1.final_states, automaton2.final_states))

    intersection_matrix.decomposition = {
        key: kron(
            automaton1.decomposition[key],
            automaton2.decomposition[key],
            format="csr",
        )
        for key in automaton1.decomposition.keys()
        if key in automaton2.decomposition
    }

    return intersection_matrix


def tensor_based_rpq(
    regex: str,
    graph: nx.MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    regex_to_matrix = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_to_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    intersection = intersect_automata(regex_to_matrix, graph_to_matrix)
    closure = intersection.transitive_сlosure()

    return {
        (graph_start, graph_final)
        for graph_start in graph_to_matrix.start_states
        for graph_final in graph_to_matrix.final_states
        if any(
            closure[
                intersection.states[(regex_start, graph_start)],
                intersection.states[(regex_final, graph_final)],
            ]
            for regex_start in regex_to_matrix.start_states
            for regex_final in regex_to_matrix.final_states
        )
    }
