from itertools import product
from typing import Iterable
from networkx import MultiDiGraph
import numpy as np
import scipy.sparse as sp
from pyformlang.finite_automaton import Symbol
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton = None) -> None:
        if automaton is None:
            self.start_nodes = set()
            self.final_nodes = set()
            self.nodes = dict()
            self.boolean_decomposition = dict()
            self.num_sts = 0
            return

        graph = automaton.to_networkx()
        self.nodes = {state: i for (i, state) in enumerate(graph.nodes)}
        self.start_nodes = {self.nodes[state] for state in automaton.start_states}
        self.final_nodes = {self.nodes[state] for state in automaton.final_states}
        self.num_sts = len(automaton.states)

        nodes_num = len(self.nodes)
        matrixes = {
            s: np.zeros((nodes_num, nodes_num), dtype=bool) for s in automaton.symbols
        }

        for u, v, label in graph.edges(data="label"):
            if label:
                s = Symbol(label)
                matrixes[s][self.nodes[u], self.nodes[v]] = True

        self.boolean_decomposition = {s: sp.csr_array(m) for (s, m) in matrixes.items()}

    def accepts(self, word: Iterable[Symbol]) -> bool:
        curr_states = self.start_nodes.copy()

        for sym in word:
            if sym not in self.boolean_decomposition.keys():
                return False
            curr_states = {
                next_state
                for (curr_state, next_state) in product(
                    curr_states, self.nodes.values()
                )
                if self.boolean_decomposition[sym][curr_state, next_state]
            }

        if any(state in self.final_nodes for state in curr_states):
            return True

        return False

    def transitive_closure(self):
        if not self.boolean_decomposition:
            return np.eye(len(self.nodes), dtype=np.bool_)
        s = sum(self.boolean_decomposition.values())
        s.setdiag(True)
        return np.linalg.matrix_power(s.toarray(), len(self.nodes))

    def is_empty(self) -> bool:
        t = self.transitive_closure()
        return not any(t[s, f] for s in self.start_nodes for f in self.final_nodes)


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    intersection = AdjacencyMatrixFA()
    intersection.nodes = {
        (n1, n2): automaton1.nodes[n1] * len(automaton2.nodes) + automaton2.nodes[n2]
        for (n1, n2) in product(automaton1.nodes.keys(), automaton2.nodes.keys())
    }
    intersection.start_nodes = {
        idx
        for (n, idx) in intersection.nodes.items()
        if automaton1.nodes[n[0]] in automaton1.start_nodes
        and automaton2.nodes[n[1]] in automaton2.start_nodes
    }
    intersection.final_nodes = {
        idx
        for (n, idx) in intersection.nodes.items()
        if automaton1.nodes[n[0]] in automaton1.final_nodes
        and automaton2.nodes[n[1]] in automaton2.final_nodes
    }
    intersection.boolean_decomposition = {
        label: sp.kron(
            automaton1.boolean_decomposition[label],
            automaton2.boolean_decomposition[label],
            format="csr",
        )
        for label in automaton1.boolean_decomposition.keys()
        if label in automaton2.boolean_decomposition.keys()
    }

    return intersection


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    regex_mfa = AdjacencyMatrixFA(regex_dfa)
    graph_nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    intersect_fa = intersect_automata(graph_nfa, regex_mfa)
    tc = intersect_fa.transitive_closure()
    pairs = set()
    for start_n in start_nodes:
        for final_n in final_nodes:
            for start_st in regex_dfa.start_states:
                for final_st in regex_dfa.final_states:
                    if tc[
                        intersect_fa.nodes[(start_n, start_st)],
                        intersect_fa.nodes[(final_n, final_st)],
                    ]:
                        pairs.add((start_n, final_n))
    return pairs
