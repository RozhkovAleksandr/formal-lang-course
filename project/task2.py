from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.regular_expression import Regex
from networkx import MultiDiGraph
from typing import Set

import networkx as nx


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    dfa = Regex(regex).to_epsilon_nfa().to_deterministic()
    return dfa.minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton.from_networkx(graph)

    start_states = start_states if start_states else set(graph)

    final_states = final_states if final_states else set(graph)

    for state in start_states:
        nfa.add_start_state(state)

    for state in final_states:
        nfa.add_final_state(state)

    return nfa.remove_epsilon_transitions()


graph = nx.MultiDiGraph()
graph.add_edges_from(
    [
        (1, 2, {"label": "a"}),
        (1, 3, {"label": "b"}),
        (2, 4, {"label": "a"}),
        (3, 4, {"label": "b"}),
    ]
)


nfa = graph_to_nfa(graph, start_states={1}, final_states={4})

print(nfa._states)
