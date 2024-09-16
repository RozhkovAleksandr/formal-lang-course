from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.regular_expression import Regex
from networkx import MultiDiGraph
from typing import Set


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    dfa = Regex(regex).to_epsilon_nfa().to_deterministic()
    return dfa.minimize()

def graph_to_nfa(
  graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    pass