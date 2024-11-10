from pyformlang.cfg import CFG, Terminal, Variable, Epsilon
from pyformlang.rsa import RecursiveAutomaton
import networkx as nx
from typing import Iterable
from scipy.sparse import csc_matrix
from pyformlang.finite_automaton import Symbol, State, NondeterministicFiniteAutomaton

from project.task3 import AdjacencyMatrixFA, intersect_automata
from project.task6 import cfg_to_weak_normal_form
from project.task2 import graph_to_nfa


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def rsm_to_nfa(rsm: RecursiveAutomaton):
    nfa = NondeterministicFiniteAutomaton()

    for nterm, box in rsm.boxes.items():
        dfa = box.dfa

        for st in dfa.start_states:
            nfa.add_start_state(State((nterm, st.value)))
        for st in dfa.final_states:
            nfa.add_final_state(State((nterm, st.value)))

        for from_st in dfa.states:
            transitions: dict[Symbol, State | set[State]] = dfa.to_dict().get(from_st)
            if transitions is None:
                continue
            for symbol in transitions.keys():
                for to_st in (
                    transitions[symbol]
                    if isinstance(transitions[symbol], Iterable)
                    else {transitions[symbol]}
                ):
                    nfa.add_transition(
                        State((nterm, from_st)), symbol, State((nterm, to_st))
                    )
    return nfa


def classify_productions(productions):
    eps_pr: set[Variable] = set()
    term_pr: dict[Terminal, set[Variable]] = {}
    nterm_pr: dict[(Variable, Variable), set[Variable]] = {}

    for production in productions:
        head, body = production.head, production.body

        if len(body) == 0 or isinstance(body[0], Epsilon):
            eps_pr.add(head)
        elif len(body) == 1 and isinstance(body[0], Terminal):
            term_pr.setdefault(body[0].value, set()).add(head)
        else:
            nterm_pr.setdefault((body[0], body[1]), set()).add(head)

    return eps_pr, term_pr, nterm_pr


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
):
    cfg = cfg_to_weak_normal_form(cfg)

    eps_pr, term_pr, nterm_pr = classify_productions(cfg.productions)
    adj_m = {(n, n, eps) for n in graph.nodes for eps in eps_pr} | {
        (start, final, term)
        for start, final, lbl in graph.edges.data("label")
        if lbl in term_pr
        for term in term_pr[lbl]
    }
    queue = adj_m.copy()

    while queue:
        new_nonterm = set()
        start1, final1, nonterm1 = queue.pop()

        for start2, final2, nonterm2 in adj_m:
            if final1 == start2 and (nonterm1, nonterm2) in nterm_pr:
                for nterm in nterm_pr[(nonterm1, nonterm2)]:
                    if (start1, final2, nterm) not in adj_m:
                        new_nonterm.add((start1, final2, nterm))
            if final2 == start1 and (nonterm2, nonterm1) in nterm_pr:
                for nterm in nterm_pr[(nonterm2, nonterm1)]:
                    if (start2, final1, nterm) not in adj_m:
                        new_nonterm.add((start2, final1, nterm))

        queue |= new_nonterm
        adj_m |= new_nonterm

    res = {
        (start, final)
        for start, final, nterm in adj_m
        if (
            start in start_nodes
            and final in final_nodes
            and nterm.value == cfg.start_symbol.value
        )
    }

    return res


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
):
    cfg = cfg_to_weak_normal_form(cfg)

    num_nodes = graph.number_of_nodes()
    node_idx = {n: i for i, n in enumerate(graph.nodes)}
    eps_pr, term_pr, nterm_pr = classify_productions(cfg.productions)

    adj_m: dict[Variable, csc_matrix] = {
        nterm: csc_matrix((num_nodes, num_nodes), dtype=bool) for nterm in cfg.variables
    }

    for eps in eps_pr:
        for n in range(num_nodes):
            adj_m[eps][n, n] = True

    for s, f, lbl in graph.edges.data("label"):
        if lbl in term_pr:
            for term in term_pr[lbl]:
                adj_m[term][node_idx[s], node_idx[f]] = True

    queue = set(cfg.variables)
    while queue:
        updated_var = queue.pop()
        for B, C in nterm_pr:
            if updated_var != B and updated_var != C:
                continue

            matrix_change = adj_m[B] @ adj_m[C]
            for nterm in nterm_pr[(B, C)]:
                old_matrix = adj_m[nterm]
                adj_m[nterm] += matrix_change
                if (old_matrix != adj_m[nterm]).count_nonzero() != 0:
                    queue.add(nterm)

    idx_node = {i: n for n, i in node_idx.items()}
    valid_pairs: set[tuple[int, int]] = set()
    for nterm in adj_m:
        if nterm.value == cfg.start_symbol.value:
            for r, c in zip(*adj_m[nterm].nonzero()):
                if idx_node[r] in start_nodes and idx_node[c] in final_nodes:
                    valid_pairs.add((idx_node[r], idx_node[c]))

    return valid_pairs


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
):
    rsm_m = AdjacencyMatrixFA(rsm_to_nfa(rsm))
    graph_m = AdjacencyMatrixFA(
        graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes)
    )

    def delta(tc: csc_matrix):
        res: dict[Symbol, csc_matrix] = {}
        for i, j in zip(*tc.nonzero()):
            rsm_i, rsm_j = i % rsm_m.num_sts, j % rsm_m.num_sts
            st1, st2 = rsm_m.idx_to_st[rsm_i], rsm_m.idx_to_st[rsm_j]
            if st1 in rsm_m.start_sts and st2 in rsm_m.final_sts:
                assert st1.value[0] == st2.value[0]
                nterm = st1.value[0]

                graph_i, graph_j = i // rsm_m.num_sts, j // rsm_m.num_sts
                if (
                    nterm in graph_m.adjacency_matrices
                    and graph_m.adjacency_matrices[nterm][graph_i, graph_j]
                ):
                    continue

                if nterm not in res:
                    res[nterm] = csc_matrix(
                        (graph_m.num_sts, graph_m.num_sts), dtype=bool
                    )
                res[nterm][graph_i, graph_j] = True
        return res

    while True:
        transitive_closure = intersect_automata(graph_m, rsm_m).transitive_closure()
        m_delta = delta(transitive_closure)
        if not m_delta:
            break
        for symbol in m_delta.keys():
            if symbol not in graph_m.adjacency_matrices:
                graph_m.adjacency_matrices[symbol] = m_delta[symbol]
            else:
                graph_m.adjacency_matrices[symbol] += m_delta[symbol]

    valid_pairs: set[tuple[int, int]] = set()
    start_m = graph_m.adjacency_matrices.get(rsm.initial_label)
    if start_m is None:
        return valid_pairs

    for start in start_nodes:
        for final in final_nodes:
            if start_m[
                graph_m.st_to_idx[State(start)], graph_m.st_to_idx[State(final)]
            ]:
                valid_pairs.add((start, final))
    return valid_pairs
