from networkx import MultiDiGraph
from functools import reduce
from itertools import product
from scipy.sparse import csr_matrix, vstack
from project.task3 import AdjacencyMatrixFA
from project.task2 import graph_to_nfa, regex_to_dfa


def _initial_front(
    adj_matrix_dfa: AdjacencyMatrixFA, adj_matrix_nfa: AdjacencyMatrixFA
):
    dfa_start_state = list(
        product(adj_matrix_dfa.start_nodes, adj_matrix_nfa.start_nodes)
    )
    matices = []
    for dfa_idx, nfa_idx in dfa_start_state:
        matrix = csr_matrix(
            (len(adj_matrix_dfa.num_sts.keys()), len(adj_matrix_nfa.num_sts.keys())),
            dtype=bool,
        )
        matrix[dfa_idx, nfa_idx] = True
        matices.append(matrix)
    return vstack(matices, "csr", dtype=bool)


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    grapth_nfa = graph_to_nfa(graph, start_nodes, final_nodes)

    adj_matrix_dfa = AdjacencyMatrixFA(regex_dfa)
    adj_matrix_nfa = AdjacencyMatrixFA(grapth_nfa)

    index_to_state_nfa = {
        index: state for state, index in adj_matrix_nfa.num_sts.items()
    }
    symbols = (
        adj_matrix_dfa.boolean_decomposition.keys()
        & adj_matrix_nfa.boolean_decomposition.keys()
    )

    permutation_m = {
        sym: adj_matrix_dfa.boolean_decomposition[sym].transpose() for sym in symbols
    }

    front = _initial_front(adj_matrix_dfa, adj_matrix_nfa)
    visited = front

    m = len(adj_matrix_dfa.num_sts.keys())
    dfa_start_state = list(
        product(adj_matrix_dfa.start_nodes, adj_matrix_nfa.start_nodes)
    )
    while front.count_nonzero() > 0:
        new_front = []
        for sym in symbols:
            sym_front = front @ adj_matrix_nfa.boolean_decomposition[sym]
            new_front.append(
                vstack(
                    [
                        permutation_m[sym] @ sym_front[m * i : m * (i + 1)]
                        for i in range(len(dfa_start_state))
                    ]
                )
            )

        front = reduce(lambda x, y: x + y, new_front, front) > visited
        visited += front

    answer = set()
    for final_dfa in adj_matrix_dfa.final_nodes:
        for i, start in enumerate(adj_matrix_nfa.start_nodes):
            fix_start = visited[m * i : m * (i + 1)]
            for reached in fix_start.getrow(final_dfa).indices:
                if reached in adj_matrix_nfa.final_nodes:
                    answer.add((index_to_state_nfa[start], index_to_state_nfa[reached]))
    return answer
