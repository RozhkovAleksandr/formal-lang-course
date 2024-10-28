import networkx as nx
from pyformlang.cfg import CFG, Terminal, Variable
from scipy.sparse import csr_matrix
from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    n: int = graph.number_of_nodes()
    node_to_index = {node: idx for idx, node in enumerate(graph.nodes())}

    matricies = {}
    for u, v, data in graph.edges(data=True):
        if data.get("label") is not None:
            for prod in cfg_to_weak_normal_form(cfg).productions:
                if len(prod.body) == 1 and isinstance(prod.body[0], Terminal):
                    terminal = prod.body[0].value
                    if terminal == data.get("label"):
                        head = prod.head
                        if head not in matricies:
                            matricies[head] = csr_matrix((n, n), dtype=bool)
                        matricies[head][node_to_index[u], node_to_index[v]] = True

    for node in graph.nodes:
        for var in cfg_to_weak_normal_form(cfg).get_nullable_symbols():
            var = Variable(var.value)
            if var not in matricies:
                matricies[var] = csr_matrix((n, n), dtype=bool)
            matricies[var][node_to_index[node], node_to_index[node]] = True

    check = True
    while check:
        check = False
        for prod in cfg_to_weak_normal_form(cfg).productions:
            if len(prod.body) == 2:
                h = prod.head
                if (
                    Variable(prod.body[0].value) in matricies
                    and Variable(prod.body[1].value) in matricies
                ):
                    if h not in matricies:
                        matricies[h] = csr_matrix((n, n), dtype=bool)

                    new_mat = (
                        matricies[Variable(prod.body[0].value)]
                        @ matricies[Variable(prod.body[1].value)]
                    )
                    new_mat_coo = new_mat.tocoo()

                    for u, v, value in zip(
                        new_mat_coo.row, new_mat_coo.col, new_mat_coo.data
                    ):
                        if value and not matricies[h][u, v]:
                            matricies[h][u, v] = True
                            check = True

    s = cfg_to_weak_normal_form(cfg).start_symbol
    idx = {idx: node for node, idx in node_to_index.items()}

    res = set()
    if s in matricies:
        final_matrix = matricies[s].tocoo()
        for u_idx, v_idx in zip(final_matrix.row, final_matrix.col):
            u = idx[u_idx]
            v = idx[v_idx]
            if (not start_nodes or u in start_nodes) and (
                not final_nodes or v in final_nodes
            ):
                res.add((u, v))

    return res
