from pyformlang.cfg import CFG, Variable, Production, Epsilon
import networkx as nx


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    productions = set(cfg.to_normal_form().productions)
    for var in cfg.get_nullable_symbols():
        productions.add(Production(Variable(var.value), [Epsilon()]))

    wcnf_cfg = CFG(
        start_symbol=cfg.start_symbol, productions=productions
    ).remove_useless_symbols()

    return wcnf_cfg


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    weak_cnf_cfg = cfg_to_weak_normal_form(cfg)

    old = []

    for v1, v2, symbol in graph.edges(data="label"):
        for prod in weak_cnf_cfg.productions:
            if len(prod.body) == 1 and prod.body[0].value == symbol:
                old.append((prod.head, v1, v2))
    for var in weak_cnf_cfg.variables:
        if Production(var, []) in weak_cnf_cfg.productions:
            for vertex in graph.nodes:
                old.append((var, vertex, vertex))
    new = old.copy()

    while new:
        (N, n, m) = new.pop()

        for M, n_p, m_p in old:
            if m_p == n:
                for prod in weak_cnf_cfg.productions:
                    if len(prod.body) == 2 and prod.body[0] == M and prod.body[1] == N:
                        N_prime = prod.head
                        new_relation = (N_prime, n_p, m)
                        if new_relation not in old:
                            old.append(new_relation)
                            new.append(new_relation)
        for M, n_p, m_p in old:
            if m == n_p:
                for prod in weak_cnf_cfg.productions:
                    if len(prod.body) == 2 and prod.body[0] == N and prod.body[1] == M:
                        N_prime = prod.head
                        new_relation = (N_prime, n, m_p)
                        if new_relation not in old:
                            old.append(new_relation)
                            new.append(new_relation)

    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    result = {
        (start, final)
        for var, start, final in old
        if start in start_nodes and final in final_nodes and var == cfg.start_symbol
    }

    return result
