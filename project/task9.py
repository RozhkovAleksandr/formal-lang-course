from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Tuple
from pyformlang.finite_automaton import Symbol, DeterministicFiniteAutomaton
from pyformlang import rsa
import networkx as nx


def gll_based_cfpq(
    rsm: rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] | None = None,
    final_nodes: Set[int] | None = None,
) -> Set[Tuple[int, int]]:
    if (start_nodes is None) or (start_nodes == set()):
        start_nodes = set(graph.nodes())
    if (final_nodes is None) or (final_nodes == set()):
        final_nodes = set(graph.nodes())

    s = GllSolver(rsm, graph)
    return s.solve_reach(start_nodes, final_nodes)


class GSSNode:
    state: RsmState
    node: int
    edges: Dict[RsmState, Set[GSSNode]]
    pop_set: Set[int]

    def __init__(self, st: RsmState, nd: int):
        self.state = st
        self.node = nd
        self.edges = {}
        self.pop_set = set()

    def pop(self, cur_node: int) -> Set[SPPFNode]:
        res_set = set()

        if cur_node not in self.pop_set:
            for new_st in self.edges:
                gses = self.edges[new_st]
                for gs in gses:
                    res_set.add(SPPFNode(gs, new_st, cur_node))

            self.pop_set.add(cur_node)
        return res_set

    def add_edge(self, ret_st: RsmState, ptr: GSSNode) -> Set[SPPFNode]:
        res_set = set()

        st_edges = self.edges.get(ret_st, set())
        if ptr not in st_edges:
            st_edges.add(ptr)
            for cur_node in self.pop_set:
                res_set.add(SPPFNode(ptr, ret_st, cur_node))

        self.edges[ret_st] = st_edges

        return res_set


class GSStack:
    body: Dict[Tuple[RsmState, int], GSSNode]

    def __init__(self):
        self.body = {}

    def get_node(self, rsm_st: RsmState, node: int):
        res = self.body.get((rsm_st, node), None)
        if res is None:
            res = GSSNode(rsm_st, node)
            self.body[(rsm_st, node)] = res
        return res


@dataclass
class RsmStateData:
    term_edges: Dict[Symbol, RsmState]
    var_edges: Dict[Symbol, Tuple[RsmState, RsmState]]
    is_final: bool


@dataclass(frozen=True)
class RsmState:
    var: Symbol
    sub_state: str


@dataclass(frozen=True)
class SPPFNode:
    gssn: GSSNode
    state: RsmState
    node: int


class GllSolver:
    def is_term(self, s: str) -> bool:
        return Symbol(s) not in self.rsmstate2data

    def init_graph_data(self, graph: nx.DiGraph):
        edges = graph.edges(data="label")

        for n in graph.nodes():
            self.nodes2edges[n] = {}

        for from_n, to_n, symb in edges:
            if symb is not None:
                edges = self.nodes2edges[from_n]
                s: Set = edges.get(symb, set())
                s.add(to_n)
                edges[symb] = s

    def init_rsm_data(self, rsm: rsa.RecursiveAutomaton):
        for var in rsm.boxes:
            self.rsmstate2data[var] = {}

        for var in rsm.boxes:
            box = rsm.boxes[var]
            fa: DeterministicFiniteAutomaton = box.dfa
            gbox = fa.to_networkx()

            sub_dict = self.rsmstate2data[var]

            for sub_state in gbox.nodes:
                is_fin = sub_state in fa.final_states
                sub_dict[sub_state] = RsmStateData({}, {}, is_fin)

            edges = gbox.edges(data="label")
            for from_st, to_st, symb in edges:
                if symb is not None:
                    st_edges = sub_dict[from_st]
                    if self.is_term(symb):
                        st_edges.term_edges[symb] = RsmState(var, to_st)
                    else:
                        bfa: DeterministicFiniteAutomaton = rsm.boxes[Symbol(symb)].dfa
                        box_start = bfa.start_state.value
                        st_edges.var_edges[symb] = (
                            RsmState(Symbol(symb), box_start),
                            RsmState(var, to_st),
                        )

        start_symb = rsm.initial_label
        start_fa: DeterministicFiniteAutomaton = rsm.boxes[start_symb].dfa
        self.start_rstate = RsmState(start_symb, start_fa.start_state.value)

    def __init__(
        self,
        rsm: rsa.RecursiveAutomaton,
        graph: nx.DiGraph,
    ):
        self.nodes2edges: Dict[int, Dict[Symbol, Set[int]]] = {}
        self.rsmstate2data: Dict[Symbol, Dict[str, RsmStateData]] = {}
        self.start_rstate: RsmState

        self.rsm = rsm
        self.graph = graph

        self.init_graph_data(graph)
        self.init_rsm_data(rsm)

        self.gss = GSStack()
        self.accept_gssnode = self.gss.get_node(RsmState(Symbol("$"), "fin"), -1)

        self.unprocessed: Set[SPPFNode] = set()
        self.added: Set[SPPFNode] = set()

    def add_sppf_nodes(self, snodes: Set[SPPFNode]):
        snodes.difference_update(self.added)

        self.added.update(snodes)
        self.unprocessed.update(snodes)

    def filter_poped_nodes(
        self, snodes: Set[SPPFNode], prev_snode: SPPFNode
    ) -> Tuple[Set[SPPFNode], Set[Tuple[int, int]]]:
        node_res_set = set()
        start_fin_res_set = set()

        for sn in snodes:
            if sn.gssn == self.accept_gssnode:
                start_node = prev_snode.gssn.node
                fin_node = sn.node
                start_fin_res_set.add((start_node, fin_node))
            else:
                node_res_set.add(sn)

        return (node_res_set, start_fin_res_set)

    def step(self, sppfnode: SPPFNode) -> Set[Tuple[int, int]]:
        rsm_st = sppfnode.state
        rsm_dat = self.rsmstate2data[rsm_st.var][rsm_st.sub_state]

        def term_step():
            rsm_terms = rsm_dat.term_edges
            graph_terms = self.nodes2edges[sppfnode.node]
            for term in rsm_terms:
                if term in graph_terms:
                    new_sppf_nodes = set()
                    rsm_new_st = rsm_terms[term]
                    graph_new_nodes = graph_terms[term]
                    for gn in graph_new_nodes:
                        new_sppf_nodes.add(SPPFNode(sppfnode.gssn, rsm_new_st, gn))

                    self.add_sppf_nodes(new_sppf_nodes)

        def var_step() -> Set[Tuple[int, int]]:
            start_fin_set = set()
            for var in rsm_dat.var_edges:
                var_start_rsm_st, ret_rsm_st = rsm_dat.var_edges[var]

                inner_gss_node = self.gss.get_node(var_start_rsm_st, sppfnode.node)
                post_pop_sppf_nodes = inner_gss_node.add_edge(ret_rsm_st, sppfnode.gssn)

                post_pop_sppf_nodes, sub_start_fin_set = self.filter_poped_nodes(
                    post_pop_sppf_nodes, sppfnode
                )

                self.add_sppf_nodes(post_pop_sppf_nodes)
                self.add_sppf_nodes(
                    set([SPPFNode(inner_gss_node, var_start_rsm_st, sppfnode.node)])
                )

                start_fin_set.update(sub_start_fin_set)

            return start_fin_set

        def pop_step() -> Set[Tuple[int, int]]:
            new_sppf_nodes = sppfnode.gssn.pop(sppfnode.node)
            new_sppf_nodes, start_fin_set = self.filter_poped_nodes(
                new_sppf_nodes, sppfnode
            )
            self.add_sppf_nodes(new_sppf_nodes)
            return start_fin_set

        term_step()
        res_set = var_step()

        if rsm_dat.is_final:
            res_set.update(pop_step())

        return res_set

    def solve_reach(
        self,
        from_n: Set[int],
        nodes_end: Set[int],
    ) -> Set[Tuple[int, int]]:
        achievable_set = set()
        for snode in from_n:
            gssn = self.gss.get_node(self.start_rstate, snode)
            gssn.add_edge(RsmState(Symbol("$"), "fin"), self.accept_gssnode)

            self.add_sppf_nodes(set([SPPFNode(gssn, self.start_rstate, snode)]))

        while self.unprocessed != set():
            achievable_set.update(self.step(self.unprocessed.pop()))

        filtered_set = set()
        for st_fin in achievable_set:
            if st_fin[1] in nodes_end:
                filtered_set.add(st_fin)
        return filtered_set
