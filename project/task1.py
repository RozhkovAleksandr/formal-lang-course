from dataclasses import dataclass
import cfpq_data
import networkx


@dataclass
class Graph:
    edges_cnt: int
    nodes_cnt: int
    labels: list
    

def graph_info(name):
    path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(path)
    return Graph(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        cfpq_data.get_sorted_labels(graph),
    )

def make_and_save_a_graph_of_two_cycles(n, m, labels, path):
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    pydot_graph = networkx.drawing.nx_pydot.to_pydot(graph)
    pydot_graph.write_raw(path)
