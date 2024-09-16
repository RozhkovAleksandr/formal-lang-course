from dataclasses import dataclass
import cfpq_data
import networkx


@dataclass
class GraphInfo:
    nodes: int
    edges: int
    labels: list[str]


def get_graph_info(name):
    path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(path)
    return GraphInfo(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        cfpq_data.get_sorted_labels(graph),
    )


def save_to_pydot_labeled_two_cycles_graph(n, m, labels, path):
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    networkx.drawing.nx_pydot.write_dot(graph, path)


save_to_pydot_labeled_two_cycles_graph(3, 2, ["a", "b"], "a.dot")
