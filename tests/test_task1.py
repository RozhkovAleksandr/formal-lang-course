import pytest
from project.task1 import (
    save_to_pydot_labeled_two_cycles_graph,
    get_graph_info,
    GraphInfo
)
import networkx


def test_graph_info():
    actual_graph = get_graph_info("generations")
    expected = GraphInfo(
        129,
        273,
        [
            "type",
            "first",
            "rest",
            "onProperty",
            "intersectionOf",
            "equivalentClass",
            "someValuesFrom",
            "hasValue",
            "hasSex",
            "hasChild",
            "hasParent",
            "inverseOf",
            "sameAs",
            "hasSibling",
            "oneOf",
            "range",
            "versionInfo",
        ],
    )
    assert actual_graph == expected


def test_graph_info_fail():
    with pytest.raises(Exception):
        get_graph_info("None")


def test_save_to_pydot_labeled_two_cycles_graph(tmp_path):
    path = tmp_path / "graph.dot"
    save_to_pydot_labeled_two_cycles_graph(3, 2, ["a", "b"], path)
    graph = networkx.DiGraph(networkx.drawing.nx_pydot.read_dot(path))
    edges = [
        (1, 2, dict(label="a")),
        (2, 3, dict(label="a")),
        (3, 0, dict(label="a")),
        (0, 1, dict(label="a")),
        (0, 4, dict(label="b")),
        (4, 5, dict(label="b")),
        (5, 0, dict(label="b")),
    ]
    expected_graph = networkx.DiGraph(edges)
    assert networkx.is_isomorphic(
        graph, expected_graph, edge_match=dict.__eq__, node_match=dict.__eq__
    )