import pytest
from project.task1 import *
import networkx


def test_graph_info():
    actual_graph = graph_info("travel")
    expected = Graph(
        131,
        277,
        [
            "type",
            "subClassOf",
            "first",
            "rest",
            "disjointWith",
            "onProperty",
            "someValuesFrom",
            "domain",
            "range",
            "comment",
            "equivalentClass",
            "intersectionOf",
            "differentFrom",
            "hasValue",
            "oneOf",
            "minCardinality",
            "inverseOf",
            "hasPart",
            "hasAccommodation",
            "unionOf",
            "complementOf",
            "versionInfo"
        ],
    )
    assert actual_graph == expected


def test_two_cycles_graph():
    make_and_save_a_graph_of_two_cycles(3, 2, ["a", "b"], "test_graph.dot")
    graph = networkx.DiGraph(networkx.drawing.nx_pydot.read_dot("test_graph.dot"))
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