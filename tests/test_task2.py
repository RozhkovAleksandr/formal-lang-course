from project.task2 import regex_to_dfa


def test_regex_to_dfa():
    regex = "a|c*"

    dfa = regex_to_dfa(regex)

    assert dfa.accepts("a")
    assert dfa.accepts("c")
    assert dfa.accepts("ccc")
    assert dfa.accepts("")


def test_regex_to_dfa2():
    dfa = regex_to_dfa("a*|b")
    assert dfa.is_deterministic()
    assert dfa.accepts("a")
    assert dfa.accepts("b")
    assert dfa.accepts("aaa")
    assert not dfa.accepts("bb")
