from project.task2 import regex_to_dfa


def test_regex_to_dfa():
    regex = "a|c*"

    dfa = regex_to_dfa(regex)

    assert dfa.accepts("a")
    assert dfa.accepts("c")
    assert dfa.accepts("ccc")
    assert dfa.accepts("")
