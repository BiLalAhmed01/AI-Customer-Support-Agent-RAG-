"""Characterization tests for the pure, side-effect-free pieces of
src/llm.py. Anything that constructs a real provider client (get_llm) or
calls a live API is out of scope here — no network access in this suite.
"""

from src.llm import cap_history


def test_cap_history_keeps_most_recent_n():
    history = [{"role": "user", "content": str(i)} for i in range(20)]
    result = cap_history(history, max_messages=4)
    assert [m["content"] for m in result] == ["16", "17", "18", "19"]


def test_cap_history_no_op_when_under_the_cap():
    history = [{"role": "user", "content": str(i)} for i in range(3)]
    result = cap_history(history, max_messages=10)
    assert result == history


def test_cap_history_zero_disables_cap():
    history = [{"role": "user", "content": str(i)} for i in range(20)]
    result = cap_history(history, max_messages=0)
    assert result == history


def test_cap_history_empty_list():
    assert cap_history([], max_messages=4) == []


def test_cap_history_uses_settings_default_when_max_messages_omitted():
    from src.config import settings

    history = [{"role": "user", "content": str(i)} for i in range(100)]
    result = cap_history(history)
    assert len(result) == min(100, settings.max_history_messages)
