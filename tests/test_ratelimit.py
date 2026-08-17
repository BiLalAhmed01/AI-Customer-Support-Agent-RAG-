"""Tests for src/ratelimit.py. session_state is faked with a plain dict
since these functions only use setdefault/get/[] — the real
st.session_state supports the same protocol. settings is a frozen
dataclass instance (see src/config.py), so individual fields can't be
monkeypatched directly — the whole module-level `settings` name is
replaced instead, same as st.session_state below.
"""

from types import SimpleNamespace

import src.ratelimit as ratelimit_module


def _patch_session_state(monkeypatch):
    fake = {}
    monkeypatch.setattr(ratelimit_module.st, "session_state", fake)
    return fake


def _patch_settings(monkeypatch, **overrides):
    defaults = dict(
        chat_rate_limit_count=20,
        chat_rate_limit_window_seconds=60,
        upload_rate_limit_mb=50,
        upload_rate_limit_window_seconds=3600,
    )
    defaults.update(overrides)
    monkeypatch.setattr(ratelimit_module, "settings", SimpleNamespace(**defaults))


def test_chat_rate_limit_allows_up_to_the_count(monkeypatch):
    _patch_session_state(monkeypatch)
    _patch_settings(monkeypatch, chat_rate_limit_count=3, chat_rate_limit_window_seconds=60)

    for _ in range(3):
        allowed, message = ratelimit_module.check_chat_rate_limit()
        assert allowed is True
        assert message == ""


def test_chat_rate_limit_rejects_after_the_count(monkeypatch):
    _patch_session_state(monkeypatch)
    _patch_settings(monkeypatch, chat_rate_limit_count=2, chat_rate_limit_window_seconds=60)

    ratelimit_module.check_chat_rate_limit()
    ratelimit_module.check_chat_rate_limit()
    allowed, message = ratelimit_module.check_chat_rate_limit()

    assert allowed is False
    assert message


def test_chat_rate_limit_disabled_when_zero(monkeypatch):
    _patch_session_state(monkeypatch)
    _patch_settings(monkeypatch, chat_rate_limit_count=0)

    for _ in range(100):
        allowed, message = ratelimit_module.check_chat_rate_limit()
        assert allowed is True


def test_chat_rate_limit_expires_old_entries(monkeypatch):
    state = _patch_session_state(monkeypatch)
    _patch_settings(monkeypatch, chat_rate_limit_count=1, chat_rate_limit_window_seconds=60)

    ratelimit_module.check_chat_rate_limit()
    # Simulate the one recorded timestamp being far outside the window.
    state["_chat_request_times"][0] -= 120

    allowed, _ = ratelimit_module.check_chat_rate_limit()
    assert allowed is True


def test_upload_rate_limit_allows_under_the_cap(monkeypatch):
    _patch_session_state(monkeypatch)
    _patch_settings(monkeypatch, upload_rate_limit_mb=10, upload_rate_limit_window_seconds=3600)

    allowed, message = ratelimit_module.check_upload_rate_limit(5 * 1024 * 1024)
    assert allowed is True
    assert message == ""


def test_upload_rate_limit_rejects_over_the_cap(monkeypatch):
    _patch_session_state(monkeypatch)
    _patch_settings(monkeypatch, upload_rate_limit_mb=10, upload_rate_limit_window_seconds=3600)

    ratelimit_module.check_upload_rate_limit(8 * 1024 * 1024)
    allowed, message = ratelimit_module.check_upload_rate_limit(5 * 1024 * 1024)

    assert allowed is False
    assert message


def test_upload_rate_limit_disabled_when_zero(monkeypatch):
    _patch_session_state(monkeypatch)
    _patch_settings(monkeypatch, upload_rate_limit_mb=0)

    allowed, _ = ratelimit_module.check_upload_rate_limit(999 * 1024 * 1024)
    assert allowed is True
