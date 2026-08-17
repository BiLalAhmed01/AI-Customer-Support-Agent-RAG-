"""Lightweight per-session rate limiting.

Session-scoped, using Streamlit's own session_state — no new dependency,
no external store (Redis, etc.). This bounds accidental or automated
hammering *within* a session; it resets on a full page reload/new tab, so
it is not a defense against a determined attacker cycling sessions. Real
protection against that belongs at a reverse-proxy/gateway layer in front
of the deployment, not in application code — this is the best available
lightweight mitigation without adding infrastructure, aimed at the
concrete audit finding (unthrottled LLM-cost and disk-fill exposure), not
a claim of production-grade abuse protection.
"""

import time

import streamlit as st

from src.config import settings


def check_chat_rate_limit() -> tuple[bool, str]:
    """Returns (allowed, message). message is "" when allowed.

    Call this before invoking the LLM pipeline for a new user message;
    only counts calls that actually reach this check (a rejected message
    is not itself counted again).
    """
    limit = settings.chat_rate_limit_count
    if limit <= 0:
        return True, ""

    window = settings.chat_rate_limit_window_seconds
    now = time.monotonic()
    timestamps: list[float] = st.session_state.setdefault("_chat_request_times", [])

    cutoff = now - window
    while timestamps and timestamps[0] < cutoff:
        timestamps.pop(0)

    if len(timestamps) >= limit:
        return False, (
            f"You've sent {limit} messages in the last {window} seconds. "
            "Please wait a moment before asking again."
        )

    timestamps.append(now)
    return True, ""


def check_upload_rate_limit(new_bytes: int) -> tuple[bool, str]:
    """Returns (allowed, message). message is "" when allowed.

    Call this before writing an uploaded file to disk, with the size of
    that specific file — on top of the existing per-file
    settings.max_upload_mb cap, this bounds *cumulative* upload volume in
    a session.
    """
    limit_mb = settings.upload_rate_limit_mb
    if limit_mb <= 0:
        return True, ""

    window = settings.upload_rate_limit_window_seconds
    now = time.monotonic()
    log: list[tuple[float, int]] = st.session_state.setdefault("_upload_bytes_log", [])

    cutoff = now - window
    while log and log[0][0] < cutoff:
        log.pop(0)

    limit_bytes = limit_mb * 1024 * 1024
    total_with_new = sum(b for _, b in log) + new_bytes
    if total_with_new > limit_bytes:
        minutes = max(1, window // 60)
        return False, (
            f"This would exceed the {limit_mb} MB upload limit per "
            f"{minutes} minute(s) for this session. Please wait before "
            "uploading more."
        )

    log.append((now, new_bytes))
    return True, ""
