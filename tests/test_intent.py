"""Characterization tests for src/intent.py's deterministic routing.

These lock in *current* behavior (including priority ordering between
overlapping patterns) so a future edit to _RULES can't silently change
which intent a message resolves to without a test failing. Not a
correctness judgment on the rules themselves.
"""

from src.intent import Intent, classify_intent, get_profile


def test_human_escalation():
    assert classify_intent("I want to speak with a human") == Intent.HUMAN_ESCALATION
    assert classify_intent("get me a real person please") == Intent.HUMAN_ESCALATION


def test_cancellation_beats_ordering_priority():
    # "cancel my order" contains both "cancel" (CANCELLATION) and "order"
    # (ORDERING) — CANCELLATION must win because it's checked first.
    assert classify_intent("I want to cancel my order") == Intent.CANCELLATION


def test_order_tracking():
    assert classify_intent("Where is my package?") == Intent.ORDER_TRACKING
    assert classify_intent("can I track my shipment") == Intent.ORDER_TRACKING


def test_returns_refunds():
    assert classify_intent("I want to return this item") == Intent.RETURNS_REFUNDS
    assert classify_intent("how do refunds work") == Intent.RETURNS_REFUNDS


def test_account_support():
    assert classify_intent("I forgot my password") == Intent.ACCOUNT_SUPPORT


def test_shipping():
    assert classify_intent("how fast does shipping take") == Intent.SHIPPING


def test_ordering():
    assert classify_intent("I want to buy this product") == Intent.ORDERING


def test_pricing():
    assert classify_intent("how much does this cost") == Intent.PRICING


def test_product_info():
    assert classify_intent("what are the product specifications") == Intent.PRODUCT_INFO


def test_general_info():
    assert classify_intent("what are your hours") == Intent.GENERAL_INFO


def test_unknown_fallback():
    assert classify_intent("asdkjasjdk qqqq") == Intent.UNKNOWN


def test_empty_string_is_unknown():
    assert classify_intent("") == Intent.UNKNOWN
    assert classify_intent("   ") == Intent.UNKNOWN


def test_case_insensitive():
    assert classify_intent("CAN I TRACK MY ORDER") == Intent.ORDER_TRACKING


def test_escalation_has_retrieval_query_override():
    profile = get_profile(Intent.HUMAN_ESCALATION)
    assert profile.use_retrieval is True
    assert profile.retrieval_query_override is not None


def test_general_info_has_no_retrieval_query_override():
    profile = get_profile(Intent.GENERAL_INFO)
    assert profile.retrieval_query_override is None


def test_every_intent_has_a_profile():
    # get_profile would KeyError if any Intent member were missing from
    # INTENT_PROFILES — this catches that at test time instead of at
    # first-user-request time in production.
    for intent in Intent:
        profile = get_profile(intent)
        assert profile.next_action_hint
