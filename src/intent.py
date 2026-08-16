"""Deterministic, rule-based intent routing.

No LLM call, no ML model — just keyword/phrase matching. This is
intentionally simple: it decides *whether retrieval is needed* and *what
kind of next-best-action guidance* to give the generation step, not what
the final answer says. The generation step (llm.py) still does the actual
answering, grounded in retrieved context.

Priority matters: checks run in order and the first match wins, most
specific first. This is what stops "cancel my order" from being
misclassified as ORDERING just because it contains the word "order".
"""

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    GENERAL_INFO = "general_info"
    PRODUCT_INFO = "product_info"
    PRICING = "pricing"
    ORDERING = "ordering"
    SHIPPING = "shipping"
    ORDER_TRACKING = "order_tracking"
    RETURNS_REFUNDS = "returns_refunds"
    CANCELLATION = "cancellation"
    ACCOUNT_SUPPORT = "account_support"
    HUMAN_ESCALATION = "human_escalation"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentProfile:
    label: str
    use_retrieval: bool
    # True for intents that commonly involve asking Orchis to actually *do*
    # something (look up a specific order, cancel it, issue a refund,
    # change an account) rather than just explain a policy. Orchis has no
    # tools connected to any real backend, so these always get an explicit
    # "I can't do that yet, here's how you can" framing.
    live_action_intent: bool
    next_action_hint: str
    # For intents with a fixed, known informational target regardless of
    # how the user phrases it (e.g. escalation always wants the same
    # contact-channel info), search with this canonical phrase instead of
    # the user's raw wording. Deterministic, not a rewrite via the LLM —
    # used only where the target genuinely doesn't vary by phrasing; most
    # intents leave this None and search on the user's own words.
    retrieval_query_override: str | None = None


INTENT_PROFILES: dict[Intent, IntentProfile] = {
    Intent.GENERAL_INFO: IntentProfile(
        label="General information",
        use_retrieval=True,
        live_action_intent=False,
        next_action_hint=(
            "This looks like a general question about the company. After "
            "answering, mention one or two related things you can also help "
            "with (products, ordering, shipping, or support hours)."
        ),
    ),
    Intent.PRODUCT_INFO: IntentProfile(
        label="Product information",
        use_retrieval=True,
        live_action_intent=False,
        next_action_hint=(
            "This is a product question. After answering, offer to share "
            "pricing or explain how to order this product next."
        ),
    ),
    Intent.PRICING: IntentProfile(
        label="Pricing",
        use_retrieval=True,
        live_action_intent=False,
        next_action_hint=(
            "This is a pricing question. After answering, offer to explain "
            "how to place an order for the product."
        ),
    ),
    Intent.ORDERING: IntentProfile(
        label="Ordering",
        use_retrieval=True,
        live_action_intent=False,
        next_action_hint=(
            "This is a question about placing an order. After answering, "
            "offer to walk them through the next step or answer a follow-up "
            "ordering question — but you cannot place an order yourself."
        ),
    ),
    Intent.SHIPPING: IntentProfile(
        label="Shipping",
        use_retrieval=True,
        live_action_intent=False,
        next_action_hint=(
            "This is a shipping question. After answering, mention that "
            "they can track their order once it ships."
        ),
    ),
    Intent.ORDER_TRACKING: IntentProfile(
        label="Order tracking",
        use_retrieval=True,
        live_action_intent=True,
        next_action_hint=(
            "This is an order-tracking request. You have no order-lookup "
            "tool, so you cannot check any specific order's live status — "
            "say that plainly. Explain the documented tracking process "
            "(tracking email, account order history), and suggest "
            "contacting support if they need their specific status now."
        ),
    ),
    Intent.RETURNS_REFUNDS: IntentProfile(
        label="Returns/refunds",
        use_retrieval=True,
        live_action_intent=True,
        next_action_hint=(
            "This is a returns/refunds request. You have no order-management "
            "tool, so you cannot start a return or issue a refund yourself — "
            "say that plainly. Explain the documented return/refund process "
            "and exactly how they can start it themselves."
        ),
    ),
    Intent.CANCELLATION: IntentProfile(
        label="Cancellation",
        use_retrieval=True,
        live_action_intent=True,
        next_action_hint=(
            "This is a cancellation request. You have no order-management "
            "tool, so you cannot cancel an order yourself — say that "
            "plainly. Explain the documented cancellation policy and tell "
            "them exactly how and how fast to request it (e.g. contacting "
            "support immediately if there's a time window)."
        ),
    ),
    Intent.ACCOUNT_SUPPORT: IntentProfile(
        label="Account/support",
        use_retrieval=True,
        live_action_intent=True,
        next_action_hint=(
            "This is an account-related request. You have no account-"
            "management tool, so you cannot change account details or reset "
            "a password yourself — say that plainly. Explain the documented "
            "self-service steps."
        ),
    ),
    Intent.HUMAN_ESCALATION: IntentProfile(
        label="Human escalation",
        use_retrieval=True,
        live_action_intent=True,
        next_action_hint=(
            "The user wants a human. You cannot connect them to one "
            "yourself — acknowledge that plainly and give the exact "
            "documented contact channels and hours (live chat, email, "
            "phone) so they can reach a person directly."
        ),
        # However the user phrases an escalation request ("real person",
        # "get me an agent", "human please"), the informational target is
        # always the same contact-channel section — search for that
        # directly rather than the literal wording, which the reranker
        # otherwise scores poorly (it reads as a capability question, not
        # an information request, against a passage about live-chat hours).
        retrieval_query_override="how to contact support live chat email phone business hours",
    ),
    Intent.UNKNOWN: IntentProfile(
        label="Unknown/out-of-scope",
        use_retrieval=True,
        live_action_intent=False,
        next_action_hint=(
            "If nothing relevant turns up in the context, don't just refuse "
            "— be honest that the current knowledge base doesn't cover this "
            "exact detail. Do not guess at what the user might actually "
            "want or invent a tangential connection to their question. If "
            "the question is unrelated to this business entirely (e.g. "
            "general trivia, travel planning, coding help, another "
            "company), say plainly that it's outside what you can help "
            "with here — do NOT answer it from general knowledge just "
            "because you know the answer. Either way, name a couple of the "
            "specific topics you genuinely can help with (e.g. products, "
            "pricing, ordering, shipping, returns, or account support) and "
            "invite them to ask about one of those, or a different "
            "specific question."
        ),
    ),
}

# Checked in order, first match wins. Tuples of (intent, [regex patterns]).
# Patterns are matched case-insensitively against the raw query.
_RULES: list[tuple[Intent, list[str]]] = [
    (
        Intent.HUMAN_ESCALATION,
        [
            r"\bhuman\b",
            r"\breal person\b",
            r"\bactual person\b",
            r"\bagent\b",
            r"\brepresentative\b",
            r"\bspeak (to|with) (someone|a person)\b",
            r"\btalk to (someone|a person)\b",
            r"\bescalate\b",
            r"\bcustomer service rep\b",
        ],
    ),
    (
        Intent.CANCELLATION,
        [r"\bcancel(l?ing|led)?\b", r"\bcall off\b"],
    ),
    (
        Intent.ORDER_TRACKING,
        [
            r"\btrack(ing)?\b",
            r"where('?s| is) my (order|package|shipment)",
            r"\bshipment status\b",
            r"\bpackage status\b",
            r"\border status\b",
        ],
    ),
    (
        Intent.RETURNS_REFUNDS,
        [r"\breturn(s|ed|ing)?\b", r"\brefund(s|ed|ing)?\b", r"\bexchange(s|d|ing)?\b", r"money back"],
    ),
    (
        Intent.ACCOUNT_SUPPORT,
        [
            r"\baccount\b",
            r"\bpassword\b",
            r"\blog[ -]?in\b",
            r"\bsign[ -]?in\b",
            r"\bmy (email|address|profile)\b",
        ],
    ),
    (
        Intent.SHIPPING,
        [r"\bshipping\b", r"\bship(s|ped|ping)?\b", r"\bdeliver(y|ed|ing)?\b", r"\barriv(e|al|ing)\b"],
    ),
    (
        Intent.ORDERING,
        [
            r"\border(s|ed|ing)?\b",
            r"\bbuy(ing)?\b",
            r"\bpurchase(s|d|ing)?\b",
            r"\bcheckout\b",
            r"\bcart\b",
            r"\bplace an order\b",
        ],
    ),
    (
        Intent.PRICING,
        [r"\bprice(s|d|ing)?\b", r"\bcost(s)?\b", r"how much", r"\bfee(s)?\b", r"\$"],
    ),
    (
        Intent.PRODUCT_INFO,
        [
            r"\bproduct(s)?\b",
            r"\bwidget\b",
            r"\bspec(s|ification)?\b",
            r"\bmaterial(s)?\b",
            r"\bcompatib(le|ility)\b",
            r"\bfeature(s)?\b",
        ],
    ),
    (
        Intent.GENERAL_INFO,
        [
            r"\bcompany\b",
            r"\babout (you|acme|the company)\b",
            r"who are you",
            r"what (is|are) (acme|this company|you)\b",
            r"\bwhat do you (sell|offer|do)\b",
            r"\bhours\b",
            r"\bcontact\b",
            r"\bsupport hours\b",
        ],
    ),
]


def classify_intent(query: str) -> Intent:
    """Return the first matching intent, or UNKNOWN if nothing matches.

    Deterministic — no model call. Greetings, thanks, and vague requests
    ("hi", "I need help") naturally fall through to UNKNOWN here; that's
    fine, because the generation step's system prompt already handles those
    conversationally regardless of intent classification.
    """
    text = query.strip()
    if not text:
        return Intent.UNKNOWN

    for intent, patterns in _RULES:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return intent

    return Intent.UNKNOWN


def get_profile(intent: Intent) -> IntentProfile:
    return INTENT_PROFILES[intent]
