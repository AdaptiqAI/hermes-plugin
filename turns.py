"""The person's turns in a conversation, each marked when it answers the agent's offer to save knowledge
(core's turns.ts). The plugin reads Hermes' history into said messages, so a plain "yes" right after the
agent's offer counts as the person's request, the same as in every other harness.

A said message is a dict: the person's `{"from": "person", "reply": "typed" | "question", "text"}`
("question" for an answer to a question the agent asked through clarify), or the agent's
`{"from": "agent", "text", "tool_arguments": [...]}`, its tool calls' arguments as it wrote them.
"""

from __future__ import annotations

import re

# An agent's question that offers to save knowledge: it names a collection (or knowledge, NCM).
OFFER = re.compile(r"(?<![^\W\d_])(?:knowledge|collection|ncm|viden|samling)|知识|集合", re.I)
# What an offer asks to do with it: save, make, add, keep... ("Based on my knowledge, should I run the tests?"
# asks nothing of it).
OFFER_DOES = re.compile(
    r"(?<![^\W\d_])(?:save|make|create|add|put|keep|store|turn|remember|gem|gemme|lav|lave|opret|oprette|tilføj|"
    r"tilføje|læg|lægge)(?![^\W\d_])|保存|创建|做成|放进|加入|存", re.I)
# Each question in a text, up to its question mark.
QUESTIONS = re.compile(r"[^.!?。！？\n]*[?？]")


def offers_knowledge(question: str) -> bool:
    """Whether a question offers to save knowledge: it names knowledge and asks to save, make or add it."""
    return bool(OFFER.search(question)) and bool(OFFER_DOES.search(question))


def _offered(agent: dict, reply: str) -> bool:
    """Whether the agent's message offers to save knowledge, for the kind of reply that answers it. A typed
    reply answers the message's last question, so that question must name knowledge: "the collection is
    mutated. Want me to fix it?" offers nothing. An answer through clarify answers the question the tool
    carried, in its arguments; other tools' arguments (a command, a URL) never make an offer."""
    if reply == "question":
        return any(offers_knowledge(question)
                   for question in QUESTIONS.findall("\n".join(agent.get("tool_arguments") or [])))
    questions = QUESTIONS.findall(agent.get("text") or "")
    return offers_knowledge(questions[-1] if questions else "")


def user_turns(messages: list[dict]) -> list[dict]:
    """The person's turns, oldest first, each `{"text", "after_offer"}`: after_offer when it answers the
    agent's offer to save knowledge (the agent asked right before). The agent's last message with text or a
    tool call before the person's reply decides; an empty message (a bare reasoning step) does not."""
    turns: list[dict] = []
    asking: dict | None = None
    for message in messages:
        if message.get("from") == "agent":
            if (message.get("text") or "").strip() or message.get("tool_arguments"):
                asking = message
            continue
        text = message.get("text") or ""
        if not text.strip():
            continue
        turns.append({"text": text,
                      "after_offer": asking is not None and _offered(asking, message.get("reply") or "typed")})
        asking = None
    return turns
