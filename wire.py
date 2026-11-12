"""What each Hermes request to AdaptiqAI carries (the DeepSeek Harness plugin's wire.ts, for Hermes).

A turn of a conversation carries the conversation's Hermes session id, so AdaptiqAI keeps one native
memory per conversation. Everything else that reaches AdaptiqAI runs once, ephemeral, and leaves
nothing in memory:

- helper calls such as titles, compression, approvals and vision, through the provider entry's
  `extra_body`;
- forks that replay a conversation under its id, such as `/btw` and the background review.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

SESSION_HEADER = "X-Adaptiq-Session-Id"
# Asks the engine for `adaptiq.progress` frames while it prepares a conversation's memory (core's adapter): Hermes'
# stream reader takes each as a chunk without choices, so a long preparation keeps the stream alive instead of
# reading as a stalled one.
PROGRESS_HEADER = "X-Adaptiq-Progress-Events"
# Every header the gateway reads as a session id. A fork must carry none of them.
SESSION_HEADER_ALIASES = frozenset({
    "x-adaptiq-session-id", "x-session-id", "x-litellm-session-id", "x-conversation-id", "x-chat-id",
    "x-thread-id", "x-openwebui-chat-id", "x-openwebui-conversation-id", "x-openwebui-session-id",
})
MEMORY_FIELD = "adaptiq_native_memory"
# A subagent names the conversation that started it (core's relations.ts): `helper` works for it
# inside the same task, `track` is a line of work of its own.
PARENT_HEADER = "X-Adaptiq-Parent-Session-Id"
ROLE_HEADER = "X-Adaptiq-Session-Role"
RELATION_HEADERS = frozenset({PARENT_HEADER.lower(), ROLE_HEADER.lower()})
SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def relation_headers(session_id: str, parent: str, role: str = "helper") -> dict:
    """The parent headers of a conversation's request; none for a root conversation, an invalid id
    or a conversation naming itself, which are dropped rather than sent."""
    if not (isinstance(session_id, str) and SESSION_ID.fullmatch(session_id)
            and isinstance(parent, str) and SESSION_ID.fullmatch(parent)) or parent == session_id:
        return {}
    return {PARENT_HEADER: parent, ROLE_HEADER: "track" if role == "track" else "helper"}


def ephemeral(kind: str) -> dict:
    """Not a turn of a conversation: it runs once and nothing is saved."""
    return {"capture": False, "capture_assistant": False, "persist_inputs": False, "timeline_mode": "ephemeral",
            "metadata": {"client_framework": "hermes", "helper_call": kind}}


def private(stored: bool) -> dict:
    """A turn the person keeps private. A stored conversation is read back without saving anything;
    one the server does not have yet runs once, ephemeral."""
    return {"capture": False, "capture_assistant": False, "persist_inputs": False,
            "timeline_mode": "read_only" if stored else "ephemeral",
            "metadata": {"conversation_persistence": "private", "client_framework": "hermes"}}


@dataclass
class Turn:
    session_id: str
    remember: bool
    # Whether AdaptiqAI stores the conversation; looked up once, for a private turn only.
    stored: bool | None = None
    # The AdaptiqAI conversation, when it is not the Hermes session: a compaction that continues in a
    # new session id stays in the memory of the conversation it continues.
    conversation: str = ""
    # The AdaptiqAI conversation of the session that started this one (a subagent's parent), and how
    # it relates to it.
    parent: str = ""
    role: str = "helper"

    @property
    def memory_id(self) -> str:
        return self.conversation or self.session_id


class Turns:
    """The turns Hermes runs for a person, by turn id.

    `pre_llm_call` names them. Hermes skips that hook for forks, which replay a conversation under
    the same session id, so a request whose turn is not here is a fork. A turn in use is the newest:
    each request of it keeps it, so a long turn is never dropped for the turns that started after it
    (a gateway runs many conversations, and subagents start turns of their own), and its later
    requests are saved like its first.
    """

    def __init__(self, keep: int = 4096) -> None:
        self._turns: OrderedDict[str, Turn] = OrderedDict()
        self._keep = keep
        self._lock = threading.Lock()

    def note(self, turn_id: str, turn: Turn) -> None:
        with self._lock:
            self._turns[turn_id] = turn
            self._turns.move_to_end(turn_id)
            while len(self._turns) > self._keep:
                self._turns.popitem(last=False)

    def get(self, turn_id: str) -> Turn | None:
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is not None:
                self._turns.move_to_end(turn_id)
            return turn


# Hermes' names for its thinking levels, as core names them.
_LEVELS = {"none": "off", "disabled": "off"}


def card_input(card: dict) -> list[str]:
    """What a model reads, from its capability card (core's inputFromCard): text when the card says nothing."""
    raw = card.get("input") if isinstance(card.get("input"), list) else None
    modes = [mode for mode in raw or [] if mode in ("text", "image")]
    return modes or ["text"]


def card_reasoning(card) -> tuple[bool, dict | None]:
    """How a model thinks, from its capability card (core's reasoningFromCard): (known, spec). Not known when the
    card says nothing about it; spec None when the model does not think."""
    if not isinstance(card, dict) or "reasoning" not in card:
        return False, None
    block = card["reasoning"]
    if block is False or block is None:
        return True, None
    if not isinstance(block, dict):
        return False, None
    levels = [level for level in block.get("levels") or [] if isinstance(level, str) and level and level != "off"]
    if not levels:
        return True, None
    off = block.get("off")
    off = off.strip() if isinstance(off, str) and off.strip() else None
    aliases = {name: to for name, to in (block.get("aliases") or {}).items() if isinstance(to, str) and to in levels} \
        if isinstance(block.get("aliases"), dict) else {}
    default = block.get("default").strip() if isinstance(block.get("default"), str) else None
    field = block.get("field").strip() if isinstance(block.get("field"), str) and block.get("field").strip() else ""
    return True, {"levels": levels, "off": off, "aliases": aliases, "field": field or "reasoning_effort",
                  "default": default if default in levels or (default == "off" and off) else None}


def wire_level(spec: dict, level: str | None) -> str | None:
    """The value to send for a level (core's wireLevel), or None to send none (the server's default)."""
    if not level:
        return None
    level = _LEVELS.get(level, level)
    if level == "off":
        return spec["off"]
    return level if level in spec["levels"] else spec["aliases"].get(level)


def by_card(request: dict, row: dict | None, chosen: bool) -> dict:
    """A request to AdaptiqAI as the model's card asks (core's buildChatBody): its thinking level in the card's field
    and words, the card's default where the person chose none (Hermes then sends medium), and the card's output cap
    where Hermes sends none. Without a card, the request is left as Hermes made it."""
    card = row.get("adaptiq_capabilities") if isinstance(row, dict) else None
    known, spec = card_reasoning(card)
    shaped = dict(request)
    body = dict(shaped.get("extra_body") or {})
    if known:
        sent = shaped.pop("reasoning_effort", None) or body.pop("reasoning_effort", None)
        value = None
        if spec is not None:
            level = sent if chosen or sent not in (None, "medium") else (spec["default"] or sent)
            value = wire_level(spec, level if isinstance(level, str) else None)
        if value is not None and spec["field"] == "reasoning_effort":
            shaped["reasoning_effort"] = value
        elif value is not None:
            body[spec["field"]] = value
    if body:
        shaped["extra_body"] = body
    else:
        shaped.pop("extra_body", None)
    cap = next((value for value in ((card or {}).get("max_output_tokens"), (row or {}).get("max_output_tokens"))
                if isinstance(value, int) and not isinstance(value, bool) and value > 0), None)
    window = (row or {}).get("context_window")
    if cap and "max_tokens" not in shaped and "max_completion_tokens" not in shaped:
        shaped["max_tokens"] = min(cap, window) if isinstance(window, int) and window > 0 else cap
    return shaped


def shape(request: dict, turn: Turn | None) -> dict:
    """The request with AdaptiqAI's session header and memory mode.

    Pass `turn=None` for anything that is not a turn of a conversation.
    """
    shaped = dict(request)
    headers = {name: value for name, value in (request.get("extra_headers") or {}).items()
               if name.lower() not in SESSION_HEADER_ALIASES and name.lower() not in RELATION_HEADERS}
    body = dict(request.get("extra_body") or {})
    if turn is None:
        body[MEMORY_FIELD] = ephemeral("fork")
    else:
        headers[PROGRESS_HEADER] = "true"
        headers[SESSION_HEADER] = turn.memory_id
        headers.update(relation_headers(turn.memory_id, turn.parent, turn.role))
        if turn.remember:
            # The provider entry marks every request ephemeral for the helper calls; a turn that is
            # remembered sends no memory mode, which saves it.
            body.pop(MEMORY_FIELD, None)
        else:
            body[MEMORY_FIELD] = private(bool(turn.stored))
    for key, value in (("extra_headers", headers), ("extra_body", body)):
        if value:
            shaped[key] = value
        else:
            shaped.pop(key, None)
    return shaped
