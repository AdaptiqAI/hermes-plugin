"""The account as `/adaptiq status` tells it, from `/v1/account/summary` (core's format.ts and the
DeepSeek Harness plugin's status command)."""

from __future__ import annotations

import time

CONSOLE = "https://memory.adaptiqai.com/memory"

# Say when the connection ends once it is this close; sign-in makes a 90-day key.
EXPIRY_NOTICE_DAYS = 14


def format_tokens(value: float) -> str:
    """A compact token count: 940, 24.6k, 1.2M, 5M."""
    if not isinstance(value, (int, float)) or value < 0:
        return "0"
    if value < 1000:
        return str(int(value + 0.5))
    if value < 1_000_000:
        return _trim(value / 1000, 1 if value < 100_000 else 0) + "k"
    return _trim(value / 1_000_000, 2 if value < 10_000_000 else 1) + "M"


def _trim(value: float, digits: int) -> str:
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_percent(fraction: float) -> str:
    """Percent of a 0-1 fraction that never shows 0% for something: 18%, <1%."""
    if not isinstance(fraction, (int, float)) or fraction <= 0:
        return "0%"
    if fraction < 0.01:
        return "<1%"
    return f"{min(100, int(fraction * 100 + 0.5))}%"


def _count(value) -> int | None:
    return value if isinstance(value, (int, float)) and value >= 0 else None


def _model_detail(model: dict, main: str | None) -> list[str]:
    parts = []
    if _count(model.get("context_window")):
        parts.append(f"{format_tokens(model['context_window'])} context")
    lanes = _count(model.get("lanes"))
    if lanes is not None:
        lane_text = f"{lanes} lane" if lanes == 1 else f"{lanes} lanes"
        if _count(model.get("running_now")) is not None:
            lane_text += f" ({model['running_now']} running, {_count(model.get('waiting_now')) or 0} waiting)"
        parts.append(lane_text)
    if model.get("status") == "offline":
        parts.append("offline")
    if model.get("id") == main:
        parts.append("main model")
    return parts


def _model_line(model: dict, main: str | None) -> str:
    return "• " + " · ".join([str(model.get("display_name") or model.get("id")), *_model_detail(model, main)])


def _memory_text(memory: dict) -> str | None:
    if not (memory.get("available", True) and _count(memory.get("limit_tokens"))):
        return None
    fraction = memory.get("fraction", memory.get("usage_ratio"))
    if not isinstance(fraction, (int, float)):
        fraction = (memory.get("used_tokens") or 0) / memory["limit_tokens"]
    return (f"{format_tokens(memory.get('used_tokens') or 0)} of {format_tokens(memory['limit_tokens'])} "
            f"({format_percent(fraction)})")


def _history(memory: dict) -> float | None:
    """Session-history fullness, where the server reports it apart from the library."""
    ratio = memory.get("session_usage_ratio")
    if isinstance(ratio, (int, float)) and ratio >= 0:
        return ratio
    used, limit = _count(memory.get("session_tokens")), _count(memory.get("session_limit_tokens"))
    return used / limit if used is not None and limit else None


def sleep_lines(view: dict | None, console: str | None = None) -> list[str]:
    """What `GET /v1/workspace/session-memory` says a person should know, in quiet lines.

    Idle conversations sleep on their own to make room: compiled memory is dropped, the transcript
    is kept, and they wake when used. Nothing is ever deleted automatically, so this asks for
    nothing; only an ask-mode suggestion names where to decide.
    """
    if not isinstance(view, dict):
        return []
    lines = []
    sleeping = sum(1 for row in view.get("sessions") or [] if isinstance(row, dict) and row.get("hibernated"))
    if sleeping == 1:
        lines.append("An idle conversation is sleeping to make room. Nothing is deleted; it wakes when you use it.")
    elif sleeping > 1:
        lines.append(f"{sleeping} idle conversations are sleeping to make room. Nothing is deleted; "
                     "they wake when you use them.")
    plan = view.get("plan") if isinstance(view.get("plan"), dict) else {}
    candidates = [row for row in plan.get("candidates") or [] if isinstance(row, dict)]
    if plan.get("state") == "suggest" and plan.get("mode", "ask") == "ask" and candidates:
        ratio = plan.get("usage_ratio")
        full = f"{format_percent(ratio)} full" if isinstance(ratio, (int, float)) else "almost full"
        lines.append(f"Conversation memory is {full}. {len(candidates)} conversations you haven't used for a while "
                     f"can sleep to make room. Nothing is deleted. Open the console: {console or CONSOLE}")
    return lines


def _ends(summary: dict, now: float | None) -> str | None:
    """When the connection ends, in its last two weeks."""
    expires = _count((summary.get("connection") or {}).get("expires_at"))
    if not expires:
        return None
    days = int((expires - (time.time() if now is None else now)) // 86400)
    if days >= EXPIRY_NOTICE_DAYS:
        return None
    return "today" if days <= 0 else ("tomorrow" if days == 1 else f"in {days} days")


def _name_of(summary: dict, model_id: str) -> str:
    for model in [*(summary.get("models") or []), *(summary.get("available_models") or [])]:
        if isinstance(model, dict) and model.get("id") == model_id and model.get("display_name"):
            return str(model["display_name"])
    return model_id


def status_text(summary: dict, *, conversation: str | None = None, main: str | None = None,
                moved_from: str | None = None, now: float | None = None, sleep: dict | None = None) -> str:
    """What `/adaptiq status` says: the essentials on the first line, then the models.

    `conversation` is "saved" or "private" inside a conversation, `main` the AdaptiqAI model Hermes
    starts on, `moved_from` the main model that had no lanes any more, `sleep` the session-memory view.
    """
    workspace = (summary.get("workspace") or {}).get("label") or "your workspace"
    first = [f"AdaptiqAI: connected to {workspace}"]
    if conversation:
        first.append(f"this conversation is {conversation}")
    memory = summary.get("memory") or {}
    if _memory_text(memory) and _history(memory) is not None:
        # used/limit are the library; session history has its own budget beside it.
        first.append(f"library {_memory_text(memory)}")
        first.append(f"conversations {format_percent(_history(memory))}")
    elif _memory_text(memory):
        first.append(f"memory {_memory_text(memory)}")
    lines = [" · ".join(first)]
    models = [model for model in summary.get("models") or [] if isinstance(model, dict)]
    if models:
        lines.extend(_model_line(model, main) for model in models)
    else:
        lines.append("No models yet. /adaptiq models shows the models you can get.")
    if moved_from and main:
        lines.append(f"Your main model was {_name_of(summary, moved_from)}, which has no lanes any more. "
                     f"It is {_name_of(summary, main)} now.")
    conversations, knowledge = _count(memory.get("conversations_tokens")), _count(memory.get("knowledge_tokens"))
    if memory.get("available") and conversations is not None and knowledge is not None and _history(memory) is None:
        lines.append(f"Memory split: conversations {format_tokens(conversations)}, knowledge {format_tokens(knowledge)}")
    if _history(memory) is not None:  # only a server that reports history apart puts conversations to sleep
        lines.extend(sleep_lines(sleep, (summary.get("links") or {}).get("console")))
    if _ends(summary, now):
        lines.append(f"This connection ends {_ends(summary, now)}. /adaptiq login renews it.")
    return "\n".join(lines)


def view(summary: dict, *, main: str | None = None, now: float | None = None) -> dict:
    """The account as Hermes Desktop's settings page shows it: the same facts as `/adaptiq status`."""
    models = [model for model in summary.get("models") or [] if isinstance(model, dict)]
    ends = _ends(summary, now)
    return {
        "workspace": (summary.get("workspace") or {}).get("label") or "your workspace",
        "memory": _memory_text(summary.get("memory") or {}),
        "models": [{"id": model.get("id"), "name": str(model.get("display_name") or model.get("id")),
                    "detail": " · ".join(_model_detail(model, main))} for model in models],
        "ends": f"This connection ends {ends}." if ends else None,
    }
