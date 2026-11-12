"""The account as `/adaptiq status` tells it, from `/v1/account/summary` (core's format.ts and the
DeepSeek Harness plugin's status command)."""

from __future__ import annotations

import time

CONSOLE = "https://memory.adaptiqai.com/memory"

# Say when the connection ends once it is this close; sign-in makes a 90-day key.
EXPIRY_NOTICE_DAYS = 14
# A conversation memory this full is past where idle conversations sleep on their own, so none could: nothing gets
# ready until some are deleted (core's SESSION_FULL). On 2026-11-09 MiMo's stood at 99.98 % for an hour.
SESSION_FULL = 0.99


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


def memory_view(raw) -> dict | None:
    """The workspace memory as core's memoryView reads it: used/limit are the library (knowledge); session history,
    where the server reports it apart, is beside it. None when its size is not known."""
    if not isinstance(raw, dict):
        return None
    if raw.get("available") is False:
        return {"available": False}
    used, limit = _count(raw.get("used_tokens")), _count(raw.get("limit_tokens"))
    if used is None or not limit:
        return None
    session_limit, session_used = _count(raw.get("session_limit_tokens")), _count(raw.get("session_tokens"))
    session_ratio = _count(raw.get("session_usage_ratio"))
    session = ({"used": session_used, "limit": session_limit,
                "fraction": min(1, session_ratio if session_ratio is not None else session_used / session_limit)}
               if session_limit and session_used is not None else None)
    conversations = _count(raw.get("conversations_tokens"))
    if conversations is None:
        conversations = _count(raw.get("session_tokens"))
    knowledge = _count(raw.get("knowledge_tokens"))
    split = conversations is not None and knowledge is not None and conversations + knowledge == used
    return {"available": True, "used": used, "limit": limit, "fraction": min(1, used / limit),
            "full": raw.get("full") is True or used >= limit, "session": session,
            "conversations": conversations if split else None, "knowledge": knowledge if split else None}


def _model_detail(model: dict, main: str | None) -> str:
    """What follows a model's name on its line (core's statusText), with Hermes' main model marked."""
    window = f" · {format_tokens(model['context_window'])} context" if _count(model.get("context_window")) else ""
    live = (f" ({model['running_now']} running, {_count(model.get('waiting_now')) or 0} waiting)"
            if _count(model.get("running_now")) is not None else "")
    offline = " · offline" if model.get("status") == "offline" else ""
    return f"{window}{live}{offline}{' · main model' if main and model.get('id') == main else ''}"


def _model_name(model: dict) -> str:
    return str(model.get("display_name") or model.get("id"))


def _model_line(model: dict, main: str | None) -> str:
    return f"• {_model_name(model)}{_model_detail(model, main)}"


def held_memory(summary: dict, model: str | None = None) -> tuple[dict | None, str | None]:
    """The memory a conversation's status shows, and the model it is on: the model's own store where the summary
    reports one per model, else the workspace's."""
    for row in summary.get("models") or []:
        if isinstance(row, dict) and model and row.get("id") == model:
            own = memory_view(row.get("memory"))
            if own is not None and own["available"]:
                return own, _model_name(row)
    return memory_view(summary.get("memory")), None


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


def memory_notice(summary: dict, console: str | None = None) -> tuple[int, str | None]:
    """The workspace memory's warning, in core's notice words, with where to act: (level, line). A server where idle
    conversations sleep warns for a full library and for a full conversation memory, which none could sleep to make
    room in (level 2); one shared pool warns from 90 % (1) and when full (2). (0, None) when there is nothing to say."""
    memory = memory_view(summary.get("memory"))
    link = console or (summary.get("links") or {}).get("console") or CONSOLE
    if memory is None or not memory["available"]:
        return 0, None
    if memory["session"] is not None:
        if not memory["full"] and memory["session"]["fraction"] >= SESSION_FULL:
            return 2, ("Memory is full. Nothing idle could sleep to make room. Delete or unpin a conversation, or add "
                       f"memory. Open the console: {link}")
        if not memory["full"]:
            return 0, None
        return 2, f"Your library is full. Remove a collection you no longer need, or add memory. Open the console: {link}"
    if memory["full"]:
        return 2, ("Workspace memory is full. New messages can't be saved until you free up space. "
                   f"Open the console: {link}")
    if memory["fraction"] >= 0.9:
        return 1, (f"Workspace memory is {format_percent(memory['fraction'])} full. {format_tokens(memory['used'])} of "
                   f"{format_tokens(memory['limit'])} used across all conversations and knowledge. Open the console: {link}")
    return 0, None


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


def _memory_part(memory: dict | None, on: str | None) -> str | None:
    """The memory on the status line's first line (core's statusText)."""
    if memory is None or not memory["available"]:
        return None
    history = "" if memory["session"] is None else f", conversations {format_percent(memory['session']['fraction'])}"
    return (f"library {format_tokens(memory['used'])} of {format_tokens(memory['limit'])} "
            f"({format_percent(memory['fraction'])}){history}{f' on {on}' if on else ''}")


def conversation_part(conversation: dict) -> str:
    """This conversation on the status line: its memory, whether it saves, and the knowledge it uses."""
    tokens = conversation.get("memory_tokens")
    held = "not saved yet" if tokens is None else f"{format_tokens(tokens)} in memory"
    knowledge = conversation.get("knowledge") or 0
    return (f"this conversation {held}, {'saving' if conversation.get('remember', True) else 'private'}"
            f"{f', {knowledge} knowledge' if knowledge else ''}")


def unconnected_text(phase: str, user_code: str | None = None, command: str = "/adaptiq") -> str:
    """The status when Hermes is not connected: signing in, expired, or never connected."""
    if phase == "signing-in":
        return f"Waiting for approval in your browser. Code: {user_code or ''}"
    said = "Your AdaptiqAI sign-in expired." if phase == "expired" else "Not connected to AdaptiqAI."
    return f"{said} Run {command} signin."


def status_text(summary: dict, *, conversation: dict | None = None, model: str | None = None,
                main: str | None = None, moved_from: str | None = None, now: float | None = None,
                sleep: dict | None = None, api_key: bool = False, command: str = "/adaptiq") -> str:
    """What `/adaptiq status` says (core's statusText): the essentials on the first line, in order, then the models.

    `conversation` is this conversation on AdaptiqAI (`memory_tokens`, `remember`, `knowledge`), `model` the model it
    runs on, `main` the AdaptiqAI model Hermes starts on, `moved_from` the main model that left the workspace,
    `sleep` the session-memory view, `api_key` whether the connection is an API key rather than this computer's
    sign-in.
    """
    workspace = (summary.get("workspace") or {}).get("label") or "your workspace"
    first = [f"Connected to {workspace}{' (API key)' if api_key else ''}"]
    if conversation is not None:
        first.append(conversation_part(conversation))
    memory, on = held_memory(summary, model)
    if _memory_part(memory, on):
        first.append(_memory_part(memory, on))
    lines = [" · ".join(first)]
    models = [row for row in summary.get("models") or [] if isinstance(row, dict)]
    if models:
        lines.extend(_model_line(row, main) for row in models)
    else:
        lines.append(f"No models yet. Open the AdaptiqAI console with {command} open.")
    if memory is not None and memory["available"] and memory["conversations"] is not None:
        lines.append(f"Memory split: conversations {format_tokens(memory['conversations'])}, "
                     f"knowledge {format_tokens(memory['knowledge'])}")
    if moved_from and main:
        lines.append(f"Your main model was {_name_of(summary, moved_from)}, which is no longer in your workspace. "
                     f"It is {_name_of(summary, main)} now.")
    if memory is not None and memory["available"] and memory["session"] is not None:
        # Only a server that reports history apart puts idle conversations to sleep.
        lines.extend(sleep_lines(sleep, (summary.get("links") or {}).get("console")))
    warning = memory_notice(summary)[1]
    if warning:
        lines.append(warning)
    if _ends(summary, now):
        lines.append(f"This connection ends {_ends(summary, now)}. {command} signin renews it.")
    return "\n".join(lines)


def view(summary: dict, *, main: str | None = None, now: float | None = None, api_key: bool = False,
         sleep: dict | None = None) -> dict:
    """The account as Hermes Desktop's settings page and status chip show it: the same facts as `/adaptiq status`,
    and the notices it raises (the memory warning, sleeping conversations)."""
    models = [row for row in summary.get("models") or [] if isinstance(row, dict)]
    memory = memory_view(summary.get("memory"))
    ends = _ends(summary, now)
    held = memory is not None and memory["available"]
    level, warning = memory_notice(summary)
    return {
        "warning": {"level": level, "text": warning} if warning else None,
        "sleeping": (sleep_lines(sleep, (summary.get("links") or {}).get("console"))
                     if held and memory["session"] is not None else []),
        "workspace": (summary.get("workspace") or {}).get("label") or "your workspace",
        "api_key": api_key,
        "memory": _memory_part(memory, None),
        "used": memory["used"] if held else None,
        "limit": memory["limit"] if held else None,
        "split": (f"Conversations {format_tokens(memory['conversations'])}, knowledge {format_tokens(memory['knowledge'])}"
                  if held and memory["conversations"] is not None else None),
        "models": [{"id": row.get("id"), "name": _model_name(row), "detail": _model_detail(row, main).lstrip(" ·")}
                   for row in models],
        "ends": f"This connection ends {ends}." if ends else None,
    }
