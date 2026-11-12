"""Earlier conversations by `@` (row U7, as DeepSeek Harness mentions them).

Hermes lets a plugin add an `@<prefix>:` reference (`ctx.register_context_reference`): the TUI and Desktop offer it
under `@` and complete its values, and every surface expands it in the message before the turn. `@adaptiq:` names an
earlier conversation on AdaptiqAI; its turn then adds that conversation to this one, so the model reads it from
memory instead of a transcript (core's recallConversations): only one on the same model, never while this
conversation is private, kept for later turns, and waited for at most RECALL_WAIT seconds.
"""
from __future__ import annotations

import re
import threading
import time

from . import knowledge
from . import tools

PREFIX = "adaptiq"
DESCRIPTION = "AdaptiqAI: an earlier conversation, read from memory"
# Hermes' reference value: a quoted name, or anything up to the next space, without trailing punctuation.
MENTION = re.compile(r"""(?<![\w/])@adaptiq:(?:`([^`\n]+)`|"([^"\n]+)"|'([^'\n]+)'|(\S+))""")
TRAILING = ",.;!?"
RECALL_WAIT = 8.0
RECALL_POLL = 1.0
LIST_FOR = 30
LIST_TIMEOUT = 4

_lock = threading.Lock()
_lists: dict[tuple[str, str], tuple[float, list[dict]]] = {}


def targets(text) -> list[str]:
    """The conversations a message mentions, each once, in order."""
    found: list[str] = []
    for match in MENTION.finditer(text if isinstance(text, str) else ""):
        quoted = next((group for group in match.groups()[:3] if group), None)
        value = (quoted if quoted is not None else match.group(4).rstrip(TRAILING)).strip()
        if value and value not in found:
            found.append(value)
    return found


def quoted(value: str) -> str:
    """A name as Hermes reads a reference value back whole (its `format_reference_value`)."""
    try:
        from agent.context_references import format_reference_value
        return format_reference_value(value)
    except ImportError:
        if not re.search(r"""[\s()\[\]{}<>"'`]""", value):
            return value
        return next((f"{quote}{value}{quote}" for quote in ("`", '"', "'") if quote not in value), value)


def earlier(model: str, conversation: str, fresh: bool = False) -> list[dict]:
    """The earlier conversations on `model` this one could read, newest first, read at most every LIST_FOR."""
    key = (model, conversation)
    with _lock:
        cached = _lists.get(key)
    if cached is not None and not fresh and time.monotonic() - cached[0] < LIST_FOR:
        return cached[1]
    items = knowledge.conversations(model, conversation, timeout=LIST_TIMEOUT)["items"]
    with _lock:
        _lists[key] = (time.monotonic(), items)
        while len(_lists) > 32:
            _lists.pop(next(iter(_lists)))
    return items


def resolve(items: list[dict], target: str) -> dict | list[dict] | None:
    """The conversation a mention means: its id, else its exact name, else the only one whose name holds it; a list
    when several do (never guessed)."""
    by_id = [item for item in items if item["conversation_id"] == target]
    if by_id:
        return by_id[0]
    wanted = tools.same_name(target)
    if not wanted:
        return None
    exact = [item for item in items if tools.same_name(item["name"]) == wanted]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return exact
    partial = [item for item in items if wanted in tools.same_name(item["name"])]
    return partial[0] if len(partial) == 1 else partial or None


def provider(where):
    """Hermes' `@adaptiq:` provider. `where()` is (conversation, model) of the chat being typed in, as the plugin
    knows it; ("", None) when no chat ran on AdaptiqAI. None outside a Hermes that has references."""
    try:
        from agent.context_references import ContextCompletionItem, ContextReferenceProvider
    except ImportError:
        return None

    class Conversations(ContextReferenceProvider):
        prefix = PREFIX
        description = DESCRIPTION

        async def autocomplete(self, query: str, *, limit: int = 10) -> list:
            conversation, model = where()
            if not model:
                return []
            try:
                items = earlier(model, conversation)
            except knowledge.KnowledgeError:
                return []
            wanted = tools.same_name(query or "")
            shown = [item for item in items if not wanted or wanted in tools.same_name(item["name"])]
            return [ContextCompletionItem(text=quoted(item["name"]), display=item["name"],
                                          meta="AdaptiqAI conversation") for item in shown[:limit]]

        async def expand(self, target: str) -> str | None:
            conversation, model = where()
            if not model:
                return f'"{target}": no conversation has run on AdaptiqAI here yet, so none can be read.'
            try:
                found = resolve(earlier(model, conversation), target)
            except knowledge.KnowledgeError as error:
                return f'"{target}" could not be looked up: {error}'
            if isinstance(found, list):
                names = ", ".join(f'"{item["name"]}"' for item in found[:5])
                return f'Several earlier conversations are called "{target}": {names}. Mention one by its full name.'
            if found is None:
                return f'No earlier AdaptiqAI conversation on this model is called "{target}".'
            return f'The earlier AdaptiqAI conversation "{found["name"]}".'

    return Conversations()
