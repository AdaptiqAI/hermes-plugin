"""A conversation's knowledge on AdaptiqAI: what it reads, the library, and using a collection
(core's workspace.ts, its knowledge and conversation parts).

A collection is built for one model and works only with that model, so the library is read from the
store of the model the conversation runs on.
"""

from __future__ import annotations

import base64
import contextlib
import http.client
import json
import os
import threading
import time
import urllib.parse
from datetime import datetime, timezone

from . import adaptiq_core as core
from . import record
from .errors import CONSOLE

PAGE = 50
# How many found lines of one collection come with their JSON record; a record takes a few searches.
RECORDS_SHOWN = 3
# How much record text one find shows, over all its collections: what open shows at most.
RECORD_CHARS = 16_000
# Local files one request may carry, and the most the gateway takes in one request body.
FILES_LIMIT = 6 * 1024 * 1024
BODY_LIMIT = 8 * 1024 * 1024 - 4096
MEDIA = {
    ".pdf": "application/pdf", ".txt": "text/plain", ".md": "text/markdown", ".markdown": "text/markdown",
    ".json": "application/json", ".csv": "text/csv", ".html": "text/html", ".htm": "text/html",
    ".xml": "application/xml", ".yaml": "application/yaml", ".yml": "application/yaml",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
SOURCE_CODE = frozenset(
    ".ts .tsx .js .jsx .mjs .cjs .py .rs .go .java .kt .swift .rb .php .c .h .cc .cpp .hpp .cs .sh .sql .toml "
    ".ini .cfg .log .tex .r .scala .lua .vue .svelte .css .scss".split())


class KnowledgeError(Exception):
    """One sentence for the agent or the person. `answer` is what AdaptiqAI answered ("403 scope_required"),
    empty when it gave no answer; `passing` says another try may go through. `status` is the HTTP status, and
    `missing_route` says the gateway has no such route yet (core's isMissingRoute)."""

    def __init__(self, message: str, answer: str = "", passing: bool = False, status: int | None = None,
                 missing_route: bool = False) -> None:
        super().__init__(message)
        self.answer = answer
        self.passing = passing
        self.status = status
        self.missing_route = missing_route


def _raw(method: str, path: str, model: str | None = None, body: dict | None = None,
         headers: dict | None = None, timeout: float = 15) -> tuple[int, dict]:
    if core.key() is None:
        raise KnowledgeError("AdaptiqAI is not connected. Ask the user to run /adaptiq signin.")
    headers = {**({"x-adaptiq-model": model} if model else {}), **(headers or {})}
    try:
        return core.authed(method, path, body=body, headers=headers or None, timeout=timeout)
    except (OSError, ValueError) as error:
        raise KnowledgeError("AdaptiqAI can't be reached right now. Try again in a moment.", passing=True) from error


def _call(method: str, path: str, model: str | None = None, body: dict | None = None,
          allow: tuple[int, ...] = (), headers: dict | None = None, timeout: float = 15) -> dict | None:
    """One knowledge call: the body, None for an allowed status, else a KnowledgeError."""
    status, reply = _answered(method, path, model, body, headers, timeout)
    if status in allow:
        return None
    if status >= 400:
        detail = reply.get("detail") if isinstance(reply, dict) else None
        code = detail.get("code") if isinstance(detail, dict) else None
        bare = detail.strip().lower() if isinstance(detail, str) else ""
        raise KnowledgeError(_failure(status, reply), answer=" ".join(str(part) for part in (status, code) if part),
                             passing=status in (408, 425, 429) or status >= 500 or code == "instance_unavailable",
                             status=status,
                             missing_route=status == 405 or (status == 404 and bare in ("not found", "route not found")))
    return reply


# Inside an agent tool's call, passing trouble is asked again before the agent hears of it (core's
# tool-failures.ts): a 5xx, no answer in time or a dropped connection after 1 and then 3 seconds, within 15
# seconds of the call's first failure; a conversation getting ready (425) after 5, 10 and 20 seconds, within
# 40, as a model request waits for it. Only what is safe to send twice is sent again, and a model that is
# offline is not waited for: it comes back in minutes. Hermes' requests keep no headers, so a Retry-After does
# not lengthen these waits. Outside a tool's call (a turn's note, the account) every request is sent once.
RETRY_WAITS = (1.0, 3.0)
RETRY_WINDOW = 15.0
READY_WAITS = (5.0, 10.0, 20.0)
READY_WINDOW = 40.0
OFFLINE_CODES = ("instance_unavailable", "session_route_unavailable")
# What the engine calls a conversation or collection getting ready (core's READY_WORDS).
READY_WORDS = ("preflightbusy", "preparing", "memorynotready", "upload_capacity")


class Tries:
    """One tool call's tries, shared by every request it sends (core's Tries)."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep) -> None:
        self.clock = clock
        self.sleep = sleep
        self.first: float | None = None
        # A request of this call failed every try it had: the others are not asked again.
        self.lasted = False
        # The call has ended: what it left running is sent once.
        self.ended = False

    def wait_after(self, ready: bool, tried: int) -> float | None:
        """How long to wait before the next try, or None when the failure goes back to the call."""
        if self.ended or self.lasted:
            return None
        now = self.clock()
        if self.first is None:
            self.first = now
        waits, window = (READY_WAITS, READY_WINDOW) if ready else (RETRY_WAITS, RETRY_WINDOW)
        if tried >= len(waits) or now + waits[tried] - self.first > window:
            self.lasted = True
            return None
        return waits[tried]


_calls = threading.local()


@contextlib.contextmanager
def calling(clock=time.monotonic, sleep=time.sleep):
    """The requests of one agent tool's call: passing trouble is asked again in them (`_answered`)."""
    outer = getattr(_calls, "tries", None)
    tries = _calls.tries = Tries(clock, sleep)
    try:
        yield tries
    finally:
        tries.ended = True
        _calls.tries = outer


def getting_ready(status: int | None, named: str) -> bool:
    """A conversation getting ready: a 425, or a 503 the engine names as such (core's gettingReady)."""
    return status == 425 or (status == 503 and any(word in named.lower() for word in READY_WORDS))


def _trouble(status: int, reply) -> str | None:
    """`ready` for a conversation getting ready, `blip` for other passing trouble worth asking again within
    seconds, None for an answer: a refusal, or a 503 that says why (a model offline, a memory service the gateway
    gave up on), which comes back in minutes, not within a call. A bare 503 is a blip."""
    detail = reply.get("detail") if isinstance(reply, dict) else None
    error = reply.get("error") if isinstance(reply, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None
    kind = error.get("type") if isinstance(error, dict) else None
    named = f"{kind or ''} {code or ''}"
    if getting_ready(status, named):
        return "ready"
    if code in OFFLINE_CODES or (status == 503 and (named.strip() or "offline" in json.dumps(detail).lower())):
        return None
    if status in (408, 429) or (status >= 500 and status != 507):
        return "blip"
    return None


def _answered(method: str, path: str, model: str | None = None, body: dict | None = None,
              headers: dict | None = None, timeout: float = 15) -> tuple[int, dict]:
    """`_raw`, asked again on passing trouble inside an agent tool's call. A reply cut off on the way is no
    answer: AdaptiqAI could not be reached."""
    tries = getattr(_calls, "tries", None)
    safe = method in ("GET", "HEAD", "PUT", "DELETE") or any(name.lower() == "idempotency-key" for name in headers or {})
    tried = 0
    while True:
        error: Exception | None = None
        try:
            status, reply = _raw(method, path, model, body, headers, timeout)
            kind = _trouble(status, reply)
        except KnowledgeError as caught:
            if not caught.passing:
                raise
            error, kind = caught, "blip"
        except http.client.HTTPException as caught:
            error, kind = caught, "blip"
        wait = tries.wait_after(kind == "ready", tried) if kind and safe and tries is not None else None
        if wait is not None:
            tries.sleep(wait)
            tried += 1
            if not tries.ended:
                continue
        if error is None:
            return status, reply
        if isinstance(error, KnowledgeError):
            raise error
        raise KnowledgeError("AdaptiqAI can't be reached right now.", passing=True) from error


def _failure(status: int, reply) -> str:
    detail = reply.get("detail") if isinstance(reply, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None
    message = detail.get("message") if isinstance(detail, dict) else detail if isinstance(detail, str) else None
    if code == "model_mismatch" and message:
        return message  # the gateway names both models
    if code == "instance_unavailable" or status in (502, 503, 504):
        return "This model is offline right now, so its knowledge can't be read. Try again when it is back."
    if status == 401:
        return "The AdaptiqAI sign-in has expired. Ask the user to run /adaptiq signin."
    if status == 403:
        return "This AdaptiqAI connection may not do that. Ask the user to run /adaptiq signin."
    return message or f"AdaptiqAI answered {status}."


def _path(session_id: str, *rest: str) -> str:
    return "/".join(["/sessions", urllib.parse.quote(session_id, safe=""),
                     *(urllib.parse.quote(part, safe="") for part in rest)])


def status_of(memory: dict) -> tuple[str, str]:
    """Readiness of a collection in plain words (core's knowledgeStatus)."""
    preparation = memory.get("preparation") if isinstance(memory.get("preparation"), dict) else {}
    compatibility = str(preparation.get("compatibility") or "")
    status = str(memory.get("memory_status") or memory.get("status") or "")
    if compatibility == "current":
        return "ready", "Updating" if status == "updating" else "Ready"
    if compatibility == "preparing" or status in ("compiling", "preparing", "updating", "pending"):
        return "preparing", "Preparing"
    if compatibility == "different_model":
        return "not-ready", "Not ready for this model"
    if status == "failed" or compatibility == "failed":
        return "failed", "Preparation failed"
    if status in ("ready", "active"):
        return "ready", "Ready"
    return "not-ready", "Not prepared"


def _tokens(*values) -> int | None:
    return next((value for value in values if isinstance(value, (int, float)) and value >= 0), None)


def _date(value) -> str | None:
    if isinstance(value, (int, float)) and value > 0:
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds, timezone.utc).date().isoformat()
    return value[:10] if isinstance(value, str) and value else None


def _stamp(value) -> float:
    """A time in seconds, from seconds, milliseconds or an ISO date; 0 when there is none."""
    if isinstance(value, (int, float)) and value > 0:
        return value / 1000 if value > 1e11 else float(value)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def collection(memory: dict, selected: bool, target: str | None = None) -> dict:
    """One collection as the tools read it (core's knowledgeItem). `fits` says it works on `target`, the model
    the conversation runs on: a collection is built for one model and works only with that model."""
    origin = memory.get("origin") if isinstance(memory.get("origin"), dict) else {}
    preparation = memory.get("preparation") if isinstance(memory.get("preparation"), dict) else {}
    readiness, label = status_of(memory)
    model = memory.get("model")
    item = {
        "memory_id": str(memory.get("memory_id") or memory.get("id") or ""),
        "name": memory.get("name") or origin.get("name") or "Untitled",
        "kind": "conversation" if origin.get("type") == "conversation" else "collection",
        "readiness": readiness,
        "status": label,
        "selected": selected,
        "fits": preparation.get("compatibility") != "different_model" and (not target or not model or model == target),
        # Newest first: when it was made, else when it changed.
        "at": _stamp(memory.get("created_at")) or _stamp(memory.get("source_updated_at") or memory.get("updated_at")),
        # When its text last changed: lines open keeps are a version's.
        "changed": _stamp(memory.get("source_updated_at") or memory.get("updated_at")),
    }
    preview = memory.get("description") or origin.get("preview")
    if preview:
        item["preview"] = str(preview)[:280]
    tokens = _tokens(memory.get("logical_tokens"), (memory.get("timeline") or {}).get("logical_tokens"))
    if tokens is not None:
        item["logical_tokens"] = tokens
    updated = _date(memory.get("source_updated_at") or memory.get("updated_at"))
    if updated:
        item["updated"] = updated
    return item


def _refusal(status: int, reply) -> KnowledgeError:
    """A failed answer as `_call` raises it: its sentence, what AdaptiqAI answered, and whether another try may go
    through, so the tools tell passing trouble from a refusal."""
    detail = reply.get("detail") if isinstance(reply, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None
    return KnowledgeError(_failure(status, reply), answer=" ".join(str(part) for part in (status, code) if part),
                          passing=status in (408, 425, 429) or status >= 500 or code == "instance_unavailable",
                          status=status)


def conversation(session_id: str, model: str | None = None, timeout: float = 15) -> dict | None:
    """What AdaptiqAI keeps for a conversation, or None while it is not stored.

    The gateway sends a session call to the model the conversation is placed on. Where several models
    keep memory and it cannot tell (409), the conversation's own model is asked.
    """
    status, reply = _answered("GET", _path(session_id), timeout=timeout)
    if status == 409 and model:
        status, reply = _answered("GET", _path(session_id), model, timeout=timeout)
        if status == 409 and _not_on(reply, model):
            return None
    if status == 404:
        return None
    if status >= 400:
        raise _refusal(status, reply)
    return reply


def _not_on(reply, model: str) -> bool:
    """Whether a 409 says the conversation is not on `model`. Any other 409 is a refusal the person
    should hear, not a conversation to store."""
    detail = reply.get("detail") if isinstance(reply, dict) and isinstance(reply.get("detail"), dict) else {}
    models = detail.get("models")
    return detail.get("code") in ("memory_route_required", "session_not_on_model") or (
        isinstance(models, list) and model not in models)


def conversation_view(session_id: str, model: str | None) -> dict:
    """This conversation's size, knowledge and attached conversations."""
    body = conversation(session_id, model)
    if body is None:
        return {"stored": False, "memory_tokens": None, "knowledge": [], "conversations": []}
    session = body.get("session") if isinstance(body.get("session"), dict) else {}
    timeline = session.get("timeline") if isinstance(session.get("timeline"), dict) else {}
    sizes = [value for value in (timeline.get("compiled_tokens"), timeline.get("logical_tokens"))
             if isinstance(value, (int, float))]
    served = session.get("model") or model
    attached = _call("GET", _path(session_id, "conversations"), model=served, allow=(404,)) or {}
    conversations = []
    for row in attached.get("data") or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").lower()
        readiness = ("preparing" if status in ("compiling", "preparing", "pending", "queued", "building", "processing")
                     else "failed" if status in ("failed", "error", "compile_failed") else "ready")
        entry = {"conversation_id": row.get("conversation_id") or row.get("session_id"),
                 "name": row.get("display_name") or row.get("name") or "Untitled conversation",
                 "readiness": readiness}
        if isinstance(row.get("compiled_tokens"), (int, float)):
            entry["logical_tokens"] = row["compiled_tokens"]
        conversations.append(entry)
    return {
        "stored": True,
        "model": served,
        "memory_tokens": max(sizes) if sizes else None,
        "knowledge": [collection(binding, True) for binding in body.get("bindings") or [] if isinstance(binding, dict)],
        "conversations": conversations,
    }


def collections_of(conversation_id: str, model: str | None) -> list[str] | None:
    """The names of the collections an earlier conversation reads, or None when it can't be read now (core's
    collectionsOfConversations reads them)."""
    try:
        body = _call("GET", _path(conversation_id), model=model, allow=(404,))
    except KnowledgeError:
        return None
    if not isinstance(body, dict):
        return None
    bindings = body.get("bindings") if isinstance(body.get("bindings"), list) else []
    return [item["name"] for item in (collection(binding, True) for binding in bindings if isinstance(binding, dict))
            if item["kind"] == "collection"]


def conversations(model: str, session_id: str, query: str | None = None, timeout: float = 15) -> dict:
    """Earlier conversations on `model` this one could read from memory (core's listConversations): newest first,
    with something in them, never this one. `attachable` is False where this connection may not list them."""
    status, reply = _raw("GET", "/sessions?" + urllib.parse.urlencode({"limit": str(PAGE)}), model, timeout=timeout)
    if status == 403:
        return {"attachable": False, "items": []}
    if status >= 400:
        raise KnowledgeError(_failure(status, reply))
    needle = (query or "").strip().lower()
    items = []
    for row in reply.get("data") or [] if isinstance(reply, dict) else []:
        if not isinstance(row, dict):
            continue
        conversation_id = row.get("conversation_id") or row.get("session_id") or row.get("id")
        if not isinstance(conversation_id, str) or not conversation_id or conversation_id == session_id:
            continue
        name = row.get("display_name") or row.get("name")
        timeline = row.get("timeline") if isinstance(row.get("timeline"), dict) else {}
        tokens = _tokens(row.get("compiled_tokens"), timeline.get("compiled_tokens"), timeline.get("logical_tokens"))
        if tokens == 0 or (row.get("model") and row.get("model") != model):
            continue
        name = name if isinstance(name, str) and name.strip() and name != conversation_id else "Untitled conversation"
        if needle and needle not in name.lower():
            continue
        items.append({"conversation_id": conversation_id, "name": name, "attached": False,
                      "at": _stamp(row.get("updated_at") or row.get("attached_at"))})
    items.sort(key=lambda item: item["at"], reverse=True)
    return {"attachable": True, "items": items}


# One plain sentence for each refusal of the conversation routes (core's conversationFailure).
CONVERSATION_REFUSALS = {
    "self_reference": "A conversation can't read itself.",
    "cycle": "Those conversations would read each other in a loop.",
    "too_many_conversations": "A conversation can read up to 16 others. Remove one first.",
    "conversation_not_compiled": "That conversation isn't ready yet. Try again in a minute.",
    "conversation_not_found": "That conversation no longer exists.",
    "session_not_found": "Send a message in this conversation first.",
}


def read_conversation(session_id: str, model: str | None, conversation_id: str, attach: bool) -> None:
    """Let a conversation read an earlier one from memory, or stop (core's setConversation)."""
    if conversation_id == session_id:
        raise KnowledgeError("A conversation can't read itself.")
    path = _path(session_id, "conversations", conversation_id)
    status, reply = _raw("PUT" if attach else "DELETE", path, model, {"version": "current"} if attach else None)
    if status < 400:
        return
    detail = reply.get("detail") if isinstance(reply, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None
    if not attach and code == "conversation_not_attached":
        return
    if status == 405 or (status == 404 and isinstance(detail, str) and detail.strip().lower() in ("not found", "route not found")):
        raise KnowledgeError("Adding conversations is coming soon.")
    if code == "model_mismatch":
        raise KnowledgeError(str(detail.get("message") or "That conversation is on another model."))
    raise KnowledgeError(CONVERSATION_REFUSALS.get(code) or _failure(status, reply))


def library(model: str, query: str | None = None, cursor: str | None = None,
            selected: frozenset[str] = frozenset()) -> dict:
    """One page of the collections built for `model`."""
    params = {"limit": str(PAGE), "status": "all"}
    if query and query.strip():
        params["query"] = query.strip()[:200]
    if cursor:
        params["cursor"] = cursor
    body = _call("GET", "/memories?" + urllib.parse.urlencode(params), model=model) or {}
    pagination = body.get("pagination") if isinstance(body.get("pagination"), dict) else {}
    items = [collection(memory, str(memory.get("memory_id")) in selected, model)
             for memory in body.get("data") or [] if isinstance(memory, dict)]
    return {"items": items, "next_cursor": pagination.get("next_cursor"), "has_more": pagination.get("has_more") is True}


def read_here(session_id: str, model: str, timeout: float = 15) -> list[dict]:
    """The collections a conversation reads. One AdaptiqAI does not store yet is given, on its first request,
    each collection on its model whose rule for new sessions is always (the gateway's session rules)."""
    body = conversation(session_id, model, timeout)
    if body is not None:
        return [collection(binding, True) for binding in body.get("bindings") or [] if isinstance(binding, dict)]
    given, cursor = [], None
    for _ in range(4):
        params = {"limit": str(PAGE), "status": "all", **({"cursor": cursor} if cursor else {})}
        page = _call("GET", "/memories?" + urllib.parse.urlencode(params), model=model, timeout=timeout) or {}
        given += [collection(memory, True) for memory in page.get("data") or []
                  if isinstance(memory, dict) and memory.get("new_session_policy") == "always"]
        pagination = page.get("pagination") if isinstance(page.get("pagination"), dict) else {}
        cursor = pagination.get("next_cursor") if pagination.get("has_more") is True else None
        if not cursor:
            break
    return given


def files(model: str, memory_id: str, offset: int = 0) -> dict:
    """A page of the documents a collection holds now (core's knowledgeFiles), with the listing's version: an empty
    page after the last document has it too."""
    params = urllib.parse.urlencode({"offset": str(max(0, int(offset))), "limit": "200"})
    body = _call("GET", f"/memories/{urllib.parse.quote(memory_id, safe='')}/files?{params}", model=model) or {}
    found = [{"path": str(row["path"]), "lines": int(row.get("lines") or 0), "chars": _count(row.get("chars")) or 0,
              "version": body.get("version")}
             for row in body.get("files") or [] if isinstance(row, dict) and row.get("path")]
    after = body.get("next_offset")
    return {"documents": found, "total": int(body.get("total") or len(found)),
            "next_offset": after if isinstance(after, int) else None, "version": _count(body.get("version"))}


def failure(memory_id: str, model: str | None) -> str | None:
    """Why a collection could not be built, in the engine's words (core's knowledgeFailure): the list of collections
    leaves the reason out, the collection itself says it. None when it gives none or can't be read now."""
    try:
        body = _call("GET", f"/memories/{urllib.parse.quote(memory_id, safe='')}", model=model)
    except KnowledgeError:
        return None
    if not isinstance(body, dict):
        return None
    memory = body.get("memory") if isinstance(body.get("memory"), dict) else {}
    preparation = memory.get("preparation") if isinstance(memory.get("preparation"), dict) else {}
    versions = body.get("versions") if isinstance(body.get("versions"), list) else []
    last = versions[-1] if versions and isinstance(versions[-1], dict) else {}
    return next((value.strip() for value in (preparation.get("error"), last.get("error"))
                 if isinstance(value, str) and value.strip()), None)


def _grep(model: str, memory_id: str, pattern: str, path: str | None, limit: int, context: int,
          chars: int = 500) -> tuple[list[dict], bool]:
    """Lines of a collection's current text a regular expression matches, each the part of its line around the
    match with `context` lines on each side (core's searchKnowledge)."""
    params = {"kind": "grep", "pattern": pattern, "limit": str(limit), "chars": str(chars), "context": str(context)}
    if path:
        params["path"] = path
    query = urllib.parse.urlencode(params)
    body = _call("GET", f"/memories/{urllib.parse.quote(memory_id, safe='')}/search?{query}", model=model) or {}
    rows = [{"path": str(row["path"]), "line": int(row.get("line") or 0), "text": str(row.get("text") or ""),
             "before": [text if isinstance(text, str) else "" for text in row.get("before") or []],
             "after": [text if isinstance(text, str) else "" for text in row.get("after") or []]}
            for row in body.get("matches") or [] if isinstance(row, dict) and row.get("path")]
    return rows, body.get("truncated") is True


def search(model: str, memory_id: str, pattern: str, path: str | None = None) -> dict:
    """Lines of a collection's current text a regular expression matches, each the part of its line around
    the match with one line on each side (reading.ts's find)."""
    rows, truncated = _grep(model, memory_id, pattern, path, 25, 1)
    return {"matches": rows, "truncated": truncated}


def first_lines(model: str, memory_id: str, pattern: str, path: str) -> tuple[list[dict], bool]:
    """The first three lines at `path` that hold a pattern, each with the line after it, and whether more hold it
    (core's holdersOf asks so, a document at a time). The engine takes the path as a start, so a longer path
    beside it (staff.json beside staff.js) may give them all."""
    return _grep(model, memory_id, pattern, path, 3, 1, 160)


def records(model: str, memory_id: str, rows: list[dict]) -> dict[int, dict]:
    """The JSON records of the first found lines in JSON documents, by id(row) (record.py). A line near one
    already chosen is likely in the same record, so it is left out; a record that can't be read leaves its line
    as find gave it."""
    chosen: list[dict] = []
    for row in rows:
        if len(chosen) == RECORDS_SHOWN:
            break
        if record.is_json_document(row["path"]) and not any(
                other["path"] == row["path"] and abs(other["line"] - row["line"]) <= record.RECORD_SIDE for other in chosen):
            chosen.append(row)
    found: dict[int, dict] = {}
    for row in chosen:
        def read(expression: str, at: str = row["path"]) -> list[dict]:
            return [match for match in _grep(model, memory_id, expression, at, 100, record.RECORD_SIDE)[0] if match["path"] == at]
        try:
            whole = record.json_record(row, 1, read)
        except KnowledgeError:
            whole = None
        if whole is not None:
            found[id(row)] = whole
    return found


def _revision(session_id: str, model: str | None) -> int:
    """The conversation's revision, storing the conversation first when AdaptiqAI has none yet."""
    refused = None
    for _ in range(2):
        body = conversation(session_id, model)
        revision = ((body or {}).get("session") or {}).get("revision")
        if isinstance(revision, int):
            return revision
        if body is None:
            status, reply = _answered("POST", "/sessions", model, {"session_id": session_id, "name": "Hermes conversation"})
            # 409: it exists after all (read again), or a refusal to say if the read still finds nothing.
            if status == 409:
                refused = reply
            elif status >= 400:
                raise _refusal(status, reply)
    if refused:
        raise KnowledgeError(_failure(409, refused))
    raise KnowledgeError("This conversation is not on AdaptiqAI yet. Send one message first.")


def use(session_id: str, model: str | None, memory_id: str, attach: bool) -> None:
    """Start or stop using a collection in a conversation; the next model call reads the change."""
    path = _path(session_id, "memories", memory_id)
    for attempt in range(2):
        body: dict = {"expected_revision": _revision(session_id, model)}
        if attach:
            body.update({"activation_policy": "always", "availability_policy": "required",
                         "version_policy": "latest_ready"})
        status, reply = _answered("PUT" if attach else "DELETE", path, model, body)
        if status < 400:
            return
        detail = reply.get("detail") if isinstance(reply.get("detail"), dict) else {}
        # A revision that moved in between is read again once; a collection on another model is not.
        if status == 409 and detail.get("code") != "model_mismatch" and attempt == 0:
            continue
        raise _refusal(status, reply)


def is_text(content: bytes) -> bool:
    """Whether a file's bytes are text: UTF-8 without a NUL byte, as git tells text from binary."""
    if b"\0" in content:
        return False
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def media_type(name: str, content: bytes | None = None) -> str:
    """The media type for a file; source code and any other text are plain text. A text of another name (a
    Containerfile, a Makefile, JSON Lines, a .env) sent as bytes was refused by the engine ("is not a supported source
    type yet"), and the whole upload with it (core's mediaTypeOf)."""
    extension = os.path.splitext(name)[1].lower()
    if extension in MEDIA:
        return MEDIA[extension]
    if extension in SOURCE_CODE:
        return "text/plain"
    return "text/plain" if content is not None and is_text(content) else "application/octet-stream"


def unreadable(raw: str, real: str, blocked=None) -> str | None:
    """Why a file may not be sent, or None.

    A collection is read by everyone in the workspace, so what goes in is held to more than
    Hermes' own read rules: nothing hidden (.env, ~/.ssh, ~/.aws, ~/.hermes and every other
    dotfile or dot-folder), judged on the file a link points to, and nothing Hermes' file tools
    refuse (`blocked`, Hermes' get_read_block_error).
    """
    if any(part.startswith(".") for part in real.split(os.sep) if part):
        return f"{raw} is a hidden file or inside a hidden folder (such as .env or .ssh), so it is not sent."
    try:
        reason = blocked(real) if blocked else None
    except Exception:  # a check that cannot answer refuses
        reason = "could not be checked"
    return f"{raw} can't be sent: {reason}" if reason else None


def sources(items: list[dict], cwd: str, collection: str, resolve=None, blocked=None) -> list[dict]:
    """The API's sources for the tool's text, file and link sources (core's createKnowledge).

    Files are read from this computer: `resolve(raw)` turns a path into an absolute one as Hermes'
    file tools do (else relative to `cwd`), and `unreadable` decides what may go; the texts
    become one text source named after the collection; GitHub links are followed as repositories.
    """
    if not items or len(items) > 64:
        raise KnowledgeError("Add between 1 and 64 sources.")
    texts, out, size = [], [], 0
    for item in items:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind == "text":
            content = str(item.get("content") or "")
            if not content.strip():
                raise KnowledgeError("A text source is empty.")
            texts.append(f"# {item['name']}\n\n{content}" if item.get("name") else content)
        elif kind == "file":
            raw = str(item.get("path") or "")
            if not raw:
                raise KnowledgeError("A file source needs a path.")
            path = str(resolve(raw)) if resolve else os.path.join(cwd, os.path.expanduser(raw))
            # Judged on the file itself: a link or `..` that leads somewhere else is that somewhere.
            real = os.path.realpath(path)
            refusal = unreadable(raw, real, blocked)
            if refusal:
                raise KnowledgeError(refusal)
            if not os.path.isfile(real):
                raise KnowledgeError(f"{raw} is not a file.")
            size += os.path.getsize(real)
            if size > FILES_LIMIT:
                raise KnowledgeError("These files are too large for one request. Use fewer or smaller files, "
                                     "or a repository link.")
            with open(real, "rb") as handle:
                data = handle.read(FILES_LIMIT + 1)
            out.append({"type": "file", "name": str(item.get("name") or os.path.basename(path))[:200],
                        "media_type": media_type(path, data), "content_base64": base64.b64encode(data).decode()})
        elif kind == "link":
            url = urllib.parse.urlsplit(str(item.get("url") or ""))
            if url.scheme != "https" or not url.hostname or url.username or url.password:
                raise KnowledgeError("Use HTTPS links without credentials.")
            out.append({"type": "github", "url": url.geturl(), "ref": "HEAD", "version_policy": "pinned_commit",
                        "refresh_policy": "manual"} if url.hostname in ("github.com", "www.github.com")
                       else {"type": "auto", "url": url.geturl()})
        else:
            raise KnowledgeError("A source is text, file or link.")
    if texts:
        out.append({"type": "text", "name": collection, "content": "\n\n---\n\n".join(texts)})
    return out


def create_refusal(access: dict | None) -> str:
    """Why a connection cannot create knowledge, said plainly, from `GET /v1/cli-auth/knowledge/status` (core's
    createRefusal, with Hermes' own sign-in): its `reason`, with the owners named when it lists them. Without a
    reason, the words say nothing about where to go instead."""
    access = access if isinstance(access, dict) else {}
    reason = access.get("reason") if isinstance(access.get("reason"), str) else None
    listed = access.get("owners") if access.get("owners") is not None else access.get("owner")
    owners = []
    for owner in listed if isinstance(listed, list) else [] if listed is None else [listed]:
        if isinstance(owner, dict):
            owner = owner.get("name") if isinstance(owner.get("name"), str) else owner.get("email")
        if isinstance(owner, str) and owner:
            owners.append(owner)
    ask = f"ask {' or '.join(owners)}" if owners else "ask a workspace owner"
    if reason == "role":
        return f"You need editor rights in this workspace to create knowledge; {ask}."
    if reason == "connection":
        return "This computer's connection can't create knowledge yet. Sign in again with /adaptiq signin."
    if reason == "scope":
        return ("This connection's key can't create knowledge. Sign in again with /adaptiq signin, which gives this "
                "computer a connection that can.")
    return "This connection can't create knowledge in the workspace. Ask an owner, or create it in the console."


def create(name: str, description: str, body_sources: list[dict], *, model: str | None,
           idempotency_key: str) -> dict:
    """Create a collection in the store of `model`; the same key makes a retried request create it once. Its
    `sources` are what it holds now, all of it the agent's, so a correction later takes out only that (None when
    they could not be read)."""
    access = _call("GET", "/cli-auth/knowledge/status", allow=(404,))
    if access is not None and access.get("create_allowed") is False:
        raise KnowledgeError(create_refusal(access))
    body = {"name": name[:160], "description": description[:2000], "sources": body_sources}
    if len(json.dumps(body_sources)) > BODY_LIMIT:
        raise KnowledgeError("These files are too large for one upload. Add fewer or smaller files.")
    reply = _call("POST", "/memories", model=model, body=body, headers={"idempotency-key": idempotency_key},
                  timeout=120) or {}
    memory = reply.get("memory") if isinstance(reply.get("memory"), dict) else None
    if memory is None or not memory.get("memory_id"):
        raise KnowledgeError("AdaptiqAI did not confirm the new collection. Check Knowledge before trying again.")
    readiness, label = status_of(memory)
    store = memory.get("model") or model
    try:
        own = source_ids(str(memory["memory_id"]), store)
    except KnowledgeError:
        own = None
    return {"memory_id": str(memory["memory_id"]), "name": memory.get("name") or name, "readiness": readiness,
            "status": label, "model": store, "sources": own}


def source_ids(memory_id: str, model: str | None) -> list[str]:
    """The ids of the sources a collection holds now."""
    body = _call("GET", f"/memories/{urllib.parse.quote(memory_id, safe='')}/sources?limit=100", model=model) or {}
    return [str(row.get("source_id") or row.get("id")) for row in body.get("data") or []
            if isinstance(row, dict) and (row.get("source_id") or row.get("id"))]


def source_list(model: str, memory_id: str) -> list[dict]:
    """The sources a collection holds now, up to 2,000, with their names, types and sizes (core's
    knowledgeSources)."""
    found, cursor = [], None
    # The gateway counts 1,000 sources to a collection, yet the engine took a 1,001st.
    for _ in range(20):
        params = {"limit": "100", **({"cursor": cursor} if cursor else {})}
        body = _call("GET", f"/memories/{urllib.parse.quote(memory_id, safe='')}/sources?"
                     + urllib.parse.urlencode(params), model=model) or {}
        for row in body.get("data") or []:
            if isinstance(row, dict) and isinstance(row.get("source_id"), str) and isinstance(row.get("name"), str):
                found.append({"id": row["source_id"], "name": row["name"], "type": str(row.get("source_type") or ""),
                              "bytes": _count(row.get("source_bytes")),
                              "compile_bytes": _count(row.get("compile_bytes"))})
        pagination = body.get("pagination") if isinstance(body.get("pagination"), dict) else body
        cursor = pagination.get("next_cursor") if isinstance(pagination.get("next_cursor"), str) else None
        if not cursor:
            break
    return found


def _count(value) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None


def source_text(model: str, memory_id: str, source_id: str) -> dict:
    """The start of the engine's text of a source, the lines its search numbers, and whether that is all of it
    (core's knowledgeSourceText)."""
    body = _call("GET", f"/memories/{urllib.parse.quote(memory_id, safe='')}/sources/"
                 f"{urllib.parse.quote(source_id, safe='')}", model=model) or {}
    text = body.get("compile_preview")
    return {"text": text if isinstance(text, str) else "", "whole": body.get("preview_truncated") is False}


def original(model: str, memory_id: str, source_id: str) -> bytes:
    """A source's original, as it was added (core's knowledgeOriginal)."""
    body = _call("GET", f"/memories/{urllib.parse.quote(memory_id, safe='')}/sources/"
                 f"{urllib.parse.quote(source_id, safe='')}/original", model=model, timeout=60) or {}
    if not isinstance(body.get("content_base64"), str):
        raise KnowledgeError("AdaptiqAI gave no original for this source.")
    return base64.b64decode(body["content_base64"])


def add_sources(memory_id: str, model: str | None, body_sources: list[dict], idempotency_key: str) -> None:
    """Add sources to a collection: it compiles a new version."""
    _call("POST", f"/memories/{urllib.parse.quote(memory_id, safe='')}/sources", model=model,
          body={"sources": body_sources}, headers={"idempotency-key": idempotency_key}, timeout=120)


def remove_source(memory_id: str, model: str | None, source_id: str) -> None:
    """Take one source out of a collection; one already gone is fine."""
    _call("DELETE", f"/memories/{urllib.parse.quote(memory_id, safe='')}/sources/{urllib.parse.quote(source_id, safe='')}",
          model=model, allow=(404,))


def rename(memory_id: str, model: str | None, name: str) -> None:
    _call("PATCH", f"/memories/{urllib.parse.quote(memory_id, safe='')}", model=model, body={"name": name[:160]})


def remove(memory_id: str, model: str | None) -> bool:
    """Remove a collection from the library; one already gone counts as removed. False when this connection
    may not remove knowledge."""
    try:
        _call("DELETE", f"/memories/{urllib.parse.quote(memory_id, safe='')}", model=model, allow=(404,))
    except KnowledgeError as error:
        if error.status in (401, 403) or error.missing_route:
            return False
        raise
    return True


def readiness(memory_id: str, model: str | None, timeout: float = 15) -> str:
    """Where a collection stands: ready, preparing, failed or not-ready; failed once it is gone."""
    body = _call("GET", "/memories/" + urllib.parse.quote(memory_id, safe=""), model=model, allow=(404,),
                 timeout=timeout)
    if body is None:
        return "failed"
    memory = body.get("memory") if isinstance(body.get("memory"), dict) else body
    return status_of(memory)[0]


def console_link(session_id: str | None = None) -> str:
    """AdaptiqAI Knowledge in the browser, signed in first, on this conversation when given (core's consoleLink)."""
    destination = "/memory?from=hermes" + (f"#/sessions/{urllib.parse.quote(session_id, safe='')}" if session_id else "")
    return CONSOLE.rsplit("/memory", 1)[0] + "/auth/login?next=" + urllib.parse.quote(destination, safe="")
