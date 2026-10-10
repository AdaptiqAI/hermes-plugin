"""A conversation's knowledge on AdaptiqAI: what it reads, the library, and using a collection
(core's workspace.ts, its knowledge and conversation parts).

A collection is built for one model and works only with that model, so the library is read from the
store of the model the conversation runs on.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
from datetime import datetime, timezone

from . import adaptiq_core as core
from .errors import CONSOLE

PAGE = 50
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
    """One sentence for the agent or the person."""


def _key() -> str:
    from hermes_cli.config import get_env_value

    key = get_env_value(core.KEY_ENV)
    if not key:
        raise KnowledgeError("AdaptiqAI is not connected. Ask the user to run /adaptiq login.")
    return key


def _raw(method: str, path: str, model: str | None = None, body: dict | None = None,
         headers: dict | None = None, timeout: float = 15) -> tuple[int, dict]:
    headers = {**({"x-adaptiq-model": model} if model else {}), **(headers or {})}
    try:
        return core._call(method, path, body=body, key=_key(), headers=headers or None, timeout=timeout)
    except (OSError, ValueError) as error:
        raise KnowledgeError("AdaptiqAI can't be reached right now. Try again in a moment.") from error


def _call(method: str, path: str, model: str | None = None, body: dict | None = None,
          allow: tuple[int, ...] = (), headers: dict | None = None, timeout: float = 15) -> dict | None:
    """One knowledge call: the body, None for an allowed status, else a KnowledgeError."""
    status, reply = _raw(method, path, model, body, headers, timeout)
    if status in allow:
        return None
    if status >= 400:
        raise KnowledgeError(_failure(status, reply))
    return reply


def _failure(status: int, reply) -> str:
    detail = reply.get("detail") if isinstance(reply, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None
    message = detail.get("message") if isinstance(detail, dict) else detail if isinstance(detail, str) else None
    if code == "model_mismatch" and message:
        return message  # the gateway names both models
    if code == "instance_unavailable" or status in (502, 503, 504):
        return "This model is offline right now, so its knowledge can't be read. Try again when it is back."
    if status == 401:
        return "The AdaptiqAI sign-in has expired. Ask the user to run /adaptiq login."
    if status == 403:
        return "This AdaptiqAI connection may not do that. Ask the user to run /adaptiq login."
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


def collection(memory: dict, selected: bool) -> dict:
    """One collection as the tools report it (the DeepSeek Harness plugin's fields)."""
    origin = memory.get("origin") if isinstance(memory.get("origin"), dict) else {}
    readiness, label = status_of(memory)
    item = {
        "memory_id": str(memory.get("memory_id") or memory.get("id") or ""),
        "name": memory.get("name") or origin.get("name") or "Untitled",
        "kind": "conversation" if origin.get("type") == "conversation" else "collection",
        "readiness": readiness,
        "status": label,
        "selected": selected,
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


def conversation(session_id: str, model: str | None = None) -> dict | None:
    """What AdaptiqAI keeps for a conversation, or None while it is not stored.

    The gateway sends a session call to the model the conversation is placed on. Where several models
    keep memory and it cannot tell (409), the conversation's own model is asked.
    """
    status, reply = _raw("GET", _path(session_id))
    if status == 409 and model:
        status, reply = _raw("GET", _path(session_id), model)
        if status == 409 and _not_on(reply, model):
            return None
    if status == 404:
        return None
    if status >= 400:
        raise KnowledgeError(_failure(status, reply))
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
    items = [collection(memory, str(memory.get("memory_id")) in selected)
             for memory in body.get("data") or [] if isinstance(memory, dict)]
    return {"items": items, "next_cursor": pagination.get("next_cursor"), "has_more": pagination.get("has_more") is True}


def _revision(session_id: str, model: str | None) -> int:
    """The conversation's revision, storing the conversation first when AdaptiqAI has none yet."""
    refused = None
    for _ in range(2):
        body = conversation(session_id, model)
        revision = ((body or {}).get("session") or {}).get("revision")
        if isinstance(revision, int):
            return revision
        if body is None:
            status, reply = _raw("POST", "/sessions", model, {"session_id": session_id, "name": "Hermes conversation"})
            # 409: it exists after all (read again), or a refusal to say if the read still finds nothing.
            if status == 409:
                refused = reply
            elif status >= 400:
                raise KnowledgeError(_failure(status, reply))
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
        status, reply = _raw("PUT" if attach else "DELETE", path, model, body)
        if status < 400:
            return
        detail = reply.get("detail") if isinstance(reply.get("detail"), dict) else {}
        # A revision that moved in between is read again once; a collection on another model is not.
        if status == 409 and detail.get("code") != "model_mismatch" and attempt == 0:
            continue
        raise KnowledgeError(_failure(status, reply))


def media_type(name: str) -> str:
    """The media type for a file name; source code is plain text."""
    extension = os.path.splitext(name)[1].lower()
    return MEDIA.get(extension) or ("text/plain" if extension in SOURCE_CODE else "application/octet-stream")


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
                content = base64.b64encode(handle.read(FILES_LIMIT + 1)).decode()
            out.append({"type": "file", "name": str(item.get("name") or os.path.basename(path))[:200],
                        "media_type": media_type(path), "content_base64": content})
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


def create(name: str, description: str, body_sources: list[dict], *, model: str | None,
           idempotency_key: str) -> dict:
    """Create a collection in the store of `model`; the same key makes a retried request create it once."""
    access = _call("GET", "/cli-auth/knowledge/status", allow=(404,))
    if access is not None and access.get("create_allowed") is False:
        raise KnowledgeError("This connection can't create knowledge in the workspace. Ask an owner, or create it "
                             "in the console.")
    body = {"name": name[:160], "description": description[:2000], "sources": body_sources}
    if len(json.dumps(body_sources)) > BODY_LIMIT:
        raise KnowledgeError("These files are too large for one upload. Add fewer or smaller files.")
    reply = _call("POST", "/memories", model=model, body=body, headers={"idempotency-key": idempotency_key},
                  timeout=120) or {}
    memory = reply.get("memory") if isinstance(reply.get("memory"), dict) else None
    if memory is None or not memory.get("memory_id"):
        raise KnowledgeError("AdaptiqAI did not confirm the new collection. Check Knowledge before trying again.")
    readiness, label = status_of(memory)
    return {"memory_id": str(memory["memory_id"]), "name": memory.get("name") or name, "readiness": readiness,
            "status": label, "model": memory.get("model") or model}


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
