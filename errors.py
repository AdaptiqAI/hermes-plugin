"""AdaptiqAI's failures in Hermes' terms (core's errors.ts, for Hermes' error classifier).

Hermes reads several of AdaptiqAI's answers wrongly. A workspace without lanes on a model is told
to fix its API key, a full memory is retried in vain, and a conversation being prepared is given
up as a malformed request. Each answer here gets the recovery Hermes should run and one sentence
the person can act on. A full memory is never called a context overflow: that would make Hermes
compress a conversation AdaptiqAI keeps whole. Anything not listed is left to Hermes.
"""

from __future__ import annotations

import re

CONSOLE = "https://memory.adaptiqai.com/memory"

# A server with one shared pool (no session history reported apart): nothing sleeps there.
MEMORY_FULL = ("Your workspace memory is full, so this conversation can't save new messages. "
               "Free up space in AdaptiqAI Knowledge, or continue privately with /adaptiq private.")
# Where idle conversations sleep on their own to make room, a 507 means none could. A new conversation
# frees nothing, so it is never suggested.
MEMORY_NO_ROOM = ("Memory is full and nothing idle could sleep. Delete or unpin a conversation in the console, "
                  "or add memory.")
MEMORY_FULL_LIBRARY = ("Your library is full. Remove a collection you no longer need in the console, "
                       "or add memory.")
SIGN_IN = "Your AdaptiqAI sign-in has expired. Run /adaptiq login to sign in again."
NO_LANES = ("Your workspace has no lanes on {name} right now. Get lanes with /adaptiq models, or pick "
            "another model with /model.")
# What older gateways said for every model without lanes. Newer ones name the model and where to
# rent it, and that sentence is passed on.
_UNNAMED = ("model access is required", "select an authorized model", "instance access is required")
GET_HERE = "Or get lanes here with /adaptiq models."
BUSY = "All of your lanes on {name} are busy. Your request will retry automatically in a moment."
# Types and codes only AdaptiqAI sends: an error that carries one is AdaptiqAI's even with no
# status, as one raised inside a stream is.
_OWN = ("adaptiq", "memory_capacity_exceeded", "inference_entitlement_required", "instance_access_required",
        "lane_access", "rental_stopped_unused", "inflightlimit", "session_route_unavailable",
        "conversation_preparing", "upload_capacity_exhausted")
PREPARING = "AdaptiqAI is getting your conversation ready. It will retry automatically."
CONVERSATION_BUSY = ("This conversation couldn't be opened on AdaptiqAI just now. It will retry "
                     "automatically; your saved memory is unchanged.")

_WORDS = {"deepseek": "DeepSeek", "qwen": "Qwen", "llama": "Llama", "mistral": "Mistral", "gpt": "GPT",
          "glm": "GLM", "kimi": "Kimi", "flash": "Flash", "pro": "Pro", "mtp": "MTP", "int8": "INT8", "fp8": "FP8"}


def model_name(model_id: str) -> str:
    """A readable name from a model id: `adaptiq-deepseek-v4.1-flash` → `DeepSeek V4.1 Flash`."""
    words = [word for word in re.split(r"[-_\s]+", re.sub(r"^adaptiq[-_]", "", model_id or "", flags=re.I)) if word]
    named = []
    for word in words:
        lower = word.lower()
        prefix = next((key for key in _WORDS if lower.startswith(key) and re.match(r"[\d.]", lower[len(key):])), None)
        if lower in _WORDS:
            named.append(_WORDS[lower])
        elif prefix is not None:
            named.append(_WORDS[prefix] + lower[len(prefix):].upper())
        elif re.match(r"v\d", lower):
            named.append("V" + word[1:])
        elif re.fullmatch(r"\d+(\.\d+)?[bm]", word, flags=re.I):
            named.append(word.upper())
        else:
            named.append(word[:1].upper() + word[1:])
    return " ".join(named) or model_id


def server_detail(body) -> tuple[str, str, str]:
    """(message, type, code) from any of the envelopes the gateway and engines send:
    `{error: {message, type, code}}`, `{detail: {code, message}}`, `{detail: "…"}`, `{error: "…"}`."""
    def text(value) -> str:
        return value.strip() if isinstance(value, str) else ""

    if not isinstance(body, dict):
        return text(body), "", ""
    inner = body.get("error", body.get("detail"))
    if isinstance(inner, str):
        return text(inner), text(body.get("type")), text(body.get("code"))
    if isinstance(inner, dict):
        return text(inner.get("message")), text(inner.get("type")), text(inner.get("code"))
    return text(body.get("message")), text(body.get("type")), text(body.get("code"))


def is_memory_full(status: int | None, body) -> bool:
    message, kind, code = server_detail(body)
    return status == 507 or "memory_capacity_exceeded" in f"{kind} {code} {message}".lower()


def classify(status: int | None, body, model: str, auto_sleep: bool = False) -> dict | None:
    """The verdict for Hermes' `transform_api_error_classification` hook, or None to leave it to Hermes.

    `auto_sleep`: the server puts idle conversations to sleep (its usage reports session history apart).
    """
    message, kind, code = server_detail(body)
    haystack = f"{kind} {code} {message}".lower()

    def has(*needles: str) -> bool:
        return any(needle.lower() in haystack for needle in needles)

    # No status, as inside a stream: only a type or code of AdaptiqAI's own says anything.
    if status is None and not any(own in f"{kind} {code}".lower() for own in _OWN):
        return None

    name = model_name(model) if model else "this model"
    if status == 507 or has("memory_capacity_exceeded"):
        # Another AdaptiqAI model keeps its own memory, so /model is a way on as well.
        if not auto_sleep:
            full = MEMORY_FULL
        else:
            full = MEMORY_FULL_LIBRARY if "library is full" in message.lower() else MEMORY_NO_ROOM
        return {"reason": "unknown", "retryable": False, "should_compress": False,
                "should_rotate_credential": False, "should_fallback": False, "message": full}
    if status == 401:
        # Hermes' own recovery for a refused key, with the way to a new one.
        return {"reason": "auth", "retryable": False, "should_rotate_credential": True, "should_fallback": True,
                "message": SIGN_IN}
    if code == "rental_stopped_unused":
        # The gateway says why and where to rent again.
        return {"reason": "model_not_found", "retryable": False, "should_fallback": True,
                "message": f"{message} {GET_HERE}" if message else NO_LANES.format(name=name)}
    if (has("AdaptiqLaneAccessRequired", "inference_entitlement_required", "instance_access_required", "lane_access")
            or (status == 403 and has("lane", "entitle", "model access"))):
        # model_not_found, not model_entitlement: that one benches the key for the model, and a
        # rental started a minute later must work at once.
        said = message and not message.lower().startswith(_UNNAMED)
        return {"reason": "model_not_found", "retryable": False, "should_fallback": True,
                "message": f"{message} {GET_HERE}" if said else NO_LANES.format(name=name)}
    if status == 429 or has("AdaptiqLaneQueueTimeout", "InflightLimit"):
        # overloaded, not rate_limit: Hermes acts on the reason, and rate_limit benches the one key
        # in its credential pool after a retry (the flag that would stop it is not read). overloaded
        # is retried on the same key and then falls back, as Hermes itself does for a busy engine.
        return {"reason": "overloaded", "should_fallback": True, "message": BUSY.format(name=name)}
    if (has("AdaptiqNativePreflightBusy", "session_route_unavailable", "conversation_preparing",
            "upload_capacity_exhausted") or (status == 503 and not has("offline"))):
        return {"reason": "overloaded", "message": PREPARING}
    if status == 409 and has("AdaptiqNativePreflight", "conversation"):
        return {"reason": "overloaded", "message": CONVERSATION_BUSY}
    return None
