"""Desktop sign-in routes, mounted at /api/plugins/adaptiq/ inside `hermes serve`.

Each request runs under the requesting profile's scope, so the Desktop half drives the polling
(one short request per interval) instead of a background thread here.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

# This file is imported on its own (no package), so load the plugin folder as a package, the way
# Hermes does, for the shared modules' relative imports.
_PLUGIN = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "adaptiq_plugin_for_desktop", _PLUGIN / "__init__.py", submodule_search_locations=[str(_PLUGIN)])
_plugin = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _plugin
try:
    _spec.loader.exec_module(_plugin)
except Exception:
    # A half-made module must not stay registered for the next import to find.
    sys.modules.pop(_spec.name, None)
    raise
core = _plugin.core

router = APIRouter()


def _state():
    """The plugin's own state in the requesting profile: the file the agent half keeps choices in."""
    from hermes_cli.plugins_state import PluginState

    return PluginState("adaptiq")


@router.get("/status")
def status() -> dict:
    return core.status()


@router.get("/account")
def account() -> dict:
    """The settings page: workspace, memory and models, or only whether it is connected; with the notices the
    page raises (the memory warning, sleeping conversations, a newer plugin)."""
    state = _state()
    _plugin._beside("adaptiq-update", _plugin._check_for_update, state)
    update = _plugin._update_notice(state)
    summary = core.account_summary()
    if summary is None:
        return {"signed_in": core.status()["signed_in"], "account": None, "console": _plugin.knowledge.console_link(),
                "update": update}
    connection = summary.get("connection") if isinstance(summary.get("connection"), dict) else {}
    view = _plugin.account.view(summary, main=core.main_model(), api_key=connection.get("kind") == "api_key",
                                sleep=core.session_memory())
    return {"signed_in": True, "account": view, "console": _plugin.knowledge.console_link(), "update": update,
            "action": _action(state, summary)}


ACTION_FOR = 3600
# DeepSeek Harness's notice for each refusal the person can end (core's strings), and its button.
ACTIONS = {
    "sign_in": ("Sign in to keep going", "Your AdaptiqAI connection needs a fresh sign-in.", "Sign in again"),
    "funds": ("Add funds to continue", "Your balance ran out. Add funds to keep going.", "Add funds"),
    "access": ("Get access to continue", "{model} isn't in your workspace yet. You pay per token from your balance; "
               "reads from memory are free.", "Get access"),
}


def _action(state, summary: dict) -> dict | None:
    """The last refusal of the past hour the person can end, with its notice and button, until a request goes
    through: sign in again (the page signs in), add funds or get access (the page opens the console there)."""
    kept = state.get(_plugin.ACTION)
    if not isinstance(kept, dict) or kept.get("kind") not in ACTIONS:
        return None
    at = kept.get("at")
    if not isinstance(at, int) or time.time() - at > ACTION_FOR:
        return None
    title, text, label = ACTIONS[kept["kind"]]
    links = summary.get("links") if isinstance(summary.get("links"), dict) else {}
    console = links.get("console") or _plugin.errors.CONSOLE
    url = {"funds": links.get("billing") or f"{console}#/billing", "access": f"{console}#/models"}.get(kept["kind"])
    model = _plugin.errors.model_name(kept.get("model") or "") if kept.get("model") else "This model"
    return {"kind": kept["kind"], "at": at, "title": title, "text": text.format(model=model), "label": label,
            "url": url}


@router.post("/logout")
def logout() -> dict:
    """Forget this computer's sign-in; an API key Connect or the person configured stays."""
    core.sign_out()
    return {"signed_in": core.status()["signed_in"]}


def _session(value) -> str:
    """A Hermes session id, or a 400: anything else would only fill the plugin's state."""
    session = value if isinstance(value, str) else ""
    if not _plugin.SESSION_ID.fullmatch(session):
        raise HTTPException(status_code=400, detail="Name the conversation by its Hermes session id.")
    return session


@router.get("/conversation")
def conversation(session: str, model: str = "") -> dict:
    """One Hermes conversation for the status chip: whether its new messages are saved to AdaptiqAI memory, how much
    of it memory holds, and its model's window (DSH's memory pill)."""
    session = _session(session)
    held, window = None, None
    if model:
        try:
            held = _plugin.knowledge.conversation_view(_plugin._conversation(session), model)["memory_tokens"]
        except _plugin.knowledge.KnowledgeError:
            held = None
        window = _window(model)
    state = _state()
    counted = state.get(_plugin.PROCESSED, {})
    rates = _plugin.pricing.prices(_row(model)) if model else None
    return {"session": session, "saving": _plugin._remember(state, session), "memory_tokens": held,
            "window": window,
            "processed": counted.get(_plugin._conversation(session)) if isinstance(counted, dict) else None,
            "memory_read_free": rates is not None and rates["input_cache_read"] == 0}


def _row(model: str) -> dict | None:
    """A model's /v1/models row, read once here when this copy has not read it yet."""
    if core.model_row(model) is None:
        try:
            core._model_rows()
        except (OSError, ValueError):
            return None
    return core.model_row(model)


def _window(model: str) -> int | None:
    """A model's context window: from its /v1/models row as last read here, else from the provider entry the plugin
    keeps in step with the workspace."""
    row = core.model_row(model) or {}
    if isinstance(row.get("context_window"), int):
        return row["context_window"]
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return None
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    entry = providers.get(core._entry_name(providers, core._main_provider(cfg)) or "") or {}
    settings = (entry.get("models") or {}).get(model) if isinstance(entry.get("models"), dict) else None
    length = settings.get("context_length") if isinstance(settings, dict) else None
    return length if isinstance(length, int) else None


@router.post("/conversation")
def set_conversation(body: dict) -> dict:
    session = _session(body.get("session"))
    # "false" is not false: only a real boolean changes the choice.
    if not isinstance(body.get("saving"), bool):
        raise HTTPException(status_code=400, detail="Say saving as true or false.")
    _plugin._set_remember(_state(), session, body["saving"])
    return {"session": session, "saving": _plugin._remember(_state(), session)}


@router.get("/knowledge")
def knowledge(session: str, model: str) -> dict:
    """The memory pill's knowledge: the collections this conversation uses, and the ready ones for its model it
    could use in one click (DSH's knowledge panel)."""
    session = _session(session)
    conversation = _plugin._conversation(session)
    try:
        view = _plugin.knowledge.conversation_view(conversation, model)
        used = [{"memory_id": item["memory_id"], "name": item["name"], "status": item["status"]}
                for item in view["knowledge"]]
        page = _plugin.knowledge.library(model, None, None, frozenset(item["memory_id"] for item in used))
    except _plugin.knowledge.KnowledgeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    others = [{"memory_id": item["memory_id"], "name": item["name"]} for item in page["items"]
              if item["kind"] == "collection" and item["fits"] and item["readiness"] == "ready" and not item["selected"]]
    return {"session": session, "used": used, "others": others, "more": page["has_more"],
            "console": _plugin.knowledge.console_link(conversation)}


@router.post("/knowledge")
def use_knowledge(body: dict) -> dict:
    """Use a collection in this conversation, or stop using it; nothing is deleted."""
    session = _session(body.get("session"))
    model, memory_id = body.get("model"), body.get("memory_id")
    if not isinstance(model, str) or not model or not isinstance(memory_id, str) or not memory_id:
        raise HTTPException(status_code=400, detail="Name the model and the collection.")
    if not isinstance(body.get("use"), bool):
        raise HTTPException(status_code=400, detail="Say use as true or false.")
    try:
        _plugin.knowledge.use(_plugin._conversation(session), model, memory_id, body["use"])
    except _plugin.knowledge.KnowledgeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return knowledge(session, model)


@router.post("/login/start")
def login_start() -> dict:
    try:
        started = core.start()
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {k: started.get(k) for k in
            ("device_code", "user_code", "verification_uri_complete", "interval", "expires_in")}


@router.post("/login/poll")
def login_poll(body: dict) -> dict:
    code, reply = core.poll_once(str(body.get("device_code") or ""))
    if code in (202, 429):
        return {"status": "pending"}
    if code == 200:
        model, models = core.apply(reply["access_token"])
        return {"status": "done", "model": model, "models": models}
    return {"status": {403: "declined", 410: "expired"}.get(code, f"failed (HTTP {code})")}
