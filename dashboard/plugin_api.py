"""Desktop sign-in routes, mounted at /api/plugins/adaptiq/ inside `hermes serve`.

Each request runs under the requesting profile's scope, so the Desktop half drives the polling
(one short request per interval) instead of a background thread here.
"""

from __future__ import annotations

import importlib.util
import sys
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
    """The settings page: workspace, memory and models, or only whether it is connected."""
    summary = core.account_summary()
    if summary is None:
        return {"signed_in": core.status()["signed_in"], "account": None, "console": _plugin.knowledge.console_link()}
    return {"signed_in": True, "account": _plugin.account.view(summary, main=core.main_model()),
            "console": _plugin.knowledge.console_link()}


def _session(value) -> str:
    """A Hermes session id, or a 400: anything else would only fill the plugin's state."""
    session = value if isinstance(value, str) else ""
    if not _plugin.SESSION_ID.fullmatch(session):
        raise HTTPException(status_code=400, detail="Name the conversation by its Hermes session id.")
    return session


@router.get("/conversation")
def conversation(session: str) -> dict:
    """Whether new messages of one Hermes conversation are saved to AdaptiqAI memory."""
    session = _session(session)
    return {"session": session, "saving": _plugin._remember(_state(), session)}


@router.post("/conversation")
def set_conversation(body: dict) -> dict:
    session = _session(body.get("session"))
    # "false" is not false: only a real boolean changes the choice.
    if not isinstance(body.get("saving"), bool):
        raise HTTPException(status_code=400, detail="Say saving as true or false.")
    _plugin._set_remember(_state(), session, body["saving"])
    return {"session": session, "saving": _plugin._remember(_state(), session)}


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
