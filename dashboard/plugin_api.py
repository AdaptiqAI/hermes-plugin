"""Desktop sign-in routes, mounted at /api/plugins/adaptiq/ inside `hermes serve`.

Each request runs under the requesting profile's scope, so the Desktop half drives the polling
(one short request per interval) instead of a background thread here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi import APIRouter, HTTPException

# This file is imported on its own (no package), so load the shared module by path.
_spec = importlib.util.spec_from_file_location(
    "adaptiq_core_for_desktop", Path(__file__).resolve().parent.parent / "adaptiq_core.py")
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)

router = APIRouter()


@router.get("/status")
def status() -> dict:
    return core.status()


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
