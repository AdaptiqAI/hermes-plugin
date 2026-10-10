"""AdaptiqAI sign-in for Hermes, shared by the slash command, the CLI and the Desktop routes.

Device flow (docs/HARNESS_PLUGINS.md in the gateway repo) -> key in Hermes' .env under
ADAPTIQ_API_KEY -> a ``providers.adaptiq`` entry pointing at it through ``key_env`` -> the main
model on the workspace model with the most lanes. Stdlib only, so the plugin has no dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import time
import urllib.error
import urllib.request

# ADAPTIQ_API_BASE points a test at a local gateway; customers never set it.
API = os.environ.get("ADAPTIQ_API_BASE", "https://api.adaptiqai.com/v1").rstrip("/")
KEY_ENV = "ADAPTIQ_API_KEY"
PROVIDER = "adaptiq"
APP = "Hermes"


def _call(method: str, path: str, body: dict | None = None, key: str | None = None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(API + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, {}


def device_name() -> str:
    return (platform.node().split(".")[0] or "hermes")[:80]


def start() -> dict:
    """Begin a sign-in: device_code, user_code, verification_uri_complete, interval, expires_in."""
    proof = secrets.token_hex(32)
    status, started = _call("POST", "/cli-auth/device", {
        "client_app": APP, "client_name": APP, "device_name": device_name(),
        "client_binding": hashlib.sha256(proof.encode()).hexdigest(), "token_lifetime": "long",
    })
    if status not in (200, 201):
        raise RuntimeError(f"AdaptiqAI could not start the sign-in (HTTP {status}).")
    return started


def poll_once(device_code: str) -> tuple[int, dict]:
    """202 pending, 200 {access_token, ...}, 403 declined, 410 expired, 429 back off."""
    return _call("POST", "/cli-auth/device/token",
                 {"device_code": device_code, "client_app": APP, "device_name": device_name()})


def wait(started: dict) -> str:
    """Poll until approved; times out with the code so an unattended run never hangs."""
    deadline = time.time() + int(started.get("expires_in") or 600)
    while time.time() < deadline:
        status, reply = poll_once(started["device_code"])
        if status == 200:
            return reply["access_token"]
        if status == 403:
            raise RuntimeError("The connection was declined in the browser.")
        if status == 410:
            raise RuntimeError("The sign-in expired. Run it again.")
        time.sleep(5 if status == 429 else int(started.get("interval") or 2))
    raise RuntimeError("The sign-in timed out.")


def apply(token: str) -> tuple[str, list[str]]:
    """Store the key where Hermes keeps keys and point Hermes at the workspace's models.

    hermes_cli.config is Hermes' own writer (the one its custom-endpoint settings use). It is not
    a documented plugin API, so a Hermes refactor can move it.
    """
    from hermes_cli.config import load_config, save_config, save_env_value

    status, listing = _call("GET", "/models", key=token)
    rows = [row for row in (listing.get("data") or []) if row.get("id")] if status == 200 else []
    save_env_value(KEY_ENV, token)

    models = {row["id"]: ({"context_length": row["context_window"]} if row.get("context_window") else {})
              for row in rows}
    lanes = lambda row: int((row.get("adaptiq") or {}).get("lanes") or 0)  # noqa: E731
    default = max(rows, key=lanes)["id"] if rows else ""

    cfg = load_config()
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    # An endpoint someone added by hand for the same URL is taken over, not
    # shown twice in the model picker.
    name = next((key for key, row in providers.items() if isinstance(row, dict)
                 and str(row.get("base_url") or "").rstrip("/") == API), PROVIDER)
    entry = dict(providers.get(name) or {})
    entry.update({"name": "AdaptiqAI", "base_url": API, "key_env": KEY_ENV,
                  "api_mode": "chat_completions", "discover_models": True, "models": models})
    # The key now comes from .env; a key pasted into the entry would win over it.
    entry.pop("api_key", None)
    if default:
        entry["model"] = default
    providers[name] = entry
    cfg["providers"] = providers
    if default:
        model = dict(cfg["model"]) if isinstance(cfg.get("model"), dict) else {}
        model.update({"provider": name, "default": default, "base_url": API,
                      "api_mode": "chat_completions", "key_env": KEY_ENV})
        model.pop("api_key", None)
        cfg["model"] = model
    save_config(cfg)
    return default, list(models)


def status() -> dict:
    from hermes_cli.config import get_env_value

    key = get_env_value(KEY_ENV)
    if not key:
        return {"signed_in": False}
    code, who = _call("GET", "/cli-auth/whoami", key=key)
    return {"signed_in": code == 200,
            "workspace": (who.get("workspace") or {}).get("label", ""),
            "expires_at": (who.get("connection") or {}).get("expires_at")}
