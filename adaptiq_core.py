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
import urllib.parse
import urllib.request

from . import wire

# ADAPTIQ_API_BASE points a test at a local gateway; customers never set it.
API = os.environ.get("ADAPTIQ_API_BASE", "https://api.adaptiqai.com/v1").rstrip("/")
KEY_ENV = "ADAPTIQ_API_KEY"
# The entry's key, as the console's setup command and Hermes Desktop's own Save name it.
PROVIDER = "AdaptiqAI"
APP = "Hermes"


def _call(method: str, path: str, body: dict | None = None, key: str | None = None,
          headers: dict | None = None, timeout: float = 15) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json", **(headers or {})}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(API + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        # The gateway explains a refusal in the body (`detail.code`, `detail.message`).
        try:
            reply = json.loads(error.read() or b"{}")
        except ValueError:
            reply = {}
        return error.code, reply if isinstance(reply, dict) else {}


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


def _model_rows(key: str) -> list[dict] | None:
    """The models the workspace has lanes on, or None when AdaptiqAI did not answer with a list."""
    status, listing = _call("GET", "/models", key=key, timeout=5)
    return [row for row in (listing.get("data") or []) if row.get("id")] if status == 200 else None


def _lanes(row: dict) -> int:
    return int((row.get("adaptiq") or {}).get("lanes") or 0)


def _main_model(rows: list[dict]) -> str:
    """The model with the most lanes, or "" when the workspace has lanes on none.

    `/v1/models` lists the engine's own model for every workspace, lanes or not; a main model
    the workspace cannot call would fail every turn.
    """
    held = [row for row in rows if _lanes(row) > 0]
    return max(held, key=_lanes)["id"] if held else ""


def _entry_name(providers: dict, main: str | None = None) -> str | None:
    """The provider entry on AdaptiqAI's URL, whatever it is called.

    Where several are (a hand-made one beside the console's `AdaptiqAI`), the one Hermes' main model
    uses, `main`, else the first.
    """
    names = [key for key, row in providers.items()
             if isinstance(row, dict) and str(row.get("base_url") or "").rstrip("/") == API]
    return main if main in names else (names[0] if names else None)


def _main_provider(cfg: dict) -> str | None:
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return model.get("provider")


def _in_step(entry: dict, rows: list[dict]) -> dict:
    """The entry with exactly the workspace's models, their context windows, and the run-once mark.

    Settings someone added to a model that still has lanes are kept.
    """
    entry = dict(entry)
    known = entry.get("models") if isinstance(entry.get("models"), dict) else {}
    models = {}
    for row in rows:
        settings = dict(known.get(row["id"]) or {})
        if row.get("context_window"):
            settings["context_length"] = row["context_window"]
        models[row["id"]] = settings
    entry["models"] = models
    if rows and entry.get("model") not in models and _main_model(rows):
        entry["model"] = _main_model(rows)
    # Hermes adds this to every request to the entry, helper calls (titles, compression, approvals)
    # included. The plugin's request middleware takes it off a turn of a conversation again.
    extra_body = entry.get("extra_body") if isinstance(entry.get("extra_body"), dict) else {}
    entry["extra_body"] = {**extra_body, wire.MEMORY_FIELD: wire.ephemeral("auxiliary")}
    return entry


def apply(token: str) -> tuple[str, list[str]]:
    """Store the key where Hermes keeps keys and point Hermes at the workspace's models.

    hermes_cli.config is Hermes' own writer (the one its custom-endpoint settings use). It is not
    a documented plugin API, so a Hermes refactor can move it.
    """
    from hermes_cli.config import load_config, save_config, save_env_value

    rows = _model_rows(token) or []
    save_env_value(KEY_ENV, token)
    default = _main_model(rows)

    cfg = load_config()
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    # An endpoint someone added by hand for the same URL is taken over, not
    # shown twice in the model picker.
    name = _entry_name(providers, _main_provider(cfg)) or PROVIDER
    entry = dict(providers.get(name) or {})
    entry.update({"name": "AdaptiqAI", "base_url": API, "key_env": KEY_ENV,
                  "api_mode": "chat_completions", "discover_models": True})
    # The key now comes from .env; a key pasted into the entry would win over it.
    entry.pop("api_key", None)
    entry = _in_step(entry, rows)
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
    return default, list(entry["models"])


def sync() -> dict:
    """Keep the provider entry in step with the workspace's lanes, and Hermes' main model on one.

    A model rented after sign-in gets its context window, and a model whose lanes ended leaves the
    entry. When the main model was an AdaptiqAI model that has no lanes any more, it moves to the
    model with the most lanes. Writes nothing when AdaptiqAI does not answer or nothing changed.
    """
    from hermes_cli.config import get_env_value, load_config, save_config

    key = get_env_value(KEY_ENV)
    rows = _model_rows(key) if key else None
    # No list, or an empty one, is not a workspace with no models: nothing is taken away on it.
    if not rows:
        return {"synced": False}
    cfg = load_config()
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    name = _entry_name(providers, _main_provider(cfg))
    if name is None:
        return {"synced": False}
    entry = _in_step(providers[name], rows)
    was = dict(cfg["model"]) if isinstance(cfg.get("model"), dict) else {}
    model = dict(was)
    moved_from = None
    # The main model moves only onto a model the workspace holds lanes on; with lanes on none,
    # it stays where it is rather than land on one that fails every turn.
    if model.get("provider") == name and model.get("default") not in entry["models"] and _main_model(rows):
        moved_from = model.get("default") or None
        model["default"] = _main_model(rows)
    if entry != providers[name] or model != was:
        providers[name] = entry
        cfg["providers"] = providers
        if model != was:
            cfg["model"] = model
        save_config(cfg)
    return {"synced": True, "models": list(entry["models"]), "moved_from": moved_from,
            "main": model.get("default") if model.get("provider") == name else None}


def start_on(model_id: str) -> bool:
    """Make a model that just got lanes Hermes' main model when the main model is an AdaptiqAI model
    without lanes, or there is none. A main model on another provider is the person's choice, and
    stays: /model switches. Returns True when it changed."""
    from hermes_cli.config import get_env_value, load_config, save_config

    key = get_env_value(KEY_ENV)
    rows = (_model_rows(key) if key else None) or []
    if not any(row["id"] == model_id and _lanes(row) > 0 for row in rows):
        return False
    cfg = load_config()
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    name = _entry_name(providers, _main_provider(cfg))
    if name is None:
        return False
    model = dict(cfg["model"]) if isinstance(cfg.get("model"), dict) else {}
    if model.get("default") and model.get("provider") != name:
        return False
    if model.get("provider") == name and any(row["id"] == model.get("default") and _lanes(row) > 0 for row in rows):
        return False
    model.update({"provider": name, "default": model_id, "base_url": API, "api_mode": "chat_completions",
                  "key_env": KEY_ENV})
    model.pop("api_key", None)
    cfg["model"] = model
    save_config(cfg)
    return True


def main_model() -> str | None:
    """Hermes' main model when it is an AdaptiqAI model, else None."""
    from hermes_cli.config import load_config

    cfg = load_config()
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    name = _entry_name(providers, _main_provider(cfg))
    return model.get("default") if name is not None and model.get("provider") == name else None


def account_summary() -> dict | None:
    """`/v1/account/summary`: workspace, connection, models with lanes and memory; None without it."""
    from hermes_cli.config import get_env_value

    key = get_env_value(KEY_ENV)
    if not key:
        return None
    try:
        code, summary = _call("GET", "/account/summary", key=key, timeout=5)
    except (OSError, ValueError):
        return None
    return summary if code == 200 and summary.get("schema") == "adaptiq-account-summary-v1" else None


def mark_entry() -> bool:
    """Give the provider entry the run-once mark for helper calls where it lacks one.

    Entries written by 0.1.0 have none, and helper calls never reach the request middleware: a
    title or a compression summary of a private conversation went out with no memory mode and
    was saved. Returns True when the config was changed.
    """
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    changed = False
    for row in providers.values():
        if not isinstance(row, dict) or str(row.get("base_url") or "").rstrip("/") != API:
            continue
        extra_body = row.get("extra_body") if isinstance(row.get("extra_body"), dict) else {}
        if wire.MEMORY_FIELD in extra_body:
            continue
        row["extra_body"] = {**extra_body, wire.MEMORY_FIELD: wire.ephemeral("auxiliary")}
        changed = True
    if changed:
        cfg["providers"] = providers
        save_config(cfg)
    return changed


def session_stored(session_id: str, model: str | None = None) -> bool:
    """Whether AdaptiqAI stores this conversation. A private turn of one it stores is read back
    without saving; anything else runs ephemeral, so a failed lookup costs speed, never privacy."""
    from hermes_cli.config import get_env_value

    key = get_env_value(KEY_ENV)
    if not key:
        return False
    try:
        code, _ = _call("GET", "/sessions/" + urllib.parse.quote(session_id, safe=""), key=key,
                        headers={"x-adaptiq-model": model} if model else None, timeout=3)
    except (OSError, ValueError):  # unreachable, slow or not JSON: ephemeral is always safe
        return False
    return code == 200


def status() -> dict:
    from hermes_cli.config import get_env_value

    key = get_env_value(KEY_ENV)
    if not key:
        return {"signed_in": False}
    code, who = _call("GET", "/cli-auth/whoami", key=key, timeout=5)
    return {"signed_in": code == 200,
            "workspace": (who.get("workspace") or {}).get("label", ""),
            "expires_at": (who.get("connection") or {}).get("expires_at")}
