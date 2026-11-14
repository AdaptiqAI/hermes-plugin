"""AdaptiqAI sign-in for Hermes, shared by the slash command, the CLI and the Desktop routes.

Device flow (docs/HARNESS_PLUGINS.md in the gateway repo) -> key in Hermes' .env under
ADAPTIQ_API_KEY (and, kept apart, ADAPTIQ_SIGNIN_KEY) -> a ``providers.AdaptiqAI`` entry pointing at it
through ``key_env`` -> the main model on the workspace model with the most lanes. Stdlib only, so the
plugin has no dependencies.

Hermes can hold several AdaptiqAI keys: the one this computer's sign-in made, the one Connect writes
into ADAPTIQ_API_KEY, and the one a long-running Hermes read from its environment when it started.
Every call tries them in turn, and a key the server refuses (401) gives way to the next (core's
apiKeys), so an old key never hides a working one.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import platform
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import account, wire

# ADAPTIQ_API_BASE points a test at a local gateway; customers never set it.
API = os.environ.get("ADAPTIQ_API_BASE", "https://api.adaptiqai.com/v1").rstrip("/")
KEY_ENV = "ADAPTIQ_API_KEY"
# This computer's own sign-in key, kept apart from ADAPTIQ_API_KEY, which Connect writes too: whichever
# wrote last, the other key is still there to give way to.
SIGNIN_ENV = "ADAPTIQ_SIGNIN_KEY"
# How long a key the server refused is skipped before it is tried again: a long-running Hermes
# must not stay signed out over one passing refusal.
REFUSED_FOR = 600
# The entry's key, as the console's setup command and Hermes Desktop's own Save name it.
PROVIDER = "AdaptiqAI"
APP = "Hermes"


def _call(method: str, path: str, body: dict | None = None, key: str | None = None,
          headers: dict | None = None, timeout: float = 15) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json", **wire.CLIENT, **(headers or {})}
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


_refused: dict[str, float] = {}
_refused_lock = threading.Lock()


def _env_values(name: str) -> list[str]:
    """A variable's values as Hermes resolves them: .env first, as Hermes' own provider auth reads it,
    then this process's environment, which a long-running gateway took when it started."""
    try:
        from hermes_cli import config
    except ImportError:
        return []
    values = []
    for read in (getattr(config, "get_env_value_prefer_dotenv", None), getattr(config, "get_env_value", None)):
        try:
            value = read(name) if read is not None else None
        except Exception:  # a scoped read that can't answer here
            value = None
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def _entry_keys() -> list[str]:
    """The keys of AdaptiqAI provider entries that keep theirs elsewhere: a Desktop-made custom endpoint's own
    variable, or a key pasted into the entry."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return []
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    rows = [row for row in [*providers.values(), model]
            if isinstance(row, dict) and str(row.get("base_url") or "").rstrip("/") == API]
    found = []
    for row in rows:
        pasted = row.get("api_key")
        if isinstance(pasted, str) and pasted.strip() and not pasted.strip().startswith(("$", "op://")):
            found.append(pasted.strip())
        env = row.get("key_env")
        if isinstance(env, str) and env and env not in (KEY_ENV, SIGNIN_ENV):
            found += _env_values(env)
    return found


def keys() -> list[str]:
    """Every AdaptiqAI key Hermes has, in the order to try them (core's apiKeys): ADAPTIQ_API_KEY as .env and
    this process hold it, this computer's own sign-in, then any other AdaptiqAI entry's key."""
    return list(dict.fromkeys([*_env_values(KEY_ENV), *_env_values(SIGNIN_ENV), *_entry_keys()]))


def _usable(value: str, now: float) -> bool:
    return now - _refused.get(value, -REFUSED_FOR) >= REFUSED_FOR


def key() -> str | None:
    """The first key the server has not refused lately, or None: not connected, or every key refused."""
    now = time.monotonic()
    with _refused_lock:
        return next((value for value in keys() if _usable(value, now)), None)


def phase() -> str:
    """`connected`, `expired` (keys, all refused) or `signed-out` (none), as core's AuthManager names them."""
    return "connected" if key() else "expired" if keys() else "signed-out"


def refuse(value: str) -> None:
    """The server refused this key (401): the next one is tried instead."""
    with _refused_lock:
        _refused[value] = time.monotonic()


def _heal(working: str) -> None:
    """Hermes' chats send ADAPTIQ_API_KEY. When the server refused what it holds and another key works, it
    holds the working key from now on, in .env and in this process."""
    configured = _env_values(KEY_ENV)
    if not configured or configured[0] == working:
        return
    # The first is what Hermes' provider auth sends: .env's value, else the environment's.
    with _refused_lock:
        dead = not _usable(configured[0], time.monotonic())
    if not dead:
        return
    try:
        from hermes_cli.config import save_env_value
        save_env_value(KEY_ENV, working)
    except Exception:  # Hermes' writer could not: the plugin's own calls still use the working key
        return


def authed(method: str, path: str, body: dict | None = None, headers: dict | None = None,
           timeout: float = 15) -> tuple[int, dict]:
    """A call with the first usable key; one the server refuses (401) gives way to the next. 401 with no key
    left. Network failures raise, as `_call`'s do."""
    status, reply, tried = 401, {}, set()
    while True:
        current = key()
        if current is None or current in tried:
            return status, reply
        tried.add(current)
        status, reply = _call(method, path, body, current, headers, timeout)
        if status != 401:
            _heal(current)
            return status, reply
        refuse(current)


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
        try:
            status, reply = poll_once(started["device_code"])
        except (OSError, ValueError, http.client.HTTPException):
            # No answer (a dropped connection, a timeout, a reply cut off): the approval in the
            # browser may still come, so wait and ask again, as the core does.
            status, reply = 0, {}
        if status == 200:
            return reply["access_token"]
        if status == 403:
            raise RuntimeError("The connection was declined in the browser.")
        if status == 410:
            raise RuntimeError("The sign-in expired. Run it again.")
        time.sleep(5 if status == 429 else int(started.get("interval") or 2))
    raise RuntimeError("The sign-in timed out.")


# Each model's `/v1/models` row as last read: its capability card (thinking levels, what it reads, its output cap)
# shapes requests without asking again.
_rows: dict[str, dict] = {}


def _model_rows() -> list[dict] | None:
    """The models the workspace has lanes on, or None when AdaptiqAI did not answer with a list."""
    status, listing = authed("GET", "/models", timeout=5)
    if status != 200:
        return None
    rows = [row for row in (listing.get("data") or []) if isinstance(row, dict) and row.get("id")]
    _rows.update({row["id"]: row for row in rows})
    return rows


def model_row(model_id: str) -> dict | None:
    """A model's `/v1/models` row as last read, or None before the first read."""
    return _rows.get(model_id)


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
        # What the model reads, from its card: Hermes sends images only to a model marked to see them, and
        # describes them for one that reads text only.
        card = row.get("adaptiq_capabilities")
        if isinstance(card, dict):
            settings["supports_vision"] = "image" in wire.card_input(card)
        models[row["id"]] = settings
    entry["models"] = models
    if rows and entry.get("model") not in models and _main_model(rows):
        entry["model"] = _main_model(rows)
    # Hermes adds this to every request to the entry, helper calls (titles, compression, approvals)
    # included. The plugin's request middleware takes it off a turn of a conversation again.
    extra_body = entry.get("extra_body") if isinstance(entry.get("extra_body"), dict) else {}
    entry["extra_body"] = {**extra_body, wire.MEMORY_FIELD: wire.ephemeral("auxiliary")}
    return entry


# Hermes compacts a conversation from a share of the model's window. AdaptiqAI keeps the whole
# conversation in native memory, as the DeepSeek Harness plugin declines compaction to keep it whole;
# Hermes has no per-provider way to decline, so its AdaptiqAI models compact only once the input
# budget is full (`compression.model_thresholds`), where the request would not fit anyway. Full is
# just under 1.0: Hermes reads a threshold at or above the window as 85% of it.
COMPACT_AT = 0.99


def _unreachable(value) -> bool:
    """A threshold Hermes reads as 85%: 1.0, which earlier versions wrote, or more."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 1


def _compact_when_full(cfg: dict, model_ids) -> bool:
    """Give each AdaptiqAI model the compact-when-full threshold, keeping one the person set unless
    Hermes reads it as 85%. Returns True when the config changed."""
    compression = dict(cfg["compression"]) if isinstance(cfg.get("compression"), dict) else {}
    thresholds = dict(compression["model_thresholds"]) if isinstance(compression.get("model_thresholds"), dict) else {}
    missing = [model for model in model_ids if model and (model not in thresholds or _unreachable(thresholds[model]))]
    if not missing:
        return False
    thresholds.update({model: COMPACT_AT for model in missing})
    compression["model_thresholds"] = thresholds
    cfg["compression"] = compression
    return True


def apply(token: str) -> tuple[str, list[str]]:
    """Store the key where Hermes keeps keys and point Hermes at the workspace's models.

    hermes_cli.config is Hermes' own writer (the one its custom-endpoint settings use). It is not
    a documented plugin API, so a Hermes refactor can move it.
    """
    from hermes_cli.config import load_config, save_config, save_env_value

    # The new key is this computer's own: kept apart too, so a key Connect writes later never loses it.
    save_env_value(KEY_ENV, token)
    save_env_value(SIGNIN_ENV, token)
    with _refused_lock:
        _refused.pop(token, None)
    rows = _model_rows() or []
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
    _compact_when_full(cfg, entry["models"])
    save_config(cfg)
    return default, list(entry["models"])


def sync() -> dict:
    """Keep the provider entry in step with the workspace's lanes, and Hermes' main model on one.

    A model rented after sign-in gets its context window, and a model whose lanes ended leaves the
    entry. When the main model was an AdaptiqAI model that has no lanes any more, it moves to the
    model with the most lanes. Writes nothing when AdaptiqAI does not answer or nothing changed.
    """
    from hermes_cli.config import load_config, save_config

    rows = _model_rows() if key() else None
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
    compaction = _compact_when_full(cfg, entry["models"])
    if entry != providers[name] or model != was or compaction:
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
    from hermes_cli.config import load_config, save_config

    rows = (_model_rows() if key() else None) or []
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
    if not key():
        return None
    try:
        code, summary = authed("GET", "/account/summary", timeout=5)
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
    if not key():
        return False
    try:
        code, _ = authed("GET", "/sessions/" + urllib.parse.quote(session_id, safe=""),
                         headers={"x-adaptiq-model": model} if model else None, timeout=3)
    except (OSError, ValueError):  # unreachable, slow or not JSON: ephemeral is always safe
        return False
    return code == 200


def session_memory() -> dict | None:
    """`GET /v1/workspace/session-memory`: which conversations sleep and any ask-mode suggestion.
    None without the `session_memory` scope, on an older gateway, or when it can't be reached."""
    if not key():
        return None
    try:
        code, view = authed("GET", "/workspace/session-memory", timeout=5)
    except Exception:  # unreachable, slow, cut short or not JSON: the status answers without it
        return None
    return view if code == 200 and isinstance(view, dict) else None


def sleeps(memory) -> bool:
    """Whether a server puts idle conversations to sleep: its memory reports session history apart."""
    return isinstance(memory, dict) and isinstance(memory.get("session_limit_tokens"), (int, float))


def auto_sleep(model: str) -> bool:
    """Whether the server behind `model` puts idle conversations to sleep, from the account summary.
    False when it can't be told: then a full memory keeps the shared-pool words."""
    try:
        summary = account_summary() or {}
    except Exception:
        return False
    own = next((row.get("memory") for row in summary.get("models") or []
                if isinstance(row, dict) and row.get("id") == model), None)
    return sleeps(own if isinstance(own, dict) else summary.get("memory"))


# Each model's memory as last read for a conversation still getting ready, and when (monotonic seconds).
_usage_read: dict[str, tuple[float, dict | None]] = {}
USAGE_EVERY = 15


def session_memory_full(model: str) -> dict | None:
    """A model's conversation memory when it is full (account.SESSION_FULL), for a conversation still getting ready:
    its `used` and `limit`, from `GET /v1/workspace/usage`, read again when the last reading is 15 seconds old (core's
    sessionMemoryFull). None when it has room, or can't be told."""
    at, known = _usage_read.get(model, (float("-inf"), None))
    now = time.monotonic()
    if now - at >= USAGE_EVERY:
        try:
            code, usage = authed("GET", "/workspace/usage", headers={"x-adaptiq-model": model}, timeout=5)
        except Exception:  # unreachable, slow, cut short or not JSON: the last reading stands
            code, usage = 0, None
        known = (account.memory_view(usage) if code == 200 else None) or known
        _usage_read[model] = (now, known)
    session = (known or {}).get("session")
    return session if session is not None and session["fraction"] >= account.SESSION_FULL else None


def status() -> dict:
    """Whether a key works, from `/v1/cli-auth/whoami`: the workspace, when it ends, and whether the key is this
    computer's sign-in (`device`) or an API key (`api_key`)."""
    if not key():
        return {"signed_in": False, "phase": phase()}
    code, who = authed("GET", "/cli-auth/whoami", timeout=5)
    connection = who.get("connection") if isinstance(who.get("connection"), dict) else {}
    return {"signed_in": code == 200, "phase": "connected" if code == 200 else phase(),
            "workspace": (who.get("workspace") or {}).get("label", ""),
            "expires_at": connection.get("expires_at"), "kind": connection.get("kind")}


def sign_out() -> bool:
    """Forget this computer's sign-in (core's signOut): its key leaves Hermes' .env, and ADAPTIQ_API_KEY too where
    it holds that key, or a device sign-in made before the key was kept apart. A key Connect or the person
    configured stays. Returns whether a usable key is left."""
    from hermes_cli import config

    own = set(_env_values(SIGNIN_ENV))
    configured = _env_values(KEY_ENV)
    if configured and (configured[0] in own or (not own and _device(configured[0]))):
        _forget(config, KEY_ENV)
    _forget(config, SIGNIN_ENV)
    with _refused_lock:
        _refused.clear()
    return key() is not None


def _device(value: str) -> bool:
    """Whether a key is a browser sign-in of Hermes, as the server knows it."""
    try:
        code, who = _call("GET", "/cli-auth/whoami", key=value, timeout=5)
    except (OSError, ValueError):
        return False
    connection = who.get("connection") if isinstance(who.get("connection"), dict) else {}
    return code == 200 and connection.get("kind") == "device" and connection.get("app") in ("", APP, None)


def _forget(config, name: str) -> None:
    """Take a variable out of Hermes' .env and this process (Hermes' writer does both, within the profile's
    scope); an older Hermes without remove_env_value empties it."""
    remove = getattr(config, "remove_env_value", None)
    if remove is not None:
        remove(name)
    else:
        config.save_env_value(name, "")
