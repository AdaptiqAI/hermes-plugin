"""AdaptiqAI for Hermes: `/adaptiq` in any chat and `hermes adaptiq` in a terminal sign in without a key.

While the plugin runs, each Hermes conversation keeps its own native memory on AdaptiqAI (wire.py).
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import re
import logging
import os
import threading
import time
import urllib.request
import webbrowser
from collections import OrderedDict
from pathlib import Path

from . import account
from . import adaptiq_core as core
from . import commands
from . import errors
from . import knowledge
from . import mentions
from . import pricing
from . import processed
from . import tools
from . import turns
from . import wire

logger = logging.getLogger(__name__)

PRIVATE = commands.PRIVATE.format(command=commands.COMMAND)
SAVING = commands.SAVING
# Conversations whose choice is kept, newest first.
KEEP_SESSIONS = 500
# How far back a conversation's lineage is followed for a choice.
LINEAGE_DEPTH = 32
# Seconds between the automatic model syncs of one profile in one Hermes process.
SYNC_EVERY = 600

_turns = wire.Turns()
# `state.get` and `state.set` each take the state file's lock on their own, so a choice written
# between them by another conversation of this process would be lost without this.
_state_lock = threading.RLock()
_session_db = None
_session_db_failed = False
_synced_at: dict[str, float] = {}
_sync_lock = threading.Lock()
# The AdaptiqAI model each recent conversation runs on, from its latest request to AdaptiqAI:
# knowledge lives in that model's store. A turn on another provider records nothing.
_models: dict[str, str] = {}
# Recent conversations whose latest request went to another provider (a chat switched away with /model).
_elsewhere: OrderedDict[str, bool] = OrderedDict()
# What the person wrote in each recent conversation, each turn marked when it answers the agent's offer to save
# knowledge, for the create-knowledge guard.
_written = tools.Written()
# Sessions Hermes started for subagents, with the session that started each when Hermes names it:
# their "user" messages are another agent's words.
_subagents: OrderedDict[str, str | None] = OrderedDict()
# Kanban task → the conversation that made it, once looked up ("" when none did): see _kanban_origin.
_kanban: dict[str, str] = {}
# How often, and for how long, a new collection is checked before it is used in its conversation.
ATTACH_CHECK_EVERY = 15
ATTACH_FOLLOW_FOR = 2 * 3600
# Seconds one readiness check may take; it runs beside the turn, never in its way.
ATTACH_TIMEOUT = 3
# The ready collections each recent conversation was last told it reads, so a turn is told again only when
# they change: Hermes keeps a turn's note in that turn's message.
_noted: OrderedDict[str, frozenset] = OrderedDict()
# Seconds one lookup of what a conversation reads may take; it runs beside the turn.
NOTE_TIMEOUT = 5
# Seconds a turn waits for that lookup: when nothing is known about the conversation yet (its first turn here),
# and when the last lookup is known, after which the turn goes on with it. Hermes runs pre_llm_call before the
# turn's first request and waits for it.
NOTE_WAIT = 2.0
NOTE_WAIT_KNOWN = 1.0
# What each recent conversation reads, as last looked up beside its turns.
_reads: OrderedDict[str, dict] = OrderedDict()
# Conversations whose collections are being checked beside their turn right now.
_attaching: set[str] = set()
# Conversations told the knowledge instructions in this process: by the system prompt section, or in a turn.
_instructed: OrderedDict[str, bool] = OrderedDict()
# Where each conversation's memory is: the AdaptiqAI model its first finished turn there ran on, in the state file,
# as a Hermes conversation is often resumed in another process. Its memory stays on that model.
MEMORY_ON = "memory_on"
# Each turn's conversation and model, from its start (pre_llm_call) to its reply (transform_llm_output).
_routes: OrderedDict[str, tuple[str, str, bool]] = OrderedDict()
# The model each recent session's latest turn ran on, for /adaptiq.
_last_turn: OrderedDict[str, str] = OrderedDict()
# The model each recent conversation was last told its memory is not on: once per switch.
_switch_told: OrderedDict[str, str] = OrderedDict()


# The browser approval waiting now, for the status: its code, until it expires.
_signing_in: dict = {}


def _finish_in_background(started: dict) -> None:
    _signing_in.clear()
    _signing_in.update(code=started.get("user_code"), until=time.monotonic() + int(started.get("expires_in") or 600))

    def finish() -> None:
        try:
            model, models = core.apply(core.wait(started))
            logger.info("AdaptiqAI connected: %d models, default %s", len(models), model)
            pricing.register(_adaptiq_entry()[0])
        except Exception as error:  # a decline, an expiry or a network failure ends the sign-in
            logger.warning("AdaptiqAI sign-in did not finish: %s", error)
        finally:
            if _signing_in.get("code") == started.get("user_code"):
                _signing_in.clear()

    # copy_context keeps the caller's profile scope, so the key lands in the right profile.
    threading.Thread(target=contextvars.copy_context().run, args=(finish,),
                     name="adaptiq-sign-in", daemon=True).start()


def _profile_home() -> str:
    try:
        from hermes_constants import get_hermes_home
    except ImportError:
        return ""
    return str(get_hermes_home())


def _sync_in_background() -> None:
    """Keep the provider entry and the main model in step with the lanes, at most every SYNC_EVERY."""
    home = _profile_home()
    with _sync_lock:
        if time.monotonic() - _synced_at.get(home, -SYNC_EVERY) < SYNC_EVERY:
            return
        # Claimed now, so two conversations starting together sync once; given back when the
        # sync did not happen, so the next session start tries again.
        _synced_at[home] = time.monotonic()

    def run() -> None:
        try:
            result = core.sync()
        except Exception as error:
            _synced_at.pop(home, None)
            logger.warning("AdaptiqAI could not sync the models: %s", error)
            return
        if not result.get("synced"):
            _synced_at.pop(home, None)
        # The sync read the rows and their prices; an entry it just wrote is priced from now on.
        pricing.register(_adaptiq_entry()[0])
        if result.get("moved_from"):
            logger.info("AdaptiqAI: %s has no lanes any more; the main model is %s now",
                        result["moved_from"], result["main"])

    # copy_context keeps the caller's profile scope, so the right config is written.
    threading.Thread(target=contextvars.copy_context().run, args=(run,),
                     name="adaptiq-sync", daemon=True).start()


def _status_text(state=None, session: str = "") -> str:
    """`/adaptiq` and `/adaptiq status` (core's statusText): the connection, this conversation, the memory and the
    models, after bringing the models in step."""
    if _signing_in and time.monotonic() < _signing_in.get("until", 0):
        return account.unconnected_text("signing-in", _signing_in.get("code"))
    if core.key() is None:
        return account.unconnected_text(core.phase())
    try:
        synced = core.sync()
    except Exception as error:  # the status still answers
        logger.warning("AdaptiqAI could not sync the models: %s", error)
        synced = {}
    model = (_models.get(session) or core.main_model()) if session else None
    conversation = None
    if session and state is not None and _on_adaptiq(session):
        try:
            view = knowledge.conversation_view(_conversation(session), model)
            conversation = {"memory_tokens": view["memory_tokens"], "remember": _remember(state, session),
                            "knowledge": len(view["knowledge"])}
        except knowledge.KnowledgeError:  # the rest still answers
            conversation = {"memory_tokens": None, "remember": _remember(state, session), "knowledge": 0}
    summary = core.account_summary()
    if summary is None:
        # A gateway without the summary, a connection that may not read it, or every key refused.
        try:
            connection = core.status()
        except (OSError, ValueError):  # unreachable, slow or not JSON
            return "AdaptiqAI is not reachable right now. Try again in a moment."
        if not connection["signed_in"]:
            return account.unconnected_text(connection["phase"])
        first = [f"Connected to {connection['workspace'] or 'your workspace'}"
                 f"{' (API key)' if connection.get('kind') == 'api_key' else ''}"]
        return " · ".join([*first, *([account.conversation_part(conversation)] if conversation else [])])
    connection = summary.get("connection") if isinstance(summary.get("connection"), dict) else {}
    return account.status_text(summary, conversation=conversation, model=model, main=synced.get("main"),
                               moved_from=synced.get("moved_from"), sleep=core.session_memory(),
                               api_key=connection.get("kind") == "api_key")


def _current_session() -> str:
    """The Hermes session a slash command runs in, or "" outside a conversation."""
    try:
        from gateway.session_context import get_session_env
    except ImportError:
        return ""
    return get_session_env("HERMES_SESSION_ID", "")


def _choice(state, session_id: str) -> bool | None:
    """The choice made for one conversation, or None when none was."""
    sessions = state.get("sessions", {})
    entry = sessions.get(session_id) if isinstance(sessions, dict) else None
    return entry["remember"] if isinstance(entry, dict) and isinstance(entry.get("remember"), bool) else None


def _sessions_db():
    """Hermes' session store, read-only, for a session's lineage; None where it cannot be opened."""
    global _session_db, _session_db_failed
    if _session_db is None and not _session_db_failed:
        try:
            from hermes_state import SessionDB
            _session_db = SessionDB(read_only=True)
        except Exception:  # not inside Hermes, or no store yet: the hook's parent still counts
            _session_db_failed = True
    return _session_db


# Compaction continuations already traced to the conversation they continue; that never changes.
_roots: OrderedDict[str, str] = OrderedDict()


def _conversation(session_id: str) -> str:
    """The AdaptiqAI conversation a Hermes session belongs to: the session its compactions started from.

    Hermes compacts in place by default, keeping the session id. Where it rotates instead
    (`compression.in_place: false`, or an older Hermes), the continuation gets a new id; sent as it is,
    that would start a second AdaptiqAI memory beside the first and grow both. A subagent, or a branch,
    is a conversation of its own. Without the session store, a session is its own conversation.
    """
    if not session_id or session_id in _subagents:
        return session_id
    known = _roots.get(session_id)
    if known is not None:
        return known
    db = _sessions_db()
    if db is None:
        return session_id
    root = session_id
    try:
        if hasattr(db, "get_compression_lineage"):  # Hermes' own walk, which knows forks from continuations
            lineage = db.get_compression_lineage(session_id) or [session_id]
            root = str(lineage[0])
        else:
            seen = {session_id}
            for _ in range(LINEAGE_DEPTH):
                row = db.get_session(root) or {}
                parent_id = row.get("parent_session_id") if isinstance(row, dict) else None
                parent = db.get_session(parent_id) if parent_id and parent_id not in seen else None
                if not isinstance(parent, dict) or parent.get("end_reason") != "compression":
                    break
                seen.add(parent_id)
                root = parent_id
    except Exception:  # a store that can't answer: the session stays its own conversation
        return session_id
    if root != session_id:
        _roots[session_id] = root
        while len(_roots) > KEEP_SESSIONS:
            _roots.popitem(last=False)
    return root


def _ancestors(session_id: str, parent_session_id: str = "") -> list[str]:
    """The conversations this one continues, nearest first.

    Hermes gives a conversation a new session id when it compacts it by rotation, and runs a subagent under
    one of its own. The hook names a subagent's parent; the session store keeps the chain a
    compaction leaves, which the hook does not follow.
    """
    chain: list[str] = []
    seen = {session_id}
    if parent_session_id and parent_session_id not in seen:
        chain.append(parent_session_id)
        seen.add(parent_session_id)
    db = _sessions_db()
    if db is None:
        return chain
    for start in (session_id, parent_session_id):
        current = start
        for _ in range(LINEAGE_DEPTH):
            if not current:
                break
            try:
                row = db.get_session(current) or {}
            except Exception:
                break
            parent = row.get("parent_session_id") if isinstance(row, dict) else None
            if not parent or parent in seen:
                break
            seen.add(parent)
            chain.append(parent)
            current = parent
    return chain


def _remember(state, session_id: str, parent_session_id: str = "") -> bool:
    """Whether new messages of a conversation are saved: yes, unless the person made it private.

    A conversation with no choice of its own takes the choice of the one it continues, and keeps
    it: a private conversation stays private across a compaction, and in the subagents it runs.
    """
    own = _choice(state, session_id)
    if own is not None:
        return own
    for ancestor in _ancestors(session_id, parent_session_id):
        inherited = _choice(state, ancestor)
        if inherited is not None:
            if not inherited:
                _set_remember(state, session_id, False)
            return inherited
    return True


# A Hermes session id, as Hermes makes them and as the plugin's state keys allow.
SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


@contextlib.contextmanager
def _remember_lock(state):
    """One writer of the conversations' choices at a time, across processes and across the
    dashboard's own copy of this module: state.get and state.set each lock alone, so a choice
    written between them by the other would be lost. A lock file of its own beside the state
    file, so it never waits on the state file's lock."""
    with _state_lock:
        folder = getattr(state, "data_dir", None)
        if not folder:
            yield
            return
        lock_path = Path(folder) / ".adaptiq-remember.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as handle:
            try:
                import fcntl
            except ImportError:  # Windows: the module lock is what there is
                fcntl = None
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _set_remember(state, session_id: str, remember: bool) -> None:
    with _remember_lock(state):
        sessions = state.get("sessions", {})
        sessions = dict(sessions) if isinstance(sessions, dict) else {}
        sessions[session_id] = {"remember": remember, "at": int(time.time())}
        newest = sorted(sessions.items(), key=lambda item: item[1].get("at", 0), reverse=True)[:KEEP_SESSIONS]
        state.set("sessions", dict(newest))


def _messaging_surface() -> bool:
    """Whether the command came over a messaging channel, where no browser on this machine is the person's.
    A check that cannot answer counts as one: then the page is shown, not opened."""
    try:
        from gateway.session_context import session_is_messaging_surface
        return session_is_messaging_surface()
    except ImportError:  # not a gateway build: a terminal
        return False
    except Exception:
        return True


def _open_page(url: str) -> bool:
    """Open a page in the person's browser, where there is one here."""
    return _browser_available() and bool(webbrowser.open(url))


def _browser_available() -> bool:
    """Whether a page can be opened here in a graphical browser, as Hermes' own device sign-in asks.
    On a headless machine webbrowser.open would start lynx or w3m and take over the terminal."""
    if _messaging_surface():
        return False
    try:
        from hermes_cli.auth_device_flow import _can_open_graphical_browser
        return bool(_can_open_graphical_browser())
    except Exception:
        return False


def _model_of(session: str) -> str | None:
    """The AdaptiqAI model a conversation runs on: the one its latest request named, else Hermes' main model."""
    return _models.get(session) or core.main_model()


def _knowledge_text(session: str) -> str:
    """`/adaptiq knowledge` (core's knowledgeText)."""
    conversation = _conversation(session)
    view = knowledge.conversation_view(conversation, _model_of(session))
    used = [{"name": item["name"], "status": item["status"]} for item in view["knowledge"]]
    model = view.get("model") or _model_of(session)
    others, more = [], False
    if model:
        try:  # a list that cannot be read is left out, never shown as "nothing else"
            page = knowledge.library(model, None, None, frozenset(item["memory_id"] for item in view["knowledge"]))
            others = [item["name"] for item in page["items"] if item["kind"] == "collection" and item["fits"]
                      and item["readiness"] == "ready" and not item["selected"]]
            more = page["has_more"]
        except knowledge.KnowledgeError:
            pass
    return commands.knowledge_text(used, view["conversations"], others, more)


def _by_name(items: list[dict], name: str) -> dict | list[dict] | None:
    """The one item a name means: an exact name first, else the only one whose name contains it; a list when
    several do (never guessed)."""
    wanted = tools.same_name(name)
    exact = [item for item in items if tools.same_name(item["name"]) == wanted]
    if exact:
        return exact[0]
    partial = [item for item in items if wanted in tools.same_name(item["name"])]
    return partial[0] if len(partial) == 1 else partial or None


def _use(session: str, name: str) -> str:
    """`/adaptiq use <name>` (core's use): a collection for this conversation's model, else an earlier conversation
    on the same model, read from memory."""
    command = commands.COMMAND
    wanted = tools.same_name(name)
    if not wanted:
        return f"Say which: {command} use <name>."
    conversation = _conversation(session)
    view = knowledge.conversation_view(conversation, _model_of(session))
    here = next((item for item in [*view["knowledge"], *view["conversations"]]
                 if tools.same_name(item["name"]) == wanted), None)
    if here is not None:
        return f'"{here["name"]}" is already used in this conversation.'
    model = view.get("model") or _model_of(session)
    if not model:
        return "Send a message on an AdaptiqAI model first."
    page = knowledge.library(model, name, None, frozenset(item["memory_id"] for item in view["knowledge"]))
    collections = [item for item in page["items"] if item["kind"] == "collection"]
    found = _by_name([item for item in collections if item["fits"]], name)
    if isinstance(found, list):
        return commands.several(name, [item["name"] for item in found])
    if found is not None:
        if found["readiness"] != "ready":
            return f'"{found["name"]}" is {found["status"].lower()}; use it again once it is ready.'
        knowledge.use(conversation, model, found["memory_id"], True)
        return f'"{found["name"]}" is now used in this conversation.'
    if any(not item["fits"] and tools.same_name(item["name"]) == wanted for item in collections):
        return (f'"{name}" is made for another model. Make it available for this one in AdaptiqAI Knowledge: '
                f"{command} open.")
    earlier = knowledge.conversations(model, conversation, name)
    found = _by_name([item for item in earlier["items"] if not item["attached"]], name)
    if isinstance(found, list):
        return commands.several(name, [item["name"] for item in found])
    if found is not None:
        if not earlier["attachable"]:
            return "Earlier conversations cannot be added here yet."
        knowledge.read_conversation(conversation, model, found["conversation_id"], True)
        return f'The conversation "{found["name"]}" is now read in this one, from memory.'
    return f'Nothing saved for this model is called "{name}". {command} knowledge lists what you can use.'


def _stop(session: str, name: str) -> str:
    """`/adaptiq stop <name>` (core's stop): a collection or an earlier conversation used here, by its exact name.
    Nothing is deleted."""
    command = commands.COMMAND
    wanted = tools.same_name(name)
    conversation = _conversation(session)
    view = knowledge.conversation_view(conversation, _model_of(session))
    model = view.get("model") or _model_of(session)
    collections = [item for item in view["knowledge"] if tools.same_name(item["name"]) == wanted]
    conversations = [item for item in view["conversations"] if tools.same_name(item["name"]) == wanted]
    if len(collections) + len(conversations) > 1:
        return f'Two things used here are called "{name}". Stop the one you mean in AdaptiqAI Knowledge: {command} open.'
    if collections:
        knowledge.use(conversation, model, collections[0]["memory_id"], False)
        return f'"{collections[0]["name"]}" is no longer used in this conversation. Nothing was deleted.'
    if conversations:
        knowledge.read_conversation(conversation, model, conversations[0]["conversation_id"], False)
        return f'The conversation "{conversations[0]["name"]}" is no longer read here. Nothing was deleted.'
    return f'Nothing used in this conversation is called "{name}". {command} knowledge lists what is used.'


def _sign_in() -> str:
    """`/adaptiq signin`: start the browser approval; the key and models are set when it is approved."""
    if core.key() is not None:
        try:
            connection = core.status()
        except (OSError, ValueError):
            connection = {}
        if connection.get("signed_in"):
            return f"Already connected to {connection.get('workspace') or 'your workspace'}."
    started = core.start()
    _finish_in_background(started)
    url = started["verification_uri_complete"]
    return commands.sign_in_text(url, started["user_code"], _open_page(url))


def _open(session: str) -> str:
    """`/adaptiq open`: this conversation in AdaptiqAI Knowledge, or its link where no browser here is the person's."""
    url = knowledge.console_link(_conversation(session) if session else None)
    return "Opened AdaptiqAI Knowledge in your browser." if _open_page(url) else f"Open {url}"


# The last failures AdaptiqAI answered in this process, newest first, for the diagnostics.
_failures: list[dict] = []
FAILURES_KEPT = 10


def _noted_failure(status_code, model: str, verdict: dict | None) -> None:
    _failures.insert(0, {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": status_code,
                         "model": model, "reason": (verdict or {}).get("reason"),
                         "message": (verdict or {}).get("message")})
    del _failures[FAILURES_KEPT:]


def _plugin_version() -> str:
    try:
        text = (Path(__file__).resolve().parent / "plugin.yaml").read_text(encoding="utf-8")
    except OSError:
        return ""
    found = re.search(r"^version:\s*(\S+)", text, re.M)
    return found.group(1) if found else ""


# The published plugin's manifest: the mirror Hermes installs it from (`hermes plugins install AdaptiqAI/hermes-plugin`).
MIRROR_MANIFEST = "https://raw.githubusercontent.com/AdaptiqAI/hermes-plugin/main/plugin.yaml"
# The newest version seen there and when it was last read, in the state file; it is read at most once a day.
UPDATE_STATE = "update_check"
UPDATE_EVERY = 24 * 3600
# A plain release, the only kind offered (core's RELEASE_VERSION).
RELEASE_VERSION = re.compile(r"\d+\.\d+\.\d+")


def _newer(a: str, b: str) -> bool:
    """Whether version `a` is newer than `b` (major.minor.patch; core's semverGreater)."""
    def parts(value: str) -> list[int]:
        found = [re.match(r"\d+", part) for part in re.split(r"[.-]", value.lstrip("v"))[:3]]
        return [int(match.group()) if match else 0 for match in found] + [0] * (3 - len(found))
    return parts(a) > parts(b)


def _published_version() -> str | None:
    """The version the mirror publishes, or None when it can't be read (offline, or a mirror not public)."""
    request = urllib.request.Request(MIRROR_MANIFEST, headers={"User-Agent": "adaptiqai-hermes-plugin"})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            text = response.read(65536).decode("utf-8", "replace")
    except (OSError, ValueError):
        return None
    found = re.search(r"^version:\s*['\"]?([^'\"\s]+)", text, re.M)
    return found.group(1) if found and RELEASE_VERSION.fullmatch(found.group(1)) else None


def _check_for_update(state) -> None:
    """Read the published version, at most once a day (core's checkForUpdate, from the mirror Hermes installs from
    rather than npm). Nothing is said when it can't be read."""
    with _state_lock:
        known = state.get(UPDATE_STATE, {})
        known = known if isinstance(known, dict) else {}
        at = known.get("at")
        if isinstance(at, (int, float)) and time.time() - at < UPDATE_EVERY:
            return
        state.set(UPDATE_STATE, {**known, "at": time.time()})
    latest = _published_version()
    if latest is not None:
        with _state_lock:
            state.set(UPDATE_STATE, {**state.get(UPDATE_STATE, {}), "latest": latest})


def _update_notice(state) -> dict | None:
    """A newer plugin is published: its version, and the line that says so with the command that installs it.
    Hermes can't update a plugin by itself, so the person runs it (Pi's words, with Hermes' command)."""
    known = state.get(UPDATE_STATE, {}) if state is not None else {}
    latest = known.get("latest") if isinstance(known, dict) else None
    running = _plugin_version()
    if not (isinstance(latest, str) and RELEASE_VERSION.fullmatch(latest) and running and _newer(latest, running)):
        return None
    command = f"hermes plugins update {Path(__file__).resolve().parent.name}"
    return {"version": latest,
            "text": f"AdaptiqAI {latest} is available. Update it with {command}, then restart Hermes."}


def _hermes_version() -> str:
    try:
        from importlib.metadata import version
        return version("hermes-agent")
    except Exception:
        return ""


def _diagnostics(state, session: str) -> str:
    """`/adaptiq diagnostics`: a report without secrets, to send to support."""
    try:
        connection = core.status()
    except (OSError, ValueError):
        connection = {"signed_in": False, "phase": "unreachable"}
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        cfg = {}
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    entry = core._entry_name(providers, core._main_provider(cfg))
    models = (providers.get(entry) or {}).get("models") if entry else None
    search = ((cfg.get("tools") or {}).get("tool_search") or {}) if isinstance(cfg.get("tools"), dict) else {}
    report = {
        "plugin": _plugin_version(),
        "host": _hermes_version(),
        "apiBase": core.API,
        "phase": connection.get("phase") or core.phase(),
        "source": {"device": "device", "api_key": "api-key"}.get(connection.get("kind")),
        "workspace": connection.get("workspace") or None,
        "keys": len(core.keys()),
        "provider": entry,
        "mainProvider": core._main_provider(cfg),
        "models": sorted(models) if isinstance(models, dict) else [],
        "toolSearch": search.get("enabled") if isinstance(search, dict) else None,
        "conversation": ({"id": session, "remember": _remember(state, session), "model": _models.get(session)}
                         if session else None),
        "lastFailures": list(_failures),
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return "\n".join(["AdaptiqAI diagnostics (no secrets; safe to share with support):", "```json",
                      json.dumps(report, indent=2, ensure_ascii=False), "```"])


def _slash_handler(state):
    """`/adaptiq …` in a chat (core's runCommand): the same actions and words as in every harness."""

    def slash(raw_args: str, session_id: str | None = None, **_) -> str:
        # Hermes names the conversation to a handler that asks (its plugin_command_dispatch); an older one sets it
        # in the session environment only.
        action, rest = commands.parse_command(raw_args or "")
        session = session_id if isinstance(session_id, str) and session_id else _current_session()
        try:
            if action in ("", "status", "refresh"):
                switch = _switch_status(state, session) if core.key() is not None else None
                update = _update_notice(state)
                return "\n".join([_status_text(state, session), *([switch] if switch else []),
                                  *([update["text"]] if update else [])])
            if action == "help":
                return commands.command_help()
            if action in ("signin", "login", "connect"):
                return _sign_in()
            if action in ("signout", "logout"):
                core.sign_out()
                return commands.SIGNED_OUT
            if action == "open":
                return _open(session)
            if action == "diagnostics":
                return _diagnostics(state, session)
            if action not in ("private", "remember", "knowledge", "use", "stop"):
                return commands.unknown(action)
            if not session:
                return commands.OPEN_FIRST
            if action in ("private", "remember"):
                _set_remember(state, session, action == "remember")
                return SAVING if action == "remember" else PRIVATE
            if action == "knowledge":
                return _knowledge_text(session)
            if not rest.strip():
                return commands.say_which(action)
            return _use(session, rest) if action == "use" else _stop(session, rest)
        except knowledge.KnowledgeError as error:  # a refusal, in one sentence
            return str(error)
        except RuntimeError as error:  # the sign-in could not start
            return str(error)
        except (OSError, ValueError):  # unreachable, slow or not JSON
            return "AdaptiqAI is not reachable right now. Try again in a moment."

    return slash


def _text_of(message) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return content if isinstance(content, str) else ""


def _persons(message) -> bool:
    """Whether a history message is the person's own words. Hermes stores its compaction summaries
    and its nudges as user messages too; the model wrote those, and a summary that says \"the user
    asked to save this\" is not the user asking."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if message.get("_compressed_summary") or message.get("display_kind") == "hidden":
        return False
    return not any(key.endswith("_synthetic") and value for key, value in message.items())


def _said(messages, current=None) -> list[dict]:
    """A conversation's messages as turns.user_turns reads them: the person's own, typed or given through clarify,
    and the agent's. Other tools' results, and Hermes' own notes and summaries, are neither. `current` is the
    person's message of a turn that is not in `messages` yet."""
    said: list[dict] = []
    tools_called: dict = {}
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            calls = [call for call in message.get("tool_calls") or [] if isinstance(call, dict)]
            arguments = []
            for call in calls:
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                tools_called[call.get("id")] = function.get("name")
                raw = function.get("arguments")
                arguments.append(raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False))
            said.append({"from": "agent", "text": _text_of(message), "tool_arguments": arguments})
        elif role == "tool":
            if tools_called.get(message.get("tool_call_id")) == "clarify":
                said += _clarified(message)
        elif _persons(message):
            said.append({"from": "person", "reply": "typed", "text": _text_of(message)})
    if current is not None:
        said.append({"from": "person", "reply": "typed", "text": current})
    return said


def _clarified(message) -> list[dict]:
    """The person's answers in a clarify result, each after the question it answers: Hermes' ask-user tool."""
    try:
        result = json.loads(_text_of(message))
    except ValueError:
        return []
    said = []
    for response in result.get("responses") or [] if isinstance(result, dict) else []:
        if not isinstance(response, dict) or response.get("status") != "answered":
            continue
        answer = response.get("user_response")
        answer = ", ".join(str(part) for part in answer) if isinstance(answer, list) else str(answer or "")
        said += [{"from": "agent", "text": "", "tool_arguments": [str(response.get("question") or "")]},
                 {"from": "person", "reply": "question", "text": answer}]
    return said


def _person_wrote(session_id: str) -> list[str] | None:
    """Every message the person wrote in a conversation, oldest first, for find_in_conversation: read from Hermes'
    session store when a find asks, as it keeps every row. Compaction archives rows and a restart keeps them, so what
    a summary replaced and what an earlier run held are there; what the person took back (undo, rewind) is not
    (`get_messages(include_compacted=True, include_ancestors=True)`, Hermes' display transcript). A subagent reads
    the conversation it works for, as does a Kanban worker; a subagent whose parent is unknown has none at hand.
    Raises when the store can't be read."""
    if session_id in _subagents and not _subagents[session_id]:
        return None
    source = _subagents.get(session_id) or _kanban_origin() or session_id
    db = _sessions_db()
    if db is None:
        # Not opened for the hooks: opened here, or the read fails.
        from hermes_state import SessionDB
        db = SessionDB(read_only=True)
    return _written_by_person(db.get_messages(source, include_compacted=True, include_ancestors=True))


def _written_by_person(messages) -> list[str]:
    """What the person wrote, a text a message, oldest first: their own messages as they typed them, and their
    answers to the agent's questions (clarify). Never Hermes' notes, summaries or scaffolding, the agent's replies or
    other tools' output."""
    written: list[str] = []
    tools_called: dict = {}
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and isinstance(call.get("function"), dict):
                    tools_called[call.get("id")] = call["function"].get("name")
        elif role == "tool":
            if tools_called.get(message.get("tool_call_id")) == "clarify":
                written += [said["text"] for said in _clarified(message) if said["from"] == "person"]
        elif role == "user":
            typed = _typed(message)
            if typed is not None:
                written.append(typed)
    return written


# How Hermes' own notes stored as the person's messages begin: its system notices and crash-recovery note (the TUI
# hides them, tui_gateway/session_history.py:228 and 261), and what its compressor counts as no user words
# (agent/context_compressor.py:934): compaction hand-offs, delegations, task lists, cron output.
_HERMES_NOTES = ("[System:", "[System note: Your previous turn was interrupted mid-run", "[CONTEXT", "[PRIOR CONTEXT",
                 "[IMPORTANT: Background", "[Your active task list", "[Planning state preserved", "[ASYNC DELEGATION",
                 "[OUT-OF-BAND", "Cronjob Response:")


def _typed(message: dict) -> str | None:
    """A stored user message as the person typed it, or None when Hermes wrote it: a compaction summary, a message
    Hermes types itself (display_kind: hidden, a model or personality switch, a finished process or delegation, a
    resumed turn) or one of its untyped notes. A steer is the person's words inside Hermes' markers, and a /skill
    turn the command as typed, never the skill's text, as Hermes' TUI shows them
    (tui_gateway/session_history.py:326-361)."""
    if message.get("_compressed_summary") or any(
            key.endswith("_synthetic") and value for key, value in message.items()):
        return None
    kind = message.get("display_kind")
    text = _text_of(message)
    if kind == "steer":
        return _steered(text)
    if kind or text.lstrip().startswith(_HERMES_NOTES):
        return None
    return _as_typed(text)


def _steered(text: str) -> str:
    """The person's words in a steer sent while a turn ran, between Hermes' markers
    (agent/prompt_builder.py:562-566); the message as stored where Hermes has other markers."""
    try:
        from agent.prompt_builder import STEER_MARKER_CLOSE, STEER_MARKER_OPEN
    except Exception:  # noqa: BLE001 - another Hermes
        return text
    start = text.find(STEER_MARKER_OPEN)
    end = text.find(STEER_MARKER_CLOSE, start + len(STEER_MARKER_OPEN)) if start >= 0 else -1
    return text[start + len(STEER_MARKER_OPEN):end].strip() if end >= 0 else text


def _as_typed(text: str) -> str:
    """A /skill turn as the person typed it ("/work fix the title"): Hermes sends the skill's whole text in the
    person's message (agent/skill_commands.py:154). Any other message as it is."""
    try:
        from agent.skill_commands import describe_skill_invocation
    except Exception:  # noqa: BLE001 - another Hermes
        return text
    return describe_skill_invocation(text, separator=" ") or text


def _since_the_person(messages) -> list:
    """A request's messages after the person's message that started its turn."""
    messages = messages if isinstance(messages, list) else []
    last = max((index for index, message in enumerate(messages)
                if isinstance(message, dict) and message.get("role") == "user"), default=-1)
    return messages[last + 1:]


def _follow(state, session_id: str, memory_id: str, model: str | None, name: str = "") -> None:
    """Use a new collection in its conversation as soon as it is ready."""
    with _state_lock:
        pending = state.get("pending_attach", {})
        pending = dict(pending) if isinstance(pending, dict) else {}
        pending[memory_id] = {"session": session_id, "model": model, "name": name, "at": int(time.time()),
                              "checked": 0}
        state.set("pending_attach", pending)
    _attach_ready(state, session_id)


def _unfollow(state, memory_id: str) -> None:
    """Stop waiting to use a collection: it was taken back."""
    with _state_lock:
        pending = state.get("pending_attach", {})
        pending = dict(pending) if isinstance(pending, dict) else {}
        if pending.pop(memory_id, None) is not None:
            state.set("pending_attach", pending)


def _preparing(state, session_id: str) -> list[dict]:
    """The collections saved in a conversation that are used there as soon as they are ready."""
    with _state_lock:
        pending = state.get("pending_attach", {})
    return [{"memory_id": memory_id, "name": entry["name"]} for memory_id, entry in
            (pending.items() if isinstance(pending, dict) else ())
            if isinstance(entry, dict) and entry.get("session") == session_id and entry.get("name")]


def _attach_ready(state, session_id: str) -> None:
    """Attach the collections made in this conversation that are ready now. One that failed, or took over two
    hours, is let go, whichever conversation it was made in. A turn runs it beside itself (`_attach_soon`)."""
    with _state_lock:
        pending = state.get("pending_attach", {})
        pending = dict(pending) if isinstance(pending, dict) else {}
    now = int(time.time())
    stale = [memory_id for memory_id, entry in pending.items()
             if not isinstance(entry, dict) or now - int(entry.get("at") or 0) > ATTACH_FOLLOW_FOR]
    if stale:
        with _state_lock:
            kept = state.get("pending_attach", {})
            kept = dict(kept) if isinstance(kept, dict) else {}
            for memory_id in stale:
                kept.pop(memory_id, None)
            state.set("pending_attach", kept)
        pending = {key: value for key, value in pending.items() if key not in stale}
    due = {memory_id: entry for memory_id, entry in pending.items()
           if isinstance(entry, dict) and entry.get("session") == session_id
           and now - int(entry.get("checked") or 0) >= ATTACH_CHECK_EVERY}
    if not due:
        return
    done, checked = set(), set()
    for memory_id, entry in due.items():
        if now - int(entry.get("at") or 0) > ATTACH_FOLLOW_FOR:
            done.add(memory_id)
            continue
        try:
            readiness = knowledge.readiness(memory_id, entry.get("model"), timeout=ATTACH_TIMEOUT)
            if readiness == "ready":
                knowledge.use(_conversation(session_id), entry.get("model"), memory_id, True)
            if readiness in ("ready", "failed"):
                done.add(memory_id)
            else:
                checked.add(memory_id)
        except knowledge.KnowledgeError as error:  # tried again on a later turn
            logger.info("AdaptiqAI: collection %s is not attached yet: %s", memory_id, error)
            checked.add(memory_id)
    with _state_lock:
        pending = state.get("pending_attach", {})
        pending = dict(pending) if isinstance(pending, dict) else {}
        for memory_id in done:
            pending.pop(memory_id, None)
        for memory_id in checked:
            if memory_id in pending:
                pending[memory_id] = {**pending[memory_id], "checked": now}
        state.set("pending_attach", pending)


def _beside(name: str, work, *args) -> threading.Thread:
    """Run work in the background, in the caller's profile scope (copy_context keeps it)."""
    thread = threading.Thread(target=contextvars.copy_context().run, args=(work, *args), name=name, daemon=True)
    thread.start()
    return thread


def _attach_soon(state, session_id: str) -> None:
    """Attach the collections made in this conversation that are ready now, beside the turn: a readiness check
    never holds up a turn, and the turn's next request reads what got attached."""
    with _state_lock:
        if session_id in _attaching:
            return
        _attaching.add(session_id)

    def run() -> None:
        try:
            _attach_ready(state, session_id)
        except Exception as error:  # tried again on a later turn
            logger.info("AdaptiqAI: collections were not checked: %s", error)
        finally:
            with _state_lock:
                _attaching.discard(session_id)

    _beside("adaptiq-attach", run)


def _read_here(conversation: str, model: str) -> list[dict] | None:
    """What a conversation reads, without holding up its turn for long: a new lookup runs beside the turn, which
    waits for it at most NOTE_WAIT seconds when nothing is known yet, else NOTE_WAIT_KNOWN and then goes on with
    the last lookup. None when unknown."""
    key = f"{conversation}\n{model}"
    with _state_lock:
        entry = _reads.get(key)
        if entry is None or not entry["running"]:
            entry = {"used": entry["used"] if entry else None, "running": True, "done": threading.Event()}
            _reads.pop(key, None)
            _reads[key] = entry
            while len(_reads) > KEEP_SESSIONS:
                _reads.popitem(last=False)
            start = True
        else:
            start = False

    def run(entry: dict = entry) -> None:
        try:
            entry["used"] = knowledge.read_here(conversation, model, NOTE_TIMEOUT)
        except Exception as error:  # told on a later turn
            logger.info("AdaptiqAI: what this conversation reads is not known now: %s", error)
        finally:
            entry["running"] = False
            entry["done"].set()

    if start:
        _beside("adaptiq-reads", run)
    entry["done"].wait(NOTE_WAIT if entry["used"] is None else NOTE_WAIT_KNOWN)
    return entry["used"]


def _reading_note(session_id: str, used: list[dict] | None) -> str | None:
    """A turn's note on the collections its conversation reads (`pre_llm_call` context), when they are new
    to it."""
    if used is None:
        return None
    ready = frozenset(item["memory_id"] for item in tools.ready_here(used))
    with _state_lock:
        if not ready or _noted.get(session_id) == ready:
            return None
        _noted.pop(session_id, None)
        _noted[session_id] = ready
        while len(_noted) > KEEP_SESSIONS:
            _noted.popitem(last=False)
    return tools.turn_note(used)


def _turn_on_adaptiq(session_id: str, model: str) -> str:
    """The AdaptiqAI model a turn runs on, or "": its model is one of the AdaptiqAI entry's, or Hermes keeps the
    conversation on AdaptiqAI's URL (a chat switched there with /model). pre_llm_call names the model, not the
    provider."""
    if model and model in _adaptiq_models():
        return model
    # The conversation's latest request went to AdaptiqAI, on this same model.
    if model and _models.get(session_id) == model and _on_adaptiq(session_id):
        return model
    db = _sessions_db()
    try:
        row = db.get_session(session_id) if db is not None else None
    except Exception:  # a store that can't answer: the last request to AdaptiqAI says it
        row = None
    if isinstance(row, dict) and str(row.get("billing_base_url") or "").rstrip("/") == core.API:
        return model or str(row.get("model") or "")
    return ""


def _untold(session_id: str, history) -> list[str]:
    """The instructions a conversation on AdaptiqAI lacks, in the order they are told: Hermes freezes a
    conversation's system prompt when it starts, so one started before this plugin had them (or before it had the
    note on find_in_conversation), or on another provider before /model moved it to AdaptiqAI, never gets them
    there. Told once, in a turn, they stay in its history."""
    if _instructed.get(session_id):
        return []
    # Hermes keeps a turn's note in its message's api_content, which later turns replay.
    said = [text for message in history or [] for text in (
        _text_of(message), str(message.get("api_content") or "") if isinstance(message, dict) else "")]
    untold = [paragraph for paragraph in INSTRUCTIONS if not any(paragraph in text for text in said)]
    if not untold:
        _instructed[session_id] = True
        return []
    db = _sessions_db()
    try:
        row = db.get_session(session_id) if db is not None else None
    except Exception:
        row = None
    prompt = row.get("system_prompt") if isinstance(row, dict) else None
    return [paragraph for paragraph in untold if not (isinstance(prompt, str) and paragraph in prompt)]


def _told(session_id: str) -> None:
    with _state_lock:
        _instructed.pop(session_id, None)
        _instructed[session_id] = True
        while len(_instructed) > KEEP_SESSIONS:
            _instructed.popitem(last=False)


def _request_hooks(state, surface=None):
    """`on_session_start` brings the models in step with the lanes. `pre_llm_call` names each turn
    Hermes runs for a person, and tells it which collections its conversation reads; the `llm_request`
    middleware then shapes every request to AdaptiqAI by its turn. Hermes skips both hooks for forks. `surface`
    names the client of the session a hook runs for (Hermes' `ctx.session_surface`)."""

    def on_desktop() -> bool:
        try:
            return surface is not None and surface() == "desktop"
        except Exception:  # noqa: BLE001 - outside a session: a terminal
            return False

    def on_session_start(**_) -> None:
        _sync_in_background()
        _beside("adaptiq-update", _check_for_update, state)

    def on_turn(session_id: str = "", turn_id: str = "", parent_session_id: str = "",
                user_message=None, conversation_history=None, model: str = "", **_) -> dict | None:
        # A subagent, named by its parent, is a conversation of its own; a compaction is not.
        conversation = session_id if parent_session_id else _conversation(session_id)
        if session_id and turn_id:
            # Hermes names a parent for delegated subagents: each works for its parent, a helper. A Kanban worker
            # works on a task of its own, made in another conversation: a track. The parent is named by its
            # AdaptiqAI conversation, which a rotated compaction keeps.
            parent = parent_session_id or _subagents.get(session_id) or ""
            role = "helper"
            if not parent and _kanban_origin() not in ("", session_id):
                parent, role = _kanban_origin(), "track"
            _turns.note(turn_id, wire.Turn(session_id, _remember(state, session_id, parent_session_id or ""),
                                           conversation=conversation if conversation != session_id else "",
                                           parent=_conversation(parent) if parent else "", role=role))
        # A subagent's "user" is another agent: announced by subagent_start, or named by its parent.
        if session_id and session_id not in _subagents and not parent_session_id:
            history = list(conversation_history or [])
            current = user_message if isinstance(user_message, str) else _text_of({"content": user_message})
            # Hermes passes the history with this turn's message last; an older one passed it without.
            last = history[-1] if history else None
            started = isinstance(last, dict) and last.get("role") == "user"
            _written.note(session_id, turns.user_turns(_said(history, None if started else current)))
        if not session_id:
            return None
        _attach_soon(state, session_id)
        adaptiq_model = _turn_on_adaptiq(session_id, model)
        if turn_id and (adaptiq_model or model):
            with _state_lock:
                tools._keep(_routes, turn_id, (conversation, adaptiq_model or model, bool(adaptiq_model)),
                            KEEP_SESSIONS)
                tools._keep(_last_turn, session_id, adaptiq_model or model, KEEP_SESSIONS)
        if not adaptiq_model or not tools.available():
            return None
        if not on_desktop():
            _read_notices_soon()
        context = []
        # A person's message that mentions earlier conversations with @adaptiq: (a subagent's comes from its parent).
        mentioned = [] if parent_session_id or session_id in _subagents else mentions.targets(
            user_message if isinstance(user_message, str) else _text_of({"content": user_message}))
        if mentioned:
            recalled = _recall(state, session_id, conversation, adaptiq_model, mentioned)
            if recalled:
                context.append("\n".join(f"AdaptiqAI: {line}" for line in recalled))
                if turn_id:
                    with _state_lock:
                        tools._keep(_recalled, turn_id, recalled, KEEP_SESSIONS)
        untold = _untold(session_id, conversation_history)
        if untold:
            context += untold
            _told(session_id)
        note = _reading_note(session_id, _read_here(conversation, adaptiq_model))
        if note:
            context.append(note)
        return {"context": "\n\n".join(context)} if context else None

    def on_request(request=None, turn_id: str = "", base_url: str = "", api_mode: str = "",
                   model: str = "", session_id: str = "", **_):
        here = (base_url or "").rstrip("/") == core.API
        _note_provider(session_id or getattr(_turns.get(turn_id or ""), "session_id", ""), here)
        if api_mode != "chat_completions" or not isinstance(request, dict) or not here:
            return None
        turn = _turns.get(turn_id or "")
        if turn is not None and model:
            # The hook that names the turn knows Hermes' model, not its provider; only a request
            # that reaches AdaptiqAI says which of its models the conversation runs on.
            with _state_lock:
                _models.pop(turn.session_id, None)
                _models[turn.session_id] = model
                while len(_models) > KEEP_SESSIONS:
                    _models.pop(next(iter(_models)))
        if turn is not None:
            # The person's answers to questions the agent asked through clarify during this turn: a yes to its
            # offer to save knowledge counts from the next request.
            answered = turns.user_turns(_said(_since_the_person(request.get("messages"))))
            if answered:
                _written.answered(turn.session_id, answered)
            # `/adaptiq private` typed during a long turn holds from the next request of it, not
            # the next turn: the choice is read again, one read of the state file per request.
            own = _choice(state, turn.session_id)
            if own is not None and own != turn.remember:
                turn.remember, turn.stored = own, None
            if not turn.remember and turn.stored is None:
                turn.stored = core.session_stored(turn.memory_id, model)
        shaped = wire.by_card(wire.shape(request, turn), core.model_row(model), _effort_chosen(model))
        return {"request": shaped, "source": "adaptiq",
                "reason": "a turn of a conversation" if turn else "not a turn of a conversation: ephemeral"}

    def on_reply(response_text: str = "", turn_id: str = "", **_) -> str | None:
        """`transform_llm_output`: a turn's reply, once it is final. The first reply on a model the conversation's
        memory is not on ends with where it is and how to go on, once per switch (Hermes prints what a plugin adds
        after the streamed reply); a conversation's first finished turn on AdaptiqAI notes the model its memory
        is on. Outside Desktop, a reply on AdaptiqAI also says a new memory warning or sleeping conversation."""
        with _state_lock:
            route = _routes.pop(turn_id, None) if turn_id else None
        if route is None or not isinstance(response_text, str) or not response_text:
            return None
        conversation, model, on_adaptiq = route
        lines = []
        home = _memory_on(state, conversation)
        if home is None:
            if on_adaptiq:
                _keep_memory_on(state, conversation, model)
        else:
            with _state_lock:
                told = home == model or _switch_told.get(conversation) == model
                if not told:
                    tools._keep(_switch_told, conversation, model, KEEP_SESSIONS)
            if not told:
                lines.append(_switch_line(home, model))
        with _state_lock:
            lines += [f"AdaptiqAI: {line}" for line in _recalled.pop(turn_id, [])]
        if on_adaptiq and not on_desktop():
            lines += _new_notices()
        return "\n\n".join([response_text, *lines]) if lines else None

    return on_session_start, on_turn, on_request, on_reply


def _effort_chosen(model: str) -> bool:
    """Whether the person chose a thinking level for this model in Hermes (agent.reasoning_effort, or one for this
    model in agent.reasoning_overrides). Without one, Hermes sends medium, and the model's card says its default."""
    try:
        from hermes_cli.config import load_config
        agent = load_config().get("agent") or {}
    except Exception:
        return True  # can't tell: what Hermes sends stands
    if not isinstance(agent, dict):
        return False
    overrides = agent.get("reasoning_overrides")
    effort = agent.get("reasoning_effort")
    return (isinstance(overrides, dict) and model in overrides) or effort is False or (
        isinstance(effort, str) and bool(effort.strip()))


def _note_provider(session: str, on_adaptiq: bool) -> None:
    """Remember whether a conversation's latest request went to AdaptiqAI."""
    if not session:
        return
    with _state_lock:
        _elsewhere.pop(session, None)
        if not on_adaptiq:
            _elsewhere[session] = True
            while len(_elsewhere) > KEEP_SESSIONS:
                _elsewhere.popitem(last=False)


def _on_adaptiq(session: str) -> bool:
    """Whether a conversation runs on AdaptiqAI, as far as this process knows: unless its latest request went
    to another provider (core's onAdaptiq: unknown counts as yes)."""
    return session not in _elsewhere


def _adaptiq_entry() -> tuple[str | None, dict]:
    """The provider entry on AdaptiqAI's URL: its name in config.yaml and what it holds; (None, {}) outside Hermes
    or without one."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
        name = core._entry_name(providers, core._main_provider(cfg))
        entry = providers.get(name) if name else None
        return (name, entry) if isinstance(entry, dict) else (None, {})
    except Exception:
        return None, {}


def _adaptiq_models() -> set[str]:
    """The models of the provider entry on AdaptiqAI's URL; empty outside Hermes or without one."""
    _, entry = _adaptiq_entry()
    models = entry.get("models") if isinstance(entry.get("models"), dict) else {}
    return set(models) | ({entry["model"]} if entry.get("model") else set())


def _memory_on(state, conversation: str) -> str | None:
    """The AdaptiqAI model a conversation's memory is on, or None while it has none there."""
    homes = state.get(MEMORY_ON, {}) if state is not None and conversation else {}
    model = homes.get(conversation) if isinstance(homes, dict) else None
    return model if isinstance(model, str) and model else None


def _keep_memory_on(state, conversation: str, model: str) -> None:
    """Note the model a conversation's memory is on: the one its first finished turn on AdaptiqAI ran on."""
    with _state_lock:
        homes = state.get(MEMORY_ON, {})
        homes = dict(homes) if isinstance(homes, dict) else {}
        if homes.get(conversation):
            return
        homes[conversation] = model
        while len(homes) > KEEP_SESSIONS:
            homes.pop(next(iter(homes)))
        state.set(MEMORY_ON, homes)


def _model_name(model: str) -> str:
    """A model's name as the workspace shows it, else its id (another provider's model)."""
    return str((core.model_row(model) or {}).get("display_name") or model)


def _switch_line(home: str, model: str) -> str:
    """Where a conversation's memory is while it runs on another model, and how to go on: switch back, or start
    fresh (OpenClaw's elsewhereLine, with Hermes' commands)."""
    entry, _ = _adaptiq_entry()
    back = f"/model {home}" + (f" --provider {entry}" if entry else "")
    return (f"AdaptiqAI: this conversation's memory is on {_model_name(home)}, not {_model_name(model)}. {back} "
            f"switches back to it; /new starts fresh on {_model_name(model)}.")


def _switch_status(state, session: str) -> str | None:
    """`/adaptiq`'s line for a conversation whose latest turn ran on a model its memory is not on."""
    model = _last_turn.get(session) if session else None
    home = _memory_on(state, _conversation(session)) if model else None
    return _switch_line(home, model) if home and home != model else None


# The workspace memory's warning and its sleeping conversations as last read beside a turn, and what a reply already
# said of them: a terminal or chat has no other place for DeepSeek Harness's notices, so the first reply on AdaptiqAI
# after one appears says it, once (a warning again when it rises). Hermes Desktop raises them itself (plugin.js).
NOTICES_EVERY = 60
_notices: dict = {"at": None, "warning": None, "sleeping": []}
# What came of a turn's @adaptiq: mentions, by turn id, for its reply.
_recalled: OrderedDict = OrderedDict()
_told_notices: dict = {"level": 0, "sleeping": set()}


def _read_notices() -> None:
    """The memory warning and the sleeping conversations, as Desktop's settings page reads them."""
    try:
        summary = core.account_summary()
        if summary is None:
            return
        view = account.view(summary, sleep=core.session_memory())
    except Exception as error:  # noqa: BLE001 - the next turn reads again
        logger.debug("AdaptiqAI: could not read the memory notices: %s", error)
        return
    with _state_lock:
        _notices.update(warning=view["warning"], sleeping=list(view["sleeping"]))


def _read_notices_soon() -> None:
    """Read the notices beside the turn, at most every NOTICES_EVERY: its reply says what is new."""
    with _state_lock:
        if _notices["at"] is not None and time.monotonic() - _notices["at"] < NOTICES_EVERY:
            return
        _notices["at"] = time.monotonic()
    _beside("adaptiq-notices", _read_notices)


def _new_notices() -> list[str]:
    """What no reply has said yet: the warning when its level rose, and each sleeping line once."""
    with _state_lock:
        warning = _notices["warning"]
        level = warning["level"] if warning else 0
        lines = [warning["text"]] if warning and level > _told_notices["level"] else []
        _told_notices["level"] = level
        for line in _notices["sleeping"]:
            if line not in _told_notices["sleeping"]:
                _told_notices["sleeping"].add(line)
                lines.append(line)
    return [f"AdaptiqAI: {line}" for line in lines]


def _typing_in() -> tuple[str, str | None]:
    """The conversation and AdaptiqAI model of the chat an `@adaptiq:` mention is typed in: the session Hermes names,
    else the latest turn on AdaptiqAI in this process, on the main model when none is known."""
    models = _adaptiq_models()
    session = _current_session()
    with _state_lock:
        recent = list(_last_turn.items())
    if not session:
        session = next((name for name, model in reversed(recent) if model in models), "")
    model = (_models.get(session) or dict(recent).get(session)) if session else None
    if model not in models:
        model = core.main_model() if core.key() is not None else None
    return (_conversation(session) if session else ""), model


def _recall(state, session_id: str, conversation: str, model: str, mentioned: list[str]) -> list[str]:
    """Add the earlier conversations a person's message mentions to this one, so the model reads them from memory
    (core's recallConversations): only ones on this model, nothing while the conversation is private, one that
    failed before tried once more, and at most RECALL_WAIT seconds for them to be ready. Returns what came of each,
    for the turn's message and the reply."""
    if not _remember(state, session_id):
        return [f"Private: a conversation you mention with @ is not added, so it is not read from memory here. "
                f"{commands.COMMAND} remember saves this conversation again."]
    name = _model_name(model)
    try:
        items = mentions.earlier(model, conversation, fresh=True)
    except knowledge.KnowledgeError as error:
        return [f"The conversations you mentioned could not be looked up: {error}"]
    lines, found = [], []
    for target in mentioned:
        match = mentions.resolve(items, target)
        if isinstance(match, list):
            names = ", ".join(f'"{item["name"]}"' for item in match[:5])
            lines.append(f'Several earlier conversations are called "{target}": {names}. Mention one by its full name.')
        elif match is None:
            lines.append(f'No earlier conversation on {name} is called "{target}".')
        elif match not in found:
            found.append(match)
    if not found:
        return lines
    try:
        here = {item["conversation_id"]: item for item in knowledge.conversation_view(conversation, model)["conversations"]}
    except knowledge.KnowledgeError:
        here = {}
    added = []
    for item in found:
        current = here.get(item["conversation_id"])
        if current is not None and current["readiness"] != "failed":
            continue  # read here already: later turns keep it
        try:
            knowledge._revision(conversation, model)  # a conversation AdaptiqAI does not store yet is stored first
            if current is not None:
                knowledge.read_conversation(conversation, model, item["conversation_id"], False)
            knowledge.read_conversation(conversation, model, item["conversation_id"], True)
            added.append(item)
        except knowledge.KnowledgeError as error:
            lines.append(f'"{item["name"]}" could not be added: {error}')
    waiting = {item["conversation_id"] for item in added}
    deadline = time.monotonic() + mentions.RECALL_WAIT
    while waiting and time.monotonic() < deadline:
        time.sleep(mentions.RECALL_POLL)
        try:
            ready = {item["conversation_id"]: item["readiness"]
                     for item in knowledge.conversation_view(conversation, model)["conversations"]}
        except knowledge.KnowledgeError:
            break
        waiting = {conversation_id for conversation_id in waiting if ready.get(conversation_id) == "preparing"}
    for item in added:
        lines.append(f'"{item["name"]}" is added and still getting ready: your next message reads it from memory.'
                     if item["conversation_id"] in waiting else
                     f'"{item["name"]}" is now read in this conversation, from memory.')
    return lines


def _failed_at_adaptiq(error, model: str) -> bool:
    """Whether a failed request was AdaptiqAI's.

    The request's URL says so where the error carries one, on itself or on what caused it. An
    error raised inside the stream carries none; then the model says whose the request was.
    """
    exc = error
    for _ in range(6):
        if exc is None:
            break
        request = getattr(getattr(exc, "response", None), "request", None)
        url = str(getattr(request, "url", "") or "")
        if url:
            return url.startswith(core.API + "/")
        exc = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    return bool(model) and model in _adaptiq_models()


def _conversation_of(error) -> str:
    """The AdaptiqAI conversation a failed request was sent for: its session header, on the error or on what caused
    it. "" for a request without one (a helper call, a fork) or an error raised inside the stream."""
    exc = error
    for _ in range(6):
        if exc is None:
            break
        request = getattr(getattr(exc, "response", None), "request", None)
        if request is not None:
            value = (getattr(request, "headers", None) or {}).get(wire.SESSION_HEADER)
            return value if isinstance(value, str) else ""
        exc = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    return ""


# Each conversation's "getting ready" answers in a row (core's preparingRun). A request that went through, or any other
# answer, ends the run.
_preparing_run: OrderedDict[str, int] = OrderedDict()


def _getting_ready(conversation: str, model: str, verdict: dict | None) -> dict | None:
    """A conversation getting ready a second time in a row, with the model's conversation memory full: nothing gets it
    ready, so the turn ends now and says why, instead of Hermes' five minutes of retries that end on a promise to retry
    (core's adapter). The first is waited for as before, without asking."""
    if not conversation:
        return verdict
    with _state_lock:
        before = _preparing_run.pop(conversation, 0)
    if verdict is None or verdict.get("message") != errors.PREPARING:
        return verdict
    full = core.session_memory_full(model) if before else None
    if full is not None:
        return errors.memory_full_preparing(account.format_tokens(full["used"]), account.format_tokens(full["limit"]))
    with _state_lock:
        tools._keep(_preparing_run, conversation, before + 1, KEEP_SESSIONS)
    return verdict


def _quote(error, words: str) -> None:
    """Put the plugin's words where Hermes quotes the provider.

    Hermes words a turn it won't retry itself, by the verdict's reason, and then adds "Provider said:" with the error
    body's message (`agent/api_error_summary.py`): the hook's own `message` never reaches the chat. For a full memory
    the engine's words suggest a new conversation, which frees nothing, and a conversation that can't get ready says
    only "Preparing". So the body's message becomes the plugin's sentence, with the engine's type and code beside it.
    The rest of the body stays, and so does the error's own text, which Hermes logs and dumps.
    """
    body = getattr(error, "body", None)
    _, kind, code = errors.server_detail(body)
    said = {"message": words, **({"type": kind} if kind else {}), **({"code": code} if code else {})}
    try:
        error.body = {**(body if isinstance(body, dict) else {}), "error": said}
    except (AttributeError, TypeError):  # an error that takes no new attributes keeps the engine's words
        pass


def _on_api_error(model: str = "", status_code: int | None = None, error_body=None, error=None, **_):
    """Hermes' recovery and wording for a failed request to AdaptiqAI (errors.py); None for others."""
    if not _failed_at_adaptiq(error, model):
        return None
    # Only a full memory depends on the server: does it put idle conversations to sleep?
    sleeps = errors.is_memory_full(status_code, error_body) and core.auto_sleep(model)
    verdict = errors.classify(status_code, error_body, model, auto_sleep=sleeps)
    verdict = _getting_ready(_conversation_of(error), model, verdict)
    if errors.ends_for_full_memory(verdict):
        _quote(error, verdict["message"])
    _noted_failure(status_code, model, verdict)
    return verdict


# What a refusal asks of the person (errors.action), and each conversation's processed tokens: Hermes Desktop's
# routes run in another copy of this module, so both live in the plugin's state file.
ACTION = "action"
PROCESSED = "processed"
PROCESSED_KEPT = 100


def _error_hook(state):
    """`transform_api_error_classification`: Hermes' wording (`_on_api_error`), and what the person can do about it
    kept for Desktop's buttons."""

    def on_error(model: str = "", status_code: int | None = None, error_body=None, error=None, **rest):
        verdict = _on_api_error(model=model, status_code=status_code, error_body=error_body, error=error, **rest)
        kind = errors.action(status_code, error_body) if verdict is not None else None
        if kind:
            with _state_lock:
                state.set(ACTION, {"kind": kind, "model": model, "at": int(time.time())})
        return verdict

    return on_error


def _counter(state):
    """`post_api_request`: count each request to AdaptiqAI into its conversation, priced as Hermes' own counter is;
    one that went through ends the refusal Desktop shows a button for, and the conversation's run of "getting ready"
    answers."""

    def on_response(session_id: str = "", turn_id: str = "", model: str = "", base_url: str = "", usage=None, **_):
        if (base_url or "").rstrip("/") != core.API or not session_id:
            return
        conversation = _conversation(session_id)
        with _state_lock:
            _preparing_run.pop(conversation, None)
        if not isinstance(usage, dict):
            return
        one = processed.step(usage, pricing.request_cost(core.model_row(model), _Usage(usage)))
        with _state_lock:
            kept = state.get(PROCESSED, {})
            kept = dict(kept) if isinstance(kept, dict) else {}
            record = kept.pop(conversation, None)
            kept[conversation] = processed.count(record, one, turn_id or "", time.time())
            while len(kept) > PROCESSED_KEPT:
                kept.pop(next(iter(kept)))
            state.set(PROCESSED, kept)
            if state.get(ACTION):
                state.set(ACTION, None)

    return on_response


class _Usage:
    """Hermes' usage dict, read as its CanonicalUsage is (attributes)."""

    def __init__(self, usage: dict) -> None:
        self.__dict__.update(usage)

    def __getattr__(self, name: str):
        return 0


def _look_up_kanban_origin(task_id: str) -> None:
    """Read which conversation made a Kanban task: the task keeps the HERMES_SESSION_ID it was made in (none when
    it was made from the CLI or the dashboard). Runs beside the worker's start, as the board's database can wait
    on a writer."""
    origin = ""
    try:
        from hermes_cli import kanban_db
        from hermes_cli.kanban_db_connect import connect_closing
        with connect_closing() as conn:
            task = kanban_db.get_task(conn, task_id)
        origin = str(getattr(task, "session_id", "") or "")
    except Exception as error:  # no board here, or a Hermes without Kanban: no parent
        logger.info("AdaptiqAI: the conversation that made Kanban task %s is not known: %s", task_id, error)
    _kanban[task_id] = origin


def _kanban_origin() -> str:
    """The conversation that made the Kanban task this process works on, as far as it is looked up yet, else "".

    Hermes' dispatcher starts each Kanban worker as a process of its own, with HERMES_KANBAN_TASK. The worker is a
    line of work of its own, started from that conversation: a track (DeepSeek Harness's Agent Team member), where
    a delegated subagent is a helper that works for its parent."""
    task_id = os.environ.get("HERMES_KANBAN_TASK", "")
    return _kanban.get(task_id, "") if task_id else ""


def _on_subagent_start(child_session_id: str = "", parent_session_id: str = "", **_) -> None:
    if child_session_id:
        _subagents[child_session_id] = parent_session_id or None
        while len(_subagents) > KEEP_SESSIONS:
            _subagents.popitem(last=False)


def _cli(args) -> None:
    """`hermes adaptiq [status|signin|signout]`: the status by default, as `/adaptiq` in a chat."""
    action = getattr(args, "adaptiq_command", None) or "status"
    if action in ("signout", "logout"):
        core.sign_out()
        print(commands.SIGNED_OUT)
        return
    if action not in ("signin", "login"):
        print(_status_text())
        return
    started = core.start()
    url = started["verification_uri_complete"]
    # The same guard as in a chat: on a headless machine a browser would take over the terminal.
    opened = _open_page(url)
    print(f"{'Your browser opened' if opened else 'Open this link'} to approve Hermes: {url} (code {started['user_code']})")
    model, models = core.apply(core.wait(started))
    print(f"Connected: {len(models)} models, default {model}." if model
          else f"Connected. No models yet: {commands.COMMAND} open shows your workspace in the console.")


def _setup_cli(subparser) -> None:
    commands_parser = subparser.add_subparsers(dest="adaptiq_command")
    commands_parser.add_parser("status", help="Show the connection, memory and models (the default)")
    commands_parser.add_parser("signin", aliases=["login"], help="Sign in in the browser (also signs in again)")
    commands_parser.add_parser("signout", aliases=["logout"], help="Forget this computer's sign-in")
    subparser.set_defaults(func=_cli)


# What a new conversation on an AdaptiqAI model is told in its system prompt: core's KNOWLEDGE_INSTRUCTIONS
# (packages/core/src/reading.ts), the same bytes Kilo and OpenCode get. Without it an agent asked about an id
# searched the person's files and past chats, or read "answer from memory, no terminal" as no tools at all.
KNOWLEDGE_INSTRUCTIONS = (
    "AdaptiqAI memory: before you answer about a specific id, name, number or date, look it up with "
    "adaptiq_knowledge, action find, with the id or the words, and answer from the lines it returns, citing "
    "document:line. Do that first: a collection this conversation uses (adaptiq_knowledge status names it) is in "
    "your memory, not on disk, so grep, glob, read and bash do not reach its documents. Each reply shows you only "
    "the part of it that matches the conversation, never all of it: documents you do not see are still there. "
    "adaptiq_knowledge is not the terminal, a local file, the web, an earlier chat or a subagent, so use it also "
    "when the person asks you to answer from memory without those. Never name a document as a source unless find "
    "or open showed it to you; if find fails, say what it answered.")
# Told right after them, as the knowledge tool offers find_in_conversation: core's CONVERSATION_FIND_INSTRUCTIONS
# (reading.ts, plugins #121). Asked for the fines over 40 operations notes, MiMo tried to recall every note in its
# reasoning, ran ten minutes and answered nothing: the tool spoke of saved knowledge, and it never counted from it.
CONVERSATION_FIND_INSTRUCTIONS = (
    "This conversation: to count, add up or list what was said or noted earlier in it (notes, logs, figures, names), "
    "use adaptiq_knowledge, action find_in_conversation, with the words to look for, and answer from the lines it "
    "returns. A long conversation is read in parts, so never count from what you remember of it.")
INSTRUCTIONS = (KNOWLEDGE_INSTRUCTIONS, CONVERSATION_FIND_INSTRUCTIONS)


def _system_prompt_section(info) -> str:
    """The plugin's section of a new session's system prompt (Hermes freezes it for the session): the knowledge
    instructions, on an AdaptiqAI model while the knowledge tool is offered; nothing elsewhere. A conversation
    that gets them later, on its first turn on AdaptiqAI, is told them in that turn instead."""
    model = str(info.get("model") or "")
    if not model or model not in _adaptiq_models() or not tools.available():
        return ""
    if info.get("session_id"):
        _told(str(info["session_id"]))
    return "\n\n".join(INSTRUCTIONS)


def register(ctx) -> None:
    # An entry written before helper calls ran ephemeral (0.1.0) would send a title or a
    # compression summary of a private conversation with no memory mode, and it would be saved.
    try:
        if core.mark_entry():
            logger.info("AdaptiqAI: helper calls of this Hermes now run ephemeral.")
    except Exception as error:  # no Hermes config here: nothing to mark
        logger.warning("AdaptiqAI: could not mark the provider entry: %s", error)
    # Hermes' own counter prices AdaptiqAI's turns from the models' prices (pricing.py).
    pricing.register(_adaptiq_entry()[0])
    on_session_start, on_turn, on_request, on_reply = _request_hooks(ctx.state, getattr(ctx, "session_surface", None))
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("pre_llm_call", on_turn)
    ctx.register_hook("transform_llm_output", on_reply)
    ctx.register_hook("transform_api_error_classification", _error_hook(ctx.state))
    ctx.register_hook("post_api_request", _counter(ctx.state))
    ctx.register_middleware("llm_request", on_request)
    knowledge_tool, memory_tool = tools.handlers(
        _models.get, lambda session: _remember(ctx.state, session),
        lambda session, remember: _set_remember(ctx.state, session, remember), _conversation,
        lambda session: _preparing(ctx.state, session), _open_page, lambda session: not _on_adaptiq(session),
        written_of=_person_wrote)
    ctx.register_tool("adaptiq_knowledge", tools.TOOLSET, tools.KNOWLEDGE, knowledge_tool, check_fn=tools.available,
                      description="Find and use AdaptiqAI Knowledge in this conversation")
    ctx.register_tool("adaptiq_memory", tools.TOOLSET, tools.MEMORY, memory_tool, check_fn=tools.available,
                      description="Check and look after AdaptiqAI shared memory")
    create_tool = tools.create_handler(
        _written, _models.get,
        lambda session, memory_id, model, name: _follow(ctx.state, session, memory_id, model, name),
        lambda memory_id: _unfollow(ctx.state, memory_id), _conversation, state=ctx.state)
    ctx.register_tool("adaptiq_create_knowledge", tools.TOOLSET, tools.CREATE, create_tool, check_fn=tools.available,
                      description="Save AdaptiqAI Knowledge the user asked for")
    ctx.register_hook("subagent_start", _on_subagent_start)
    task_id = os.environ.get("HERMES_KANBAN_TASK", "")
    if task_id and task_id not in _kanban:
        _beside("adaptiq-kanban", _look_up_kanban_origin, task_id)
    # Earlier conversations by @adaptiq: in the message (mentions.py); a Hermes without references goes without.
    conversations = mentions.provider(_typing_in) if hasattr(ctx, "register_context_reference") else None
    if conversations is not None:
        ctx.register_context_reference(conversations)
    # Hermes added system prompt sections in 2026-08; an older one goes without.
    if hasattr(ctx, "register_system_prompt_section"):
        ctx.register_system_prompt_section("adaptiq-knowledge", _system_prompt_section)
    ctx.register_command("adaptiq", _slash_handler(ctx.state), description="AdaptiqAI: workspace, memory and knowledge",
                         args_hint=f"[{commands.COMMAND_HINT.replace(' · ', '|')}]")
    ctx.register_cli_command(name="adaptiq", help="AdaptiqAI: status, sign in and sign out", setup_fn=_setup_cli,
                             handler_fn=_cli)
