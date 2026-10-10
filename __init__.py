"""AdaptiqAI for Hermes: `/adaptiq` in any chat and `hermes adaptiq` in a terminal sign in without a key.

While the plugin runs, each Hermes conversation keeps its own native memory on AdaptiqAI (wire.py).
"""

from __future__ import annotations

import contextlib
import contextvars
import re
import logging
import threading
import time
import webbrowser
from collections import OrderedDict
from pathlib import Path

from . import account
from . import adaptiq_core as core
from . import errors
from . import knowledge
from . import purchase
from . import tools
from . import wire

logger = logging.getLogger(__name__)

USAGE = "Usage: /adaptiq [login|status|private|remember|knowledge|models|get <number> [lanes]]"
PRIVATE = ("Private: new messages of this conversation are no longer saved to workspace memory. "
           "Existing memory is kept. /adaptiq remember turns saving back on.")
SAVING = "Saving: new messages of this conversation are saved to workspace memory again."
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
# What the person wrote in each recent conversation, for the create-knowledge guard.
_written = tools.Written()
# Sessions Hermes started for subagents: their "user" messages are another agent's words.
_subagents: OrderedDict[str, None] = OrderedDict()
# How often, and for how long, a new collection is checked before it is used in its conversation.
ATTACH_CHECK_EVERY = 15
ATTACH_FOLLOW_FOR = 2 * 3600
# Seconds a readiness check may take at the start of a turn.
ATTACH_TIMEOUT = 3


def _finish_in_background(started: dict) -> None:
    def finish() -> None:
        try:
            model, models = core.apply(core.wait(started))
            logger.info("AdaptiqAI connected: %d models, default %s", len(models), model)
        except Exception as error:  # a decline, an expiry or a network failure ends the sign-in
            logger.warning("AdaptiqAI sign-in did not finish: %s", error)

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
        if result.get("moved_from"):
            logger.info("AdaptiqAI: %s has no lanes any more; the main model is %s now",
                        result["moved_from"], result["main"])

    # copy_context keeps the caller's profile scope, so the right config is written.
    threading.Thread(target=contextvars.copy_context().run, args=(run,),
                     name="adaptiq-sync", daemon=True).start()


def _status_text(conversation: str | None = None) -> str:
    """The account, its models and lanes, after bringing the models in step."""
    try:
        synced = core.sync()
    except Exception as error:  # the status still answers
        logger.warning("AdaptiqAI could not sync the models: %s", error)
        synced = {}
    summary = core.account_summary()
    if summary is not None:
        return account.status_text(summary, conversation=conversation, main=synced.get("main"),
                                   moved_from=synced.get("moved_from"), sleep=core.session_memory())
    # A gateway without the summary, or a connection that may not read it.
    try:
        connection = core.status()
    except (OSError, ValueError):  # unreachable, slow or not JSON
        return "AdaptiqAI is not reachable right now. Try again in a moment."
    if not connection["signed_in"]:
        return "AdaptiqAI: not connected. /adaptiq login connects it."
    line = f"AdaptiqAI: connected to {connection['workspace']}"
    return line + (f" · this conversation is {conversation}" if conversation else "")


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


def _knowledge_text() -> str:
    session = _current_session()
    if not session:
        return "Open a conversation first."
    try:
        view = knowledge.conversation_view(_conversation(session), _models.get(session))
    except knowledge.KnowledgeError as error:
        return str(error)
    if not view["knowledge"]:
        return ("No knowledge is used in this conversation. Ask the agent to find and attach a collection, "
                f"or open AdaptiqAI Knowledge: {knowledge.console_link(_conversation(session))}")
    return "\n".join(["Knowledge in this conversation:",
                      *(f"• {item['name']} — {item['status']}" for item in view["knowledge"])])


def _models_text() -> str:
    """The models on offer, with what the workspace already has."""
    summary = core.account_summary() or {}
    owned = {str(model.get("display_name") or "").lower(): int(model.get("lanes") or 0)
             for model in summary.get("models") or [] if isinstance(model, dict)}
    try:
        items, mode = purchase.offers()
    except purchase.PurchaseError as error:
        return str(error)
    return purchase.offers_text(items, mode, owned)


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


# Purchases being followed, by offer: a second `/adaptiq get` for one shows the waiting page instead
# of asking again, so a missed tab never becomes two rentals.
_purchases: dict[str, dict] = {}
_purchases_lock = threading.Lock()


def _get(words: list[str]) -> str:
    """`/adaptiq get <number|offer> [lanes]`: ask for lanes, and follow the purchase until they are on."""
    from hermes_cli.config import get_env_value

    if not words:
        return "Usage: /adaptiq get <number> [lanes]. /adaptiq models lists the models you can get."
    key = get_env_value(core.KEY_ENV)
    if not key:
        return "Connect AdaptiqAI first: /adaptiq login."
    try:
        items, mode = purchase.offers()
    except purchase.PurchaseError as error:
        return str(error)
    pick = words[0]
    number = int(pick) if pick.isascii() and pick.isdecimal() else 0
    offer = (items[number - 1] if 1 <= number <= len(items)
             else next((item for item in items if item["id"] == pick), None))
    if offer is None:
        return "That model isn't on offer. /adaptiq models lists the models you can get."
    if len(words) > 1 and not (words[1].isascii() and words[1].isdecimal()):
        return "Give the lanes as a number, for example /adaptiq get 1 2."
    lanes = int(words[1]) if len(words) > 1 else offer["min_lanes"]
    with _purchases_lock:
        waiting = _purchases.get(offer["id"])
    if waiting is not None:
        if waiting["lanes"] != lanes:
            return (f"A purchase of {waiting['lanes']} on {offer['name']} is already waiting for you: "
                    f"{waiting['url']} Finish or cancel it first.")
        return f"A purchase of {offer['name']} is already waiting for you in the browser: {waiting['url']}"
    model_id, purchase_url = purchase.model_for(offer, core.account_summary() or {})

    def lanes_now() -> int:
        return purchase.lanes_in(core._model_rows(key) or [], model_id, offer["name"])

    try:
        baseline = lanes_now()
        started = purchase.start(offer, lanes, key, purchase_url)
    except purchase.PurchaseError as error:
        return str(error)
    except (OSError, ValueError):  # unreachable, slow or not JSON
        return "AdaptiqAI can't be reached right now. Try again in a moment."
    with _purchases_lock:
        _purchases[offer["id"]] = {"lanes": lanes, "url": started["url"]}

    def ready() -> None:
        core.sync()
        rows = core._model_rows(key) or []
        bought = model_id or next((row["id"] for row in rows if str(row.get("display_name") or "").lower()
                                   == offer["name"].lower()), None)
        if bought and core.start_on(bought):
            logger.info("AdaptiqAI: %s is ready and Hermes' main model now", offer["name"])
        else:
            logger.info("AdaptiqAI: %s is ready; /model switches to it", offer["name"])

    def run() -> None:
        try:
            ended = purchase.follow({"request_id": started["request_id"], "baseline": baseline}, key, lanes_now, ready)
        finally:
            with _purchases_lock:
                _purchases.pop(offer["id"], None)
        if ended != "ready":
            logger.info("AdaptiqAI: the purchase of %s ended: %s", offer["name"], ended)

    # copy_context keeps the caller's profile scope, so the right config is written.
    threading.Thread(target=contextvars.copy_context().run, args=(run,), name="adaptiq-purchase", daemon=True).start()
    opened = _browser_available() and webbrowser.open(started["url"])
    count = f"{lanes} {'lane' if lanes == 1 else 'lanes'}"
    lines = [f"{'Your browser opened' if opened else 'Open this page'} to confirm {count} on {offer['name']}: {started['url']}"]
    note = purchase.billing_note(offer, mode)
    if note:
        lines.append(note)
    lines.append(f"{offer['name']} becomes usable here as soon as its lanes are on; /adaptiq status shows them.")
    return "\n".join(lines)


def _slash_handler(state):
    def slash(raw_args: str) -> str:
        words = (raw_args or "").split()
        action = words[0].lower() if words else "login"
        if action == "models":
            return _models_text()
        if action == "get":
            return _get(words[1:])
        action = " ".join(words).lower() or "login"
        if action == "status":
            session = _current_session()
            return _status_text(("saved" if _remember(state, session) else "private") if session else None)
        if action == "knowledge":
            return _knowledge_text()
        if action in ("private", "remember"):
            session = _current_session()
            if not session:
                return "Open a conversation first."
            _set_remember(state, session, action == "remember")
            return SAVING if action == "remember" else PRIVATE
        if action != "login":
            return USAGE
        started = core.start()
        _finish_in_background(started)
        return (f"Approve AdaptiqAI in your browser: {started['verification_uri_complete']} "
                f"(code {started['user_code']}). The key and models are set when you approve; "
                "new chats then use your workspace's model.")

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


def _follow(state, session_id: str, memory_id: str, model: str | None) -> None:
    """Use a new collection in its conversation as soon as it is ready."""
    with _state_lock:
        pending = state.get("pending_attach", {})
        pending = dict(pending) if isinstance(pending, dict) else {}
        pending[memory_id] = {"session": session_id, "model": model, "at": int(time.time()), "checked": 0}
        state.set("pending_attach", pending)
    _attach_ready(state, session_id)


def _attach_ready(state, session_id: str) -> None:
    """Attach the collections made in this conversation that are ready now; run at the start of a turn,
    so the turn's first model call reads them. One that failed, or took over two hours, is let go,
    whichever conversation it was made in. It runs on the turn's way in, so each call is short."""
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


def _request_hooks(state):
    """`on_session_start` brings the models in step with the lanes. `pre_llm_call` names each turn
    Hermes runs for a person; the `llm_request` middleware then shapes every request to AdaptiqAI by
    its turn. Hermes skips both hooks for forks."""

    def on_session_start(**_) -> None:
        _sync_in_background()

    def on_turn(session_id: str = "", turn_id: str = "", parent_session_id: str = "",
                user_message=None, conversation_history=None, **_) -> None:
        if session_id and turn_id:
            # A subagent, named by its parent, is a conversation of its own; a compaction is not.
            conversation = session_id if parent_session_id else _conversation(session_id)
            _turns.note(turn_id, wire.Turn(session_id, _remember(state, session_id, parent_session_id or ""),
                                           conversation=conversation if conversation != session_id else ""))
        # A subagent's "user" is another agent: announced by subagent_start, or named by its parent.
        if session_id and session_id not in _subagents and not parent_session_id:
            history = [_text_of(message) for message in conversation_history or [] if _persons(message)]
            current = user_message if isinstance(user_message, str) else _text_of({"content": user_message})
            _written.note(session_id, [text for text in [*history, current] if text.strip()])
        if session_id:
            _attach_ready(state, session_id)

    def on_request(request=None, turn_id: str = "", base_url: str = "", api_mode: str = "",
                   model: str = "", **_):
        if api_mode != "chat_completions" or not isinstance(request, dict) or (base_url or "").rstrip("/") != core.API:
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
            # `/adaptiq private` typed during a long turn holds from the next request of it, not
            # the next turn: the choice is read again, one read of the state file per request.
            own = _choice(state, turn.session_id)
            if own is not None and own != turn.remember:
                turn.remember, turn.stored = own, None
            if not turn.remember and turn.stored is None:
                turn.stored = core.session_stored(turn.memory_id, model)
        return {"request": wire.shape(request, turn), "source": "adaptiq",
                "reason": "a turn of a conversation" if turn else "not a turn of a conversation: ephemeral"}

    return on_session_start, on_turn, on_request


def _adaptiq_models() -> set[str]:
    """The models of the provider entry on AdaptiqAI's URL; empty outside Hermes or without one."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
        name = core._entry_name(providers, core._main_provider(cfg))
        entry = providers.get(name) if name else None
        if not isinstance(entry, dict):
            return set()
        models = entry.get("models") if isinstance(entry.get("models"), dict) else {}
        return set(models) | ({entry["model"]} if entry.get("model") else set())
    except Exception:
        return set()


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


def _on_api_error(model: str = "", status_code: int | None = None, error_body=None, error=None, **_):
    """Hermes' recovery and wording for a failed request to AdaptiqAI (errors.py); None for others."""
    if not _failed_at_adaptiq(error, model):
        return None
    # Only a full memory depends on the server: does it put idle conversations to sleep?
    sleeps = errors.is_memory_full(status_code, error_body) and core.auto_sleep(model)
    return errors.classify(status_code, error_body, model, auto_sleep=sleeps)


def _on_subagent_start(child_session_id: str = "", **_) -> None:
    if child_session_id:
        _subagents[child_session_id] = None
        while len(_subagents) > KEEP_SESSIONS:
            _subagents.popitem(last=False)


def _cli(args) -> None:
    if getattr(args, "adaptiq_command", None) == "status":
        print(_status_text())
        return
    started = core.start()
    print(f"Approve AdaptiqAI in your browser: {started['verification_uri_complete']} (code {started['user_code']})")
    webbrowser.open(started["verification_uri_complete"])
    model, models = core.apply(core.wait(started))
    print(f"Connected: {len(models)} models, default {model}." if model
          else "Connected. No models yet: /adaptiq models in a chat shows the models you can get.")


def _setup_cli(subparser) -> None:
    commands = subparser.add_subparsers(dest="adaptiq_command")
    commands.add_parser("login", help="Sign in in the browser (also signs in again)")
    commands.add_parser("status", help="Show the connection")
    subparser.set_defaults(func=_cli)


def register(ctx) -> None:
    # An entry written before helper calls ran ephemeral (0.1.0) would send a title or a
    # compression summary of a private conversation with no memory mode, and it would be saved.
    try:
        if core.mark_entry():
            logger.info("AdaptiqAI: helper calls of this Hermes now run ephemeral.")
    except Exception as error:  # no Hermes config here: nothing to mark
        logger.warning("AdaptiqAI: could not mark the provider entry: %s", error)
    on_session_start, on_turn, on_request = _request_hooks(ctx.state)
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("pre_llm_call", on_turn)
    ctx.register_hook("transform_api_error_classification", _on_api_error)
    ctx.register_middleware("llm_request", on_request)
    knowledge_tool, memory_tool = tools.handlers(
        _models.get, lambda session: _remember(ctx.state, session),
        lambda session, remember: _set_remember(ctx.state, session, remember), _conversation)
    ctx.register_tool("adaptiq_knowledge", tools.TOOLSET, tools.KNOWLEDGE, knowledge_tool, check_fn=tools.available,
                      description="Find and use AdaptiqAI Knowledge in this conversation")
    ctx.register_tool("adaptiq_memory", tools.TOOLSET, tools.MEMORY, memory_tool, check_fn=tools.available,
                      description="Check and look after AdaptiqAI shared memory")
    create_tool = tools.create_handler(_written, _models.get,
                                       lambda session, memory_id, model: _follow(ctx.state, session, memory_id, model))
    ctx.register_tool("adaptiq_create_knowledge", tools.TOOLSET, tools.CREATE, create_tool, check_fn=tools.available,
                      description="Save AdaptiqAI Knowledge the user asked for")
    ctx.register_hook("subagent_start", _on_subagent_start)
    ctx.register_command("adaptiq", _slash_handler(ctx.state), description="Sign in to AdaptiqAI and use your workspace's models",
                         args_hint="[login|status|private|remember|knowledge|models|get]")
    ctx.register_cli_command(name="adaptiq", help="Sign in to AdaptiqAI", setup_fn=_setup_cli, handler_fn=_cli)
