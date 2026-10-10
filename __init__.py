"""AdaptiqAI for Hermes: `/adaptiq` in any chat and `hermes adaptiq` in a terminal sign in without a key."""

from __future__ import annotations

import contextvars
import logging
import threading
import webbrowser

from . import adaptiq_core as core

logger = logging.getLogger(__name__)


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


def _slash(raw_args: str) -> str:
    action = (raw_args or "").strip().lower() or "login"
    if action == "status":
        state = core.status()
        return f"AdaptiqAI: connected to {state['workspace']}" if state["signed_in"] else "AdaptiqAI: not connected."
    if action != "login":
        return "Usage: /adaptiq [login|status]"
    started = core.start()
    _finish_in_background(started)
    return (f"Approve AdaptiqAI in your browser: {started['verification_uri_complete']} "
            f"(code {started['user_code']}). The key and models are set when you approve; "
            "new chats then use your workspace's model.")


def _cli(args) -> None:
    if getattr(args, "adaptiq_command", None) == "status":
        print(core.status())
        return
    started = core.start()
    print(f"Approve AdaptiqAI in your browser: {started['verification_uri_complete']} (code {started['user_code']})")
    webbrowser.open(started["verification_uri_complete"])
    model, models = core.apply(core.wait(started))
    print(f"Connected: {len(models)} models, default {model}.")


def _setup_cli(subparser) -> None:
    commands = subparser.add_subparsers(dest="adaptiq_command")
    commands.add_parser("login", help="Sign in in the browser (also signs in again)")
    commands.add_parser("status", help="Show the connection")
    subparser.set_defaults(func=_cli)


def register(ctx) -> None:
    ctx.register_command("adaptiq", _slash, description="Sign in to AdaptiqAI and use your workspace's models",
                         args_hint="[login|status]")
    ctx.register_cli_command(name="adaptiq", help="Sign in to AdaptiqAI", setup_fn=_setup_cli, handler_fn=_cli)
