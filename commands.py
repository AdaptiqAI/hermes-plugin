"""`/adaptiq` in plain text: core's commands.ts for Hermes, with the same actions and the same words as every other
harness. Only the words in `WORDS` are Hermes' own. __init__.py runs the actions; this module says them."""

from __future__ import annotations

import re

COMMAND = "/adaptiq"


def command_words(app: str, **words: str) -> dict:
    """The words for one harness (core's commandWords): its name, and whatever it says its own way."""
    command = words.get("command", COMMAND)
    return {
        "command": command,
        "app": app,
        "connect": words.get("connect", f"Run {command} signin."),
        "no_models": words.get("no_models", f"Open the AdaptiqAI console with {command} open."),
        "after_approval": words.get("after_approval", "Your models are ready as soon as you approve."),
        "add_knowledge": words.get("add_knowledge", f"{command} use <name> adds a collection or an earlier "
                                                    "conversation; or ask the agent to find one."),
    }


# Hermes' words: an approval sets the key and the main model, which new chats start on; a chat already open keeps
# the model it has.
WORDS = command_words("Hermes", after_approval="Your models are ready as soon as you approve; new chats then use "
                                               "your workspace's model.")

# Every action, in the order help lists them, with what it does.
ACTIONS = (
    ("", "workspace, models and memory"),
    ("signin", "connect your AdaptiqAI account in the browser"),
    ("private", "stop saving new messages of this conversation"),
    ("remember", "save this conversation to memory again"),
    ("knowledge", "knowledge used in this conversation, and what else you can use"),
    ("use <name>", "use a collection or an earlier conversation here"),
    ("stop <name>", "stop using it here (nothing is deleted)"),
    ("open", "open this conversation in AdaptiqAI Knowledge"),
    ("diagnostics", "a report you can send to support (no secrets)"),
    ("signout", "forget the connection on this computer"),
)

# The actions a command line offers after the command name: every one help lists.
COMMAND_HINT = " · ".join(["status", *(action.split(" ")[0] for action, _ in ACTIONS[1:])])

PRIVATE = ("Private: new messages of this conversation are no longer saved to workspace memory. Existing memory is "
           "kept. {command} remember turns saving back on.")
SAVING = "Saving: new messages of this conversation are saved to workspace memory again."
SIGNED_OUT = "Signed out of AdaptiqAI on this computer. Your saved memory stays in your workspace."
OPEN_FIRST = "Open a conversation first."
# How many collections the knowledge list names besides the ones in use.
OTHERS_SHOWN = 8


def command_help(words: dict = WORDS) -> str:
    """Help: one line per action."""
    return "\n".join(f"{words['command']}{'' if action == '' else f' {action}'} — {does}" for action, does in ACTIONS)


def parse_command(raw: str) -> tuple[str, str]:
    """The action and what follows it in `/adaptiq …`, however Hermes passes it: with or without the command name,
    a leading slash, or a completion glued on with "/". The action is lower case ("" for none); the rest as typed."""
    words = (raw or "").split()
    if words and re.fullmatch(r"/?adaptiq", words[0], re.I):
        words.pop(0)
    first = [part for part in (words.pop(0) if words else "").split("/") if part and part.lower() != "adaptiq"]
    return (first[0].lower() if first else ""), " ".join(words)


def unknown(action: str, words: dict = WORDS) -> str:
    return f"{words['command']} has no \"{action}\". Try one of these:\n{command_help(words)}"


def say_which(action: str, words: dict = WORDS) -> str:
    return f"Say which: {words['command']} {action} <name>. {words['command']} knowledge lists them."


def sign_in_text(url: str, user_code: str, opened: bool, words: dict = WORDS) -> str:
    return (f"{'Your browser opened' if opened else 'Open this link'} to approve {words['app']}:\n{url}\n\n"
            f"Code: {user_code}\n{words['after_approval']}")


def several(name: str, names: list[str], words: dict = WORDS) -> str:
    quoted = ", ".join(f'"{each}"' for each in names[:6])
    return f"Several match \"{name}\": {quoted}. Say the whole name: {words['command']} use <name>."


def knowledge_text(used: list[dict], conversations: list[dict], others: list[str], more: bool,
                   words: dict = WORDS) -> str:
    """What `/adaptiq knowledge` says (core's knowledgeText): the collections and earlier conversations the
    conversation uses, then the ready collections for its model it could use. `others` is None-free: a list that
    could not be read is passed as empty, never shown as "nothing else"."""
    lines = [*(f"• {item['name']} — {item['status']}" for item in used),
             *(f"• {item['name']} — earlier conversation"
               + ("" if item["readiness"] == "ready"
                  else f", {'could not be prepared' if item['readiness'] == 'failed' else 'getting ready'}")
               for item in conversations)]
    text = ["No knowledge is used in this conversation."] if not lines else ["Knowledge in this conversation:", *lines]
    if others:
        text.append(f"Ready to use here: {', '.join(others[:OTHERS_SHOWN])}"
                    f"{', and more' if len(others) > OTHERS_SHOWN or more else ''}.")
    text.append(words["add_knowledge"])
    return "\n".join(text)
