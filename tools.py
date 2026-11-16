"""Agent tools: find and use knowledge, read and look after shared memory, and save knowledge (core's tools.ts and
reading.ts, the DeepSeek Harness plugin's tools: the same names, descriptions, results and refusals).

Every result is a few plain sentences the agent can pass on as they are: names, never ids, JSON or readiness
codes. A refusal is Hermes' tool error, `{"error": <one such sentence>}`. The words for Hermes' own controls
(where the person signs in, adds or stops knowledge) are Hermes'. Agents never delete anything but a collection
they saved themselves in the last 15 minutes; clean-up happens in the console."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from decimal import ROUND_HALF_UP, Context, Decimal

from . import adaptiq_core as core
from . import json_records
from . import knowledge
from . import record
from . import series

TOOLSET = "adaptiq"

# How the agent names Hermes' controls to the person (core's AgentToolWords).
SIGN_IN = "`/adaptiq signin` in a chat (or `hermes adaptiq signin` in a terminal)"
ADD_KNOWLEDGE = "the AdaptiqAI console"
ADD_CONVERSATION = "from this conversation's page in the AdaptiqAI console"
STOP_KNOWLEDGE = "the AdaptiqAI console"
# Starts the idempotency keys of knowledge the agent saves.
KEY_PREFIX = "hermes"

# The person's own messages kept per conversation for the create guard, newest last.
WRITTEN_MESSAGES = 32
WRITTEN_BYTES = 256 * 1024
WRITTEN_SESSIONS = 64
# Words that name knowledge (a collection) in the plugin's languages, not inside another word ("acknowledge").
KNOWLEDGE_WORDS = re.compile(r"(?<![^\W\d_])(?:knowledge|collection|adaptiq|ncm|viden|samling)|知识|知識|资料|資料|集合",
                             re.I)
# A request declined in the same breath: "don't make a collection".
NEGATED = re.compile(r"(?:\b(?:don'?t|do not|never|without|ikke|aldrig|lad være med at)|不要|不用|别)\s+(?:\S+\s+)?$", re.I)
# A yes to the agent's offer to save knowledge, as the whole reply: "yes", "ok", "yes please", "ja tak", "好的", and a
# name for it after ("yes please, call it The book"). Anything else in the reply ("ok wait", "sure, later", "ok never
# mind") is not a yes.
YES_PHRASES = frozenset([
    "yes", "yes please", "yes do it", "yes go ahead", "yeah", "yep", "sure", "sure go ahead", "ok", "okay", "ok please",
    "ok do it", "ok go ahead", "okay do it", "please do", "do it", "go ahead", "ja", "ja tak", "ja gerne", "gerne", "gør det",
    "好", "好的", "是", "是的", "可以", "行", "好的 谢谢",
])
# A no before the knowledge a request names: "yes, but not as a collection".
REFUSED_BEFORE = re.compile(r"(?<![^\W\d_])(?:no|not|don'?t|do not|never|nope|nej|ikke|aldrig)(?![^\W\d_])|不|别", re.I)
# "call it The book", after the yes.
NAMED = re.compile(r"^(?:call it|name it|kald den|kald det|叫它|叫|命名为)\s*(\S.*)$", re.I)
NAMED_AFTER = re.compile(r"[,，:：;；.。]\s*(?:call it|name it|kald den|kald det|叫它|叫|命名为)", re.I)
OPERATION_ID = re.compile(r"[A-Za-z0-9_.:-]{1,100}")
# How long a collection the agent saved can be corrected or taken back from its conversation.
UNDO_WINDOW = 15 * 60
# Collections the agent saved, kept for correcting and taking back: the newest.
MADE_KEPT = 256
# Pages of a search, an inventory, a contents listing or an open document kept for `more`: the newest.
PAGES_KEPT = 256
# How long the collections an earlier conversation reads are kept before they are read again (seconds), and for how
# many conversations: the newest.
THEIRS_FOR = 60
THEIRS_KEPT = 200
# How many reads in a row with nothing new before a read says so (one may be asked again on purpose); how many of a
# conversation's reads are remembered, for how many conversations, and for how long after its last read (seconds).
NOTHING_NEW_AFTER = 2
READS_KEPT = 64
READERS_KEPT = 256
READS_IDLE = 10 * 60
# How many lines open shows at once: a page of text, or a few of the JSON records find points into; how many of them
# come before the line asked for, where its record or paragraph starts.
OPEN_LINES = 120
OPEN_BEFORE = 10
# How many characters open shows at once, and of one line: a longer one (a paragraph is shorter) is cut.
OPEN_CHARS = 16_000
LINE_CHARS = 2_000
# How many other sections open names beside the one it shows (a contract's clauses, a chapter's parts) or under it,
# and how much of a heading it shows where it names one.
BESIDE_SHOWN = 24
HEADING_CHARS = 100
# The source types whose engine text is their original under a heading (a repository's file in a fenced block under
# it), and the largest original open reads: as large as a file the engine takes. At 4 MiB, a log of 10 MiB could be
# read to its 24,000th character and no further.
VERBATIM_TYPES = frozenset({"text", "code", "repository_file"})
ORIGINAL_BYTES = 16 * 1024 * 1024
# How many characters the documents kept for open hold in all: eight of 16 MiB would not fit a harness's memory.
KNOWN_CHARS = 32 * 1024 * 1024
# The source types the engine gives as a table, one row a line (CSV, TSV): their originals hold the true cells.
TABLE_TYPES = frozenset({"spreadsheet"})
# A delimited table by its name, whose columns contents shows; at most COLUMN_READS a page, one request apiece.
DELIMITED_NAME = re.compile(r"\.(csv|tsv)$", re.IGNORECASE)
COLUMN_READS = 12
# How many collections status and a refused read list: the newest, each with its description.
COLLECTIONS_LISTED = 30
# What JavaScript's \s takes for a space, as core's expressions split and match by it: Python's own adds \x1c-\x1f
# and \x85, and leaves out \ufeff. And what its "." takes: any character but a line's end.
_JS_SPACE = "\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
DOT = "[^\n\r\u2028\u2029]"
# A Markdown heading's line: "## Title", its closing hashes aside.
ATX = re.compile(rf" {{0,3}}(#{{1,6}})[ \t]+([^{_JS_SPACE}]{DOT}*?)(?:[ \t]+#+)?[ \t]*\Z")
# A line that underlines the paragraph above it as a heading: "===" for the first level, "---" for the second.
SETEXT = re.compile(r" {0,3}(=+|-+)[ \t]*\Z")
# A line that starts a block whose lines a setext heading cannot underline: a list item, a quote, a table's row or
# rule, a comment. Its later lines belong to it until a blank line, as CommonMark reads a lazy continuation.
BLOCK = re.compile(r" {0,3}(?:[-*+](?:[ \t]|\Z)|[0-9]{1,9}[.)](?:[ \t]|\Z)|>|\||<!--)"
                   r"| {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*\Z")
# A line of YAML front matter: a key, a list item, an indented line, a comment, or nothing; and a key.
YAML_LINE = re.compile(r"(?:[A-Za-z0-9_.-]+[ \t]*:|[ \t]|-[ \t]|#|\Z)")
YAML_KEY = re.compile(r"[A-Za-z0-9_.-]+[ \t]*:")
# File endings of Markdown, and of what the gateway writes as Markdown: a PDF, a Word document, a deck, a workbook, a
# web page.
MARKDOWN_ENDINGS = re.compile(r"(?:md|markdown|mdx|mdown|mkd|mkdn|qmd|rmd)\Z", re.I)
WRITTEN_AS_MARKDOWN = re.compile(r"(?:pdf|docx|pptx|xlsx|html?)\Z", re.I)
# Files named by their kind, with no ending: a container's, make's, a CI's or a package manager's own files.
NOT_MARKDOWN_NAMES = re.compile(r"(?:(?:docker|container|make|gnumake|jenkins|proc|gem|rake|vagrant|brew|pod|just|snake"
                                r"|caddy|berks|guard|cap|earth|tilt)file|codeowners|build|workspace)\Z", re.I)
# The numbers after a rotated log's ending (app.log.1): no ending.
ROTATED = re.compile(r"(?:\.[0-9]+)+\Z")
# Excel's first line naming a CSV's delimiter: "sep=;".
SEP = re.compile(rf"sep=({DOT})\Z")
# A character outside the basic plane: the core counts it as two, as JavaScript does.
ASTRAL = re.compile("[\U00010000-\U0010ffff]")
# A fence's first line, front matter's last, a closing fence, an indented line, a name's ending, and the fence the
# engine puts a repository's file in.
FENCE = re.compile(rf" {{0,3}}(`{{3,}}|~{{3,}})({DOT}*)\Z")
FRONT_END = re.compile(rf"(?:---|\.\.\.)[{_JS_SPACE}]*\Z")
CLOSER = re.compile(rf"`{{3,}}[{_JS_SPACE}]*\Z")
INDENTED = re.compile(r" {4}|\t")
ENDING = re.compile(r"\.([A-Za-z][A-Za-z0-9]{0,7})\Z")
FENCE_OPENED = re.compile(r"(?:^|\n)(`{3,})[^`\n]*\n\Z")
# JSON Lines; the lines of the engine's heading over a repository's file; a JSON text's strings, brackets and line
# ends.
JSON_LINES = re.compile(r"\.(?:jsonl|ndjson)\Z", re.I)
SOURCE_LINE = re.compile(r"(?:Source|Path|Language): ")
JSON_TOKEN = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[{}\[\]:,\n]')
# How many documents' lines are kept for the next open or find: the most recent ones.
DOCUMENTS_KEPT = 16
# How many documents one find lists together: a year's months, a set of volumes.
LISTED_DOCUMENTS = 12
# Small documents one find lists together, up to this many characters in all: a year's letters, one a file. With
# twelve at most, an agent grouping 300 letters by subject was sent to the engine's search, then named twelve at a
# time: 37 finds and four minutes for one count.
LISTED_SMALL = 1_000
LISTED_CHARS = 1_000_000
# How many documents find reads at once: each read took a quarter of a second (2026-11-08), sixteen at once about 60 a
# second, 300 letters in five seconds. The gateway allows a workspace's keys 6,000 reads a minute.
READ_AT_ONCE = 16
# The documents of finds over many are kept up to a find's worth of characters, so the next page or find over them
# reads none again. Kept by count with the rest (sixteen), twenty documents of 8,001 characters were read again on
# every page: all twenty in core, four here (core's MANY_KEPT_CHARS).
MANY_KEPT_CHARS = LISTED_CHARS
# A collection of up to this many documents is searched a document at a time when the engine's first lines are cut
# off, to name every document that holds the words: asked for an employee's name, an agent saw 25 lines from four
# documents, never the directory that held the name, and said the collection had no names.
PROBE_DOCUMENTS = 40
# How many of those documents a note shows the first line of.
HOLDER_LINES = 6
# How many records a sorted find shows at once: sorting is for the first few ("the five latest"); how many groups a
# find shows a page.
SORTED_PAGE = 20
GROUPS_PAGE = 60

# About what one reply reads of a large collection (the engine's working set).
READ_PER_REPLY = 130_000


def format_tokens(value) -> str:
    """A token count, short (core's formatTokens): 950, 1.2k, 130k, 2.29M."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return "0"
    if value < 1000:
        return str(math.floor(value + 0.5))
    if value < 1_000_000:
        return _trim(value / 1000, 1 if value < 100_000 else 0) + "k"
    return _trim(value / 1_000_000, 2 if value < 10_000_000 else 1) + "M"


def _trim(value: float, digits: int) -> str:
    # Rounded as core's toFixed rounds, a value halfway up: Python's format rounds it to even (274,500 tokens read
    # 274k where core says 275k).
    fixed = str(Decimal(value).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))
    return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed


# What the knowledge tool says first. Asked about lot ids in a collection its conversation read, an agent called
# search, which matches collection names only, found nothing, and asked the person for their local files.
FIND_FIRST = ("To look anything up in the collections used here (an id, a name, a number, a reference, a fact), use "
              "find: it searches their text for words and gives the exact lines as document:line. search finds "
              "collections by words in their names only, never their contents, so a search that finds nothing says "
              "nothing about what a collection holds.")
QUERY_WORDS = ("For search: words in collection names, never their contents. For find: words, ids or names in the text "
               "of the collections used here (a phrase, else all its words on one line; any case).")
READING_NOTE = ("A collection used here is in your memory, but a large one is read in parts: each reply reads what "
                f"matches the conversation (about {format_tokens(READ_PER_REPLY)} tokens of it), never all of it at "
                "once. contents lists its documents; open shows a document's exact lines (around a line, if given; "
                "more: true reads on). Before you say something is not there, find it; for an overview, take contents "
                "first. Answer from what these actions return, citing document:line as a result shows it, and never "
                "name a document or a line no result showed.")
# How the agent speaks about knowledge: said once, in the knowledge tool, as every request carries all three. Hermes
# shows no readiness itself.
SPEAK = ("Speak to the person in their own language, only about what matters to them; never mention ids, JSON, tool "
         "names, status checks or readiness codes.")
TURN_NOTE = "AdaptiqAI Knowledge this conversation reads: "
# How find ends (core's reading.ts): the lines it gave, or plainly that it found none.
FOUND = "These are the exact lines. To read a whole document, open it."
# Said after a find that left out a document's Columns line, as the one above holds for it.
COLUMNS_ABOVE = "Rows of a document with no Columns line of its own have the columns last shown above them."
NOTHING_FOUND = ("Nothing was found, so there are no lines to answer from. Try other or fewer words, or take contents to "
                 "see the documents.")
# What the knowledge tool says of find_in_conversation, right after FIND_FIRST (core's FIND_IN_CONVERSATION, plugins
# #121): Hermes keeps every message of a conversation, so the plugin reads what the person wrote in it.
FIND_IN_CONVERSATION = (
    "find_in_conversation: every line the person wrote in this conversation that holds the words (every turn, as the "
    "harness holds them, a line a record), page by page; fields, sort, sum and group work as in find. Use it to "
    "count, add up or list what was said or noted in this conversation, or, with conversation, in one added here (by "
    'its name as status shows it) or in "all" of them and this one together, as one list: a long conversation is '
    "read in parts, and this shows every line.")
# find_in_conversation's conversation (core's CONVERSATION_PARAMETER), and what status says of the ones added here
# that are ready (core's ADDED_READ): a long one is in the model's context in parts, so a detail or a count is found.
CONVERSATION_PARAMETER = ("For find_in_conversation: a conversation added here, by its name as status shows it, to find "
                          'in it instead of this one; or "all" for this conversation and every one added here '
                          "together, as one list. Leave it out for this conversation.")
ADDED_READ = ("They are read-only memory in your context, a long one in parts: answer from them directly, and to find a "
              "detail in them, or count, add up or list what they hold, use find_in_conversation with conversation set "
              'to one\'s name, or "all" for them and this one together.')
# What find says first when it is given a conversation (core's CONVERSATION_NOT_FOR_FIND): sent with find, it was
# passed over in silence, and the collections' "nothing" read as the conversations' (MiMo eval, 2026-11-15).
CONVERSATION_NOT_FOR_FIND = ("conversation is for find_in_conversation, not find: this found in the collections alone. "
                             "To find in the conversations added here, use find_in_conversation with conversation.")


def unsearched_note(names: list[str]) -> list[str]:
    """What a find in this conversation alone says of the conversations added here, by their names (core's
    unsearchedNote): its line, or no line when none is added here."""
    if not names:
        return []
    one = len(names) == 1
    which = "the conversation" if one else f"the {len(names)} conversations"
    listed = ", ".join(f'"{name}"' for name in names)
    their = "its" if one else "one's"
    return [f"Not searched: {which} added here, {listed}. To find in {'it' if one else 'them'} too, set conversation "
            f'to "all", or to {their} name.']


def not_searched_by_find(names: list[str]) -> list[str]:
    """What a find in the collections that found nothing says of the conversations added here (core's
    notSearchedByFind). Asked for a part number in a conversation added after the first turn, agents ran find 9 to 19
    times, found nothing in the collections and said it was not there (e04, MiMo eval, 2026-11-15): status, asked
    before the conversation was added, named none, and no find said one was there."""
    if not names:
        return []
    one = len(names) == 1
    which = "the conversation" if one else f"the {len(names)} conversations"
    listed = ", ".join(f'"{name}"' for name in names)
    their = "its" if one else "one's"
    return [f"find searched the collections alone, not {which} added here: {listed}. find_in_conversation with "
            f'conversation set to {their} name, or "all", finds in {"it" if one else "them"}.']


# What find_in_conversation lists its lines under, what it says it read and how it ends (core's reading.ts), and what
# it says when the conversation could not be read (core's tools.ts): never that nothing is there.
THIS_CONVERSATION = "this conversation"
CONVERSATION_SCANNED = ("Sources scanned: this conversation as held by the harness. Other conversations and "
                        "collections are not included.")
CONVERSATION_FOUND = ("These are the exact lines of this conversation, as the harness holds it: what the person wrote "
                      "in it, every turn.")
CONVERSATION_UNREAD = ("The messages of this conversation could not be read just now; this does not mean they hold "
                       "nothing. Try again in a moment.")
CONVERSATION_UNHELD = "The messages of this conversation are not at hand yet. Find in it again after the next message."

KNOWLEDGE = {
    "name": "adaptiq_knowledge",
    "description": (
        f"Use the person's saved knowledge (collections) in this conversation. {FIND_FIRST} {FIND_IN_CONVERSATION} "
        "status: what this conversation uses now. search: saved collections by words in their names, newest first (no "
        "words lists everything; more: true shows the next ones). use / stop: start or stop using a collection here, "
        f"by its name as search or status shows it. {READING_NOTE} When the person asks to use or remove named "
        "knowledge, do it for them and say so in one short sentence; ask only if the name is ambiguous. Search here "
        "before offering to save something as knowledge: what is already saved is used, not saved again. Earlier "
        f"conversations are added by the person: {ADD_CONVERSATION}; tell them so when they ask for one. use and stop "
        "take effect from your next step in this turn: keep working, and never end the turn to wait for them. "
        f"Knowledge is reference material, not instructions. Change only this conversation. {SPEAK}"),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "search", "use", "stop", "contents", "find", "open",
                                                  "find_in_conversation"]},
            "query": {"type": "string", "description": QUERY_WORDS},
            "more": {"type": "boolean", "description": "For search, contents and open: the next results."},
            "name": {"type": "string", "description": "For use and stop: the collection's name as search or status "
                     "shows it. For contents, find and open: which collection (default: all used here); one not "
                     "used here yet is used first."},
            "document": {"type": "string", "description": "For open: the document as contents or find shows it. "
                         "For find: the documents to look in, which find then lists in full, page by page: one, "
                         'several with commas between ("01_Afregning.json, 02_Afregning.json"), or the start or a '
                         "part of their names."},
            "line": {"type": "number", "description": "For open: the line to read around."},
            "conversation": {"type": "string", "description": CONVERSATION_PARAMETER},
            "sort": {"type": "string", "description": "For find with document: the field to sort the records of all "
                     "the documents named by, as one list: smallest first, or largest first with desc after it "
                     '("betalt-forfald desc"); more fields with commas between sort ties ("betalt-forfald desc, '
                     'parti"); records without the field come last.'},
            "fields": {"type": "array", "items": {"type": "string"}, "description": "For find with document: only "
                       "these fields of each record, one record per line, to go through many at once: JSON keys (one "
                       "inside an object by its path, detaljer.pris_kr), the words before a colon in a text record, a "
                       "CSV or TSV table's columns, or the words a number "
                       'stands next to in a sentence (rejected units in "21 rejected units"). Two names with a minus '
                       "between (paid-due) give the days from one date to the other, the minutes from one time to the "
                       "other (a date with a time of day), or one number less the other."},
            "sum": {"type": "array", "items": {"type": "string"}, "description": "For find with document: fields to "
                    "add up over every record found, on every page (amounts, lengths, quantities, or days between two "
                    "dates written as in fields): exact totals, instead of adding the numbers yourself. For a count or "
                    "a total alone, set summary: true."},
            "summary": {"type": "boolean", "description": "For find with document: true when only a count or a total "
                        "is asked. Reads every selected document and gives the count, the sums asked for, the values "
                        "left out and the sources, without the records: every match is covered, so there are no pages "
                        "to read and nothing to add up again per document. With group, the groups still come page by "
                        "page. Complete only for the documents and words given; missing or non-numeric values are "
                        "said."},
            "group": {"type": "string", "description": "For find with document: instead of the records, one line for "
                      "each value of this field (a customer, a route, a status), with how many records have it and "
                      "the sums asked for, largest sum first, or most records first without sum (sort: \"count\", or "
                      "a sum field, with desc or asc). Answers which customers, routes or types have the most of "
                      "something in one call."},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

MEMORY = {
    "name": "adaptiq_memory",
    "description": (
        "Help the person keep working with their AdaptiqAI memory. status: how full the workspace memory is. "
        "inventory: their saved collections with size and date (sizes are not guaranteed savings; age is not proof "
        "of disuse). review: opens Knowledge in the browser so the person can tidy up; never delete anything "
        "yourself. continue_without_saving: only on the person's explicit choice; what is saved stays, and new "
        "messages of this conversation stop being saved. A growing conversation needs no action: idle conversations "
        "rest by themselves to make room and nothing is deleted. Only a full library or a memory-full message asks "
        "for action: then offer concrete options (tidy up in Knowledge; for a memory-full message also continue "
        "without saving, which makes no room for knowledge; more memory only when they ask). Starting a new "
        "conversation or compacting does not free memory. Answer in one short sentence."),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "inventory", "review", "continue_without_saving"]},
            "more": {"type": "boolean", "description": "For inventory: the next collections."},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

CREATE = {
    "name": "adaptiq_create_knowledge",
    "description": (
        "Save material as knowledge: a named collection from text, local files, or HTTPS links (GitHub repositories "
        "too), which this and later conversations can use. Offer it, in one short question with a proposed name, "
        "after reading a sizeable set of files or documents the person will keep working with (a book, a project's "
        "documents), or when they say remember, save, collection or NCM about such material; search with "
        "adaptiq_knowledge first so nothing is saved twice. Create only when the person asks for it or says yes to "
        "your offer, and quote their words in user_request (required for create): their request, or their yes. The "
        "agent talking to the person creates it, not a subagent. The conversation itself is kept in memory "
        "automatically: a fact said here, \"remember this\", or a mentioned conversation needs no collection; say so "
        "instead. Never turn guesses into facts, save credentials, or copy unrelated private content. Give each "
        "collection its own operation_id and keep it. To fix a mistake in a collection you just saved (wrong or "
        "missing text, a typo in the name), call create again with the same operation_id and the corrected content: "
        "the same collection is corrected, no second one is made. undo takes back a collection you saved in this "
        "conversation in the last 15 minutes, when the person asks; nothing else can be removed. Confirm in one short "
        "sentence, e.g. Saved as knowledge: \"The book\" — it is used in this conversation. When creating is "
        "refused, pass on the reason as given; never send the person to the console unless the reason says so."),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "undo"]},
            "operation_id": {"type": "string", "description": "Your own short label for this collection (letters, "
                             "digits, . _ : -). Reuse it to retry or correct it. Never show it."},
            "user_request": {"type": "string", "description": "Required for create: the person's own words in this "
                             "conversation asking to save knowledge or a collection, or their yes to your offer, "
                             "quoted exactly."},
            "name": {"type": "string", "description": "Collection name, in the person's language."},
            "description": {"type": "string"},
            "use_here": {"type": "boolean", "description": "Use in this conversation once ready. Defaults to true."},
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type"],
                    "properties": {
                        "type": {"type": "string", "enum": ["text", "file", "link"]},
                        "name": {"type": "string"},
                        "content": {"type": "string", "description": "For text sources."},
                        "path": {"type": "string", "description": "For file sources: absolute, or relative to the "
                                 "working directory."},
                        "url": {"type": "string", "description": "For link sources: an HTTPS URL."},
                    },
                },
            },
        },
        "required": ["action", "operation_id"],
        "additionalProperties": False,
    },
}

ASK_FIRST = ("Create knowledge only when the person asks for it in this conversation or says yes to your offer: quote "
             "their request (words that ask to save knowledge or a collection) or their yes in user_request. If you "
             "have not offered yet, offer first. A subagent cannot: the agent talking to the person creates it. "
             "Conversations are kept in memory automatically; nothing else is needed to remember something said here.")


def _plain(text: str) -> str:
    """Text compared as written: case, quotes and runs of whitespace aside."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’«»`]", "", text)).strip()


def same_name(name: str) -> str:
    """A name compared the way a person reads it: case, spacing and punctuation aside."""
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", name).lower()).strip()


def _is_yes(reply: str) -> bool:
    """Whether a reply (as `_plain` leaves it) is a yes and nothing else, a name for it aside."""
    def words(said: str) -> str:
        return re.sub(r"[\s,，、!！.。:：;；~-]+", " ", said.lower()).strip()

    at = NAMED_AFTER.search(reply)
    if at is None:
        return words(reply) in YES_PHRASES
    name = re.sub(r"^[,，:：;；.。]\s*", "", reply[at.start():])
    return words(reply[:at.start()]) in YES_PHRASES and NAMED.search(name) is not None


def _keep(table: OrderedDict, key: str, value, kept: int) -> None:
    """Keep `value` as the newest entry of `table`, dropping the oldest beyond `kept`."""
    table.pop(key, None)
    table[key] = value
    while len(table) > kept:
        table.popitem(last=False)


class Written:
    """What the person wrote in each recent conversation, for the create guard (core's noteUserTexts and
    userAskedForKnowledge). A turn is `{"text", "after_offer"}`, marked when it answers the agent's offer to save
    knowledge (turns.user_turns); a plain string is a turn that answers nothing."""

    def __init__(self) -> None:
        self._turns: OrderedDict[str, list[dict]] = OrderedDict()
        # What each conversation's current turn started with, for the answers given during it.
        self._start: OrderedDict[str, list[dict]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _kept(turns: list) -> list[dict]:
        kept: list[dict] = []
        size = 0
        for turn in reversed(turns[-WRITTEN_MESSAGES:]):
            text = turn if isinstance(turn, str) else str(turn.get("text") or "")
            size += len(text.encode())
            if size > WRITTEN_BYTES and kept:  # the newest message always counts, however long
                break
            kept.insert(0, {"text": _plain(text), "after_offer": isinstance(turn, dict) and turn.get("after_offer") is True})
        return kept

    def note(self, session_id: str, turns: list) -> None:
        """At the start of a turn: the person's turns in the conversation so far, oldest first."""
        kept = self._kept(list(turns))
        with self._lock:
            _keep(self._start, session_id, kept, WRITTEN_SESSIONS)
            _keep(self._turns, session_id, kept, WRITTEN_SESSIONS)
        FAILURES.turn(session_id)

    def answered(self, session_id: str, turns: list) -> None:
        """During a turn: the person's answers since it started, to questions the agent asked through clarify."""
        with self._lock:
            start = self._start.get(session_id)
        if start is None:  # a subagent's, or a turn this process never saw start
            return
        kept = self._kept(start + list(turns))
        with self._lock:
            _keep(self._turns, session_id, kept, WRITTEN_SESSIONS)

    def asked_for_knowledge(self, session_id: str, quote: str) -> bool:
        """Whether the person asked for knowledge in their own words: `quote` is a passage of one of their
        messages here that names knowledge or a collection, and the message does not decline it right before
        ("don't make a collection"); or it is their yes to the agent's offer, as their latest message."""
        wanted = _plain(quote)
        with self._lock:
            turns = list(self._turns.get(session_id, []))
        if not wanted:
            return False
        # A yes is their request only as their latest message, and only as a yes and nothing else: a later
        # "actually no", or "ok wait", takes it back.
        if _is_yes(wanted):
            return bool(turns) and turns[-1]["after_offer"] and turns[-1]["text"] == wanted
        named = KNOWLEDGE_WORDS.search(wanted)
        if len(wanted) < 6 or named is None:
            return False
        # Declined before it names knowledge, in the quote itself: "yes, but not as a collection".
        if REFUSED_BEFORE.search(wanted[:named.start()]):
            return False
        for turn in turns:
            text = turn["text"]
            at = text.find(wanted)
            if at >= 0 and not NEGATED.search(text[max(0, at - 48):at]):
                return True
        return False


# What a tool does when AdaptiqAI can't answer it (core's tool-failures.ts). A model request is retried by Hermes; a
# tool's failure goes to the model, and told "try again", an agent called again and again, each step sending the whole
# conversation. So passing trouble is asked again inside the call (knowledge.calling), and trouble that lasts says what
# failed, that this is no answer about the knowledge, and to tell the person rather than call again in this turn. After
# the same action failed so twice in one of the person's turns, the tool says so without asking AdaptiqAI. A refusal
# keeps its own words and never counts.
STOP = "Do not call it again in this turn: "
AFTER = {
    "adaptiq_knowledge": "tell the person their knowledge could not be read right now, and that they can ask again shortly.",
    "adaptiq_memory": "tell the person their AdaptiqAI memory could not be read right now, and that they can ask again "
                      "shortly.",
    "adaptiq_create_knowledge": "tell the person it could not be saved right now, and that they can ask again shortly.",
}
FAILED_TWICE = "This failed twice already in this turn, so AdaptiqAI was not asked again."
NOTHING_MISSING = "This does not mean anything is missing."
FAILURES_PER_TURN = 2
# Without a known turn (a subagent's, or one this process never saw start), failures count as one turn this long
# after the last; whatever the turn, AdaptiqAI is asked again this long after the last failure.
UNKNOWN_TURN = 2 * 60
FAILURES_KEPT = 10 * 60
FAILURE_SESSIONS = 64


def stop_words(tool: str) -> str:
    """The end of a tool's failure: not to call again in this turn, and what to tell the person (core's stopWords)."""
    return STOP + AFTER.get(tool, AFTER["adaptiq_knowledge"])


class _NotUsedHere(knowledge.KnowledgeError):
    """A read the conversation has no collection for: the one named is not used here, or none is (core's
    NotUsedHere). The knowledge tool uses the one named, or names the collections there are; the message is what the
    reading alone says. `asked` is the name as the agent gave it, or None; `none` is true when the conversation reads
    no ready collection, nor do the conversations added here."""

    def __init__(self, message: str, asked: str | None, none: bool) -> None:
        super().__init__(message)
        self.asked = asked
        self.none = none


class _OutOfReach(knowledge.KnowledgeError):
    """What a tool could not read, said for the agent; the guard adds what the agent does (core's ToolOutOfReach)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, passing=True)


def _passing(error: knowledge.KnowledgeError) -> bool:
    """Passing trouble at AdaptiqAI rather than an answer; a full memory (507) is an answer."""
    return isinstance(error, _OutOfReach) or (error.passing and error.status != 507)


def trouble_words(error: knowledge.KnowledgeError) -> str:
    """What failed, for passing trouble that lasted through the call's tries (core's troubleWords)."""
    offline = any(code in error.answer for code in knowledge.OFFLINE_CODES)
    if error.status is None:
        return "AdaptiqAI cannot be reached right now."
    if knowledge.getting_ready(error.status, error.answer):
        return "AdaptiqAI is still getting this conversation ready."
    if offline:
        return "The model this conversation runs on is offline right now."
    if error.status == 429:
        return "AdaptiqAI is busy right now."
    if error.status == 408:
        return "AdaptiqAI took too long to answer."
    return "AdaptiqAI did not answer just now."


class Failures:
    """Each tool call's tries, and each conversation's failed actions in the person's current turn (core's
    ToolFailures). Hermes says when a turn starts (pre_llm_call, through Written.note); a conversation it never did for
    counts by time alone."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep) -> None:
        self._clock = clock
        self._sleep = sleep
        self._turns: OrderedDict[str, int] = OrderedDict()
        self._failed: OrderedDict[str, dict] = OrderedDict()
        self._started = 0
        self._lock = threading.Lock()

    def turn(self, session_id: str) -> None:
        """A turn of the person's starts in a conversation."""
        with self._lock:
            self._started += 1
            _keep(self._turns, session_id, self._started, FAILURE_SESSIONS)

    def _current(self, session_id: str, now: float) -> dict | None:
        record = self._failed.get(session_id)
        if record is None:
            return None
        turn, age = self._turns.get(session_id), now - record["at"]
        known = record["turn"] is not None and turn is not None
        if age <= FAILURES_KEPT and (record["turn"] == turn if known else age <= UNKNOWN_TURN):
            return record
        self._failed.pop(session_id, None)
        return None

    def run(self, session_id: str, tool: str, action: str, work):
        """One tool call: answered here when this action failed twice in this turn; else the call, with passing
        trouble asked again inside it, and trouble that lasted said for the agent and counted."""
        key = f"{tool} {action}"
        with self._lock:
            record = self._current(session_id, self._clock()) if session_id else None
            stopped = record is not None and record["counts"].get(key, 0) >= FAILURES_PER_TURN
        if stopped:
            raise _OutOfReach(f"{FAILED_TWICE} {stop_words(tool)}")
        try:
            with knowledge.calling(self._clock, self._sleep):
                result = work()
        except knowledge.KnowledgeError as error:
            if not _passing(error):
                raise
            if session_id:
                with self._lock:
                    now = self._clock()
                    record = self._current(session_id, now) or {"turn": self._turns.get(session_id), "counts": {}}
                    record["at"] = now
                    record["counts"][key] = record["counts"].get(key, 0) + 1
                    _keep(self._failed, session_id, record, FAILURE_SESSIONS)
            said = str(error) if isinstance(error, _OutOfReach) else f"{trouble_words(error)} {NOTHING_MISSING}"
            raise _OutOfReach(f"{said} {stop_words(tool)}") from error
        if session_id:
            with self._lock:
                record = self._failed.get(session_id)
                if record is not None:
                    record["counts"].pop(key, None)
        return result


FAILURES = Failures()
# The call a guarded handler is running in this thread: its conversation, tool and action, for `_run`.
_calling = threading.local()


def _guarded(name: str):
    """A tool handler whose calls `_run` guards by their action."""
    def wrap(tool):
        def handler(args: dict, *rest, **named) -> str:
            outer = getattr(_calling, "call", None)
            action = args.get("action") if isinstance(args, dict) else None
            _calling.call = (str(named.get("session_id", rest[0] if rest else "") or ""), name, str(action))
            try:
                return tool(args, *rest, **named)
            finally:
                _calling.call = outer
        return handler
    return wrap


def _cwd(task_id: str) -> str:
    """The conversation's working directory, as Hermes' own file tools resolve it."""
    try:
        from tools.terminal_tool import get_session_cwd
        return get_session_cwd(task_id) or os.getcwd()
    except Exception:
        return os.getcwd()


def _container(task_id: str) -> str | None:
    """The terminal backend when the agent's paths are inside a container, so not on this computer.

    Inside Hermes, a backend that cannot be told counts as a container: a file is then refused
    rather than read from this computer under a path that meant another one.
    """
    try:
        import tools.file_tools_paths as paths
    except Exception:  # not inside Hermes
        return None
    try:
        if hasattr(paths, "_uses_container_paths"):
            return "container" if paths._uses_container_paths(task_id) else None
        if hasattr(paths, "container_backend_for_task"):
            return paths.container_backend_for_task(task_id)
    except Exception:
        pass
    return "unknown"


def _resolver(task_id: str):
    """How Hermes' own file tools turn a path into an absolute one; relative to the session's
    directory outside Hermes."""
    try:
        from tools.file_tools_paths import _resolve_path_for_task
    except Exception:
        cwd = _cwd(task_id)
        return lambda raw: os.path.join(cwd, os.path.expanduser(raw))
    return lambda raw: _resolve_path_for_task(raw, task_id)


def _blocked():
    """Hermes' own read denylist (credential stores, .env files, skill caches), where there is one."""
    try:
        from agent.file_safety import get_read_block_error
    except Exception:
        return None
    return get_read_block_error


def available() -> bool:
    """The tools are offered while AdaptiqAI is connected and Hermes has an AdaptiqAI provider entry, whatever the
    main provider is. Hermes gives a conversation its tools when it starts and keeps them, so a chat moved onto
    AdaptiqAI with /model has them only if they were offered from the start; in a chat on another provider they
    say so (`elsewhere`)."""
    from hermes_cli.config import load_config

    if core.key() is None:
        return False
    cfg = load_config()
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    return (str(model.get("base_url") or "").rstrip("/") == core.API
            or core._entry_name(providers, model.get("provider")) is not None)


ELSEWHERE = ("This conversation runs on another provider, and AdaptiqAI's knowledge and memory work in a "
             "conversation on an AdaptiqAI model. The person can switch with /model.")


def _quoted(name: str) -> str:
    return f'"{name}"'


def _names(items: list[dict]) -> str:
    return ", ".join(_quoted(item["name"]) for item in items)


def _state(item: dict) -> str:
    """A collection's state in plain words."""
    return {"ready": "ready", "preparing": "getting ready; used by itself when ready",
            "failed": "could not be prepared"}.get(item.get("readiness"), "not prepared")


def build_failure(error: str) -> str:
    """Why a collection could not be built, in plain words, and what the person can do about it when there is
    something (core's buildFailure). A full workspace was "is could not be prepared; it can be used once it is
    ready" before: it never would be, and nothing said why."""
    full = re.search(r"knowledge capacity exceeded: ([0-9]+) used \+ ([0-9]+) requested > ([0-9]+)", error)
    if full is None:
        return re.sub(r"^knowledge compile failed: ", "", error)
    used, asked, limit = (int(value) for value in full.groups())
    return (f"the workspace's knowledge is full ({format_tokens(used)} of {format_tokens(limit)} taken; it needs "
            f"{format_tokens(asked)}). To make room, the person can delete a collection no longer needed from "
            f"{ADD_KNOWLEDGE}, then add this one again")


def _count(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return value


def _literal(text: str) -> str:
    """A text as a regular expression that matches it literally."""
    return re.sub(r"[.*+?^${}()|\[\]\\]", r"\\\g<0>", text)


def ready_here(used: list[dict]) -> list[dict]:
    """The collections a conversation reads that are ready."""
    return [item for item in used if item["kind"] == "collection" and item["readiness"] == "ready"]


def turn_note(used: list[dict]) -> str | None:
    """What a turn is told about the collections its conversation reads (core's turnNote). Nothing in a turn said
    a collection was in use, so an agent asked about an id searched past chats and the person's files instead."""
    ready = ready_here(used)
    if not ready:
        return None
    return (f"{TURN_NOTE}{_names(ready)}. To answer from it, look the facts up with adaptiq_knowledge, action find "
            "(words, ids or names; it returns the exact lines as document:line), and cite what it returns. search "
            "only finds collections by their names.")


def search_miss(query: str, used: list[dict]) -> str:
    """What a search that finds no name says: that it read names only, and, when the conversation uses ready
    collections, that find searches their text (core's searchMiss)."""
    ready = ready_here(used)
    missed = f"No collection has {_quoted(query)} in its name"
    if not ready:
        return f"{missed}. search reads names only, never what a collection holds."
    return (f"{missed}: search reads names only, never what a collection holds. This conversation uses "
            f"{_names(ready)}: find {_quoted(query)} searches {'its' if len(ready) == 1 else 'their'} text.")


def their_collections(names: list[str]) -> str:
    """What the agent is told about collections that earlier conversations added here read and this one does not
    (core's theirCollections)."""
    return (f"The earlier conversations added here read {', '.join(_quoted(name) for name in names)}, which this "
            f"conversation does not: to look something up in {'it' if len(names) == 1 else 'one of them'}, use it "
            "here (action use, with its name), then find in it.")


def document_lines(start: dict, source: dict, original: str | None) -> dict:
    """A document's lines, numbered as the engine's search numbers them: from the heading the engine gives every
    document ("# name", "", "Source type: …", ""). The engine's text of a text or code source is its original under
    that heading, so with the original all of its lines are known, when the start of the engine's text and its size
    say the original is what it holds. Otherwise (a PDF, a page, no original) only the start the engine shows is
    (core's documentLines). `start` is {text, whole}; `source` has its `bytes` and `compile_bytes`."""
    def lines_in(text: str) -> list[str]:
        # A last line break ends the last line, it starts none.
        return (text[:-1] if text.endswith("\n") else text).split("\n")

    text = start["text"]
    if start["whole"]:
        return {"lines": lines_in(text), "whole": True}
    if original is not None and source.get("bytes") is not None and source.get("compile_bytes") is not None:
        at = text.find(original[:200])
        heading = text[:max(0, at)]
        # A repository's file stands in a fenced block under its heading, which the engine closes after it.
        fence = FENCE_OPENED.search(heading)
        close = "" if fence is None else "\n" + fence.group(1)
        # The engine's text drops the line breaks the original ends with: one, or more after an empty last line,
        # or \r\n. Its size then lies between the two.
        kept = original.rstrip("\r\n")
        engine_size = source["compile_bytes"] - len(heading.encode("utf-8")) - len(close.encode("utf-8"))
        # The start may end inside a character, so its last one is left out.
        if (at >= 0 and original.startswith(text[at:-1])
                and len(kept.encode("utf-8")) <= engine_size <= source["bytes"] + 1):
            return {"lines": lines_in(heading + kept + close), "whole": True}
    # A start that is not all of the text may end inside a line.
    return {"lines": text.split("\n")[:-1], "whole": False}


def columns_line(table: dict) -> str:
    """A table's columns, for its rows shown without their header row."""
    return f"Columns (line {table['line']}): {'; '.join(table['cells'])}"


def _wide(text: str) -> bytes | None:
    """A text as UTF-16, the units the core counts its characters in, where that differs from Python's: None for a
    text with no character outside the basic plane."""
    return None if text.isascii() or not ASTRAL.search(text) else text.encode("utf-16-le", "surrogatepass")


def _units(text: str) -> int:
    """A text's length as the core counts it: a character outside the basic plane is two."""
    wide = _wide(text)
    return len(text) if wide is None else len(wide) // 2


def _high(text: str, wide: bytes | None, at: int) -> bool:
    """Whether the core's character `at` of a text is the first half of one outside the basic plane."""
    unit = ord(text[at]) if wide is None else int.from_bytes(wide[2 * at:2 * at + 2], "little")
    return 0xD800 <= unit <= 0xDBFF


def cells_of(row: str, delimiter: str) -> list[str]:
    """A row of a table cut into its cells at the delimiter, a quoted cell as CSV writes it (a delimiter inside, a
    quote doubled) (core's cellsOf). The cells are trimmed, as the core's trim() trims them."""
    if delimiter == "|":
        row = row.strip(_JS_TRIMMED)
        row = row[1:] if row.startswith("|") else row
        row = row[:-1] if row.endswith("|") else row
    cells, cell, quoted, at = [], "", False, 0
    while at < len(row):
        char = row[at]
        if quoted:
            if char != '"':
                cell += char
            elif row[at + 1:at + 2] == '"':
                cell += '"'
                at += 1
            else:
                quoted = False
        elif char == '"' and cell.strip(_JS_TRIMMED) == "":
            quoted, cell = True, ""
        elif char == delimiter:
            cells.append(cell)
            cell = ""
        else:
            cell += char
        at += 1
    cells.append(cell)
    return [value.strip(_JS_TRIMMED) for value in cells]


def columns_in(start: str) -> list[str]:
    """A delimited table's columns, from the start of the engine's text: heading lines, then the header row, cut at
    commas (" | ") as the engine cuts every row (core's columnsIn). None when the first row is not a header of two
    or more."""
    for raw in start.split("\n")[:12]:
        line = raw.strip(_JS_TRIMMED)
        if not line or line.startswith("#") or line.startswith("Source type:"):
            continue
        row = line.replace(" | ", ",")
        cells = cells_of(row, _delimiter_of(row))
        return cells if len(cells) >= 2 and all(cells) else []
    return []


def _delimiter_of(header: str) -> str:
    """The delimiter of a table's header row: the one of semicolon, tab and comma it holds most of."""
    best = ";"
    for delimiter in ("\t", ","):
        if header.count(delimiter) > header.count(best):
            best = delimiter
    return best


def table_lines(start: dict, original: str | None, engine_lines: int | None) -> dict:
    """A table's lines (a CSV or TSV document), numbered as the engine's search numbers them (core's tableLines).
    The engine gives a table one row a line under its heading, with " | " where it cut the row into cells, and it
    cuts at commas: a semicolon row's decimal commas become cell breaks ("1.150,00" reads "1.150 | 00"), and open
    showed only its first 24,000 characters. The original holds the true rows. It is taken when it has as many rows
    as the engine's text and each row the start shows is its row, cut differently; its header row is then `table`
    ({line, cells, delimiter}). `engine_lines` is how many lines the engine's text has, as its documents list says."""
    text = start["text"]
    shown = (text[:-1] if text.endswith("\n") else text).split("\n")
    engine = {"lines": shown, "whole": True} if start["whole"] else {"lines": text.split("\n")[:-1], "whole": False}
    if original is None:
        return engine
    rows = re.split(r"\r?\n", original.removeprefix("\ufeff").rstrip("\r\n"))

    def bare(line: str) -> str:
        # A row without what the engine changes: where it cuts, and the quotes it takes off.
        return re.sub(rf'[{_JS_SPACE}|,;"]', "", line)

    at = next((index for index, line in enumerate(shown) if bare(line) and bare(line) == bare(rows[0])), -1)
    if at < 0 or (len(shown) if start["whole"] else engine_lines) != at + len(rows):
        return engine
    # A start that is not all of the text may end inside a row, so its last line is left out.
    complete = min(len(rows), (len(shown) if start["whole"] else len(shown) - 1) - at)
    if any(bare(shown[at + row]) != bare(rows[row]) for row in range(1, complete)):
        return engine
    # Excel writes "sep=;" above the header row to name the delimiter: the columns are on the next line.
    sep = SEP.match(rows[0].strip(_JS_TRIMMED))
    head = 1 if sep and len(rows) > 1 else 0
    delimiter = sep.group(1) if sep else _delimiter_of(rows[head])
    return {"lines": shown[:at] + rows, "whole": True,
            "table": {"line": at + 1 + head, "cells": cells_of(rows[head], delimiter), "delimiter": delimiter}}


def table_rows_of(read: dict) -> dict[int, dict]:
    """A document's Markdown tables, by its lines, found once."""
    if "rows" not in read:
        read["rows"] = markdown_rows(read["lines"])
    return read["rows"]


def _table_number(value: str, delimiter: str) -> float | None:
    """A number as a table's cell writes it (core's tableNumber): "1.150,00" or "80.913" where a semicolon cuts the
    rows (a comma marks the decimals there), "1,150.00" where a comma or a tab does. With a point and a comma, the
    last one marks the decimals. None when the cell is not one."""
    text = re.sub(r"[\s\u00a0]", "", value)
    if not re.fullmatch(r"-?[0-9][0-9.,]*", text):
        return None
    comma, point = text.rfind(","), text.rfind(".")
    decimal = None
    if comma >= 0 and point >= 0:
        decimal = "," if comma > point else "."
    elif comma >= 0:
        decimal = "," if delimiter == ";" or not re.fullmatch(r"-?[0-9]{1,3}(?:,[0-9]{3})+", text) else None
    elif point >= 0:
        decimal = None if delimiter == ";" and re.fullmatch(r"-?[0-9]{1,3}(?:\.[0-9]{3})+", text) else "."
    group = ("," if comma >= 0 else ".") if decimal is None else "." if decimal == "," else ","
    plain = text.replace(group, "").replace(decimal or group, ".", 1)
    return float(plain) if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", plain) else None


_MARKDOWN_ROW = re.compile(rf"[{_JS_SPACE}]*\|{DOT}*\|[{_JS_SPACE}]*")
_MARKDOWN_RULE = re.compile(rf"[{_JS_SPACE}]*\|(?:[{_JS_SPACE}]*:?-+:?[{_JS_SPACE}]*\|)+[{_JS_SPACE}]*")


def markdown_rows(lines: list[str]) -> dict[int, dict]:
    """The Markdown tables in a document's lines, as a workbook reaches the engine (the gateway writes each sheet as
    one) or as a text holds them (core's markdownRows): each data row's line with its table's header row. A row is
    a record of its own, its cells the header's fields: as one paragraph of up to 40 lines, a table's rows were one
    record, and fields found none of them."""
    rows: dict[int, dict] = {}
    index = 0
    while index + 1 < len(lines):
        if (not _MARKDOWN_ROW.fullmatch(lines[index]) or _MARKDOWN_RULE.fullmatch(lines[index])
                or not _MARKDOWN_RULE.fullmatch(lines[index + 1])):
            index += 1
            continue
        header = {"line": index + 1, "cells": cells_of(lines[index], "|"), "delimiter": "|"}
        at = index + 2
        while at < len(lines) and _MARKDOWN_ROW.fullmatch(lines[at]) and not _MARKDOWN_RULE.fullmatch(lines[at]):
            rows[at + 1] = header
            at += 1
        index = at
    return rows


def names_in(value) -> list[str]:
    """The names an argument holds, however a model sends them (core's namesIn): a list, a list turned into text (a
    harness makes one of a list where a string is asked for: sort ["afsendt desc"] reached find as
    '["afsendt desc"]', which sorted nothing), or names with commas between, in a list's items too: Hermes turns
    "parti, dato" sent for a list into a list of that one string. Up to 12."""
    items = value
    if isinstance(value, str):
        text = value.strip()
        try:
            items = json.loads(text) if text.startswith("[") else text.split(",")
        except (ValueError, RecursionError):
            items = re.sub(r"^\[|\]$", "", text).split(",")
    if not isinstance(items, list):
        return []
    names = (re.sub(r"""^["']|["']$""", "", part.strip()).strip()
             for item in items if isinstance(item, str) for part in item.split(","))
    return [name for name in names if name][:12]


def fields_of(value) -> list[str]:
    """The fields a find asks for: up to 12 names."""
    return names_in(value)


def sort_of(value) -> list[dict]:
    """How a find sorts its records (core's sortOf): by a field ("betalt-forfald desc" for the largest first), ties
    by the next, up to three."""
    keys = []
    for name in names_in(value)[:3]:
        found = re.fullmatch(r"(.*?)(?:\s+(asc|desc))?", name, re.I | re.S)
        keys.append({"field": found.group(1).strip(), "descending": (found.group(2) or "").lower() == "desc"})
    return keys


def group_of(value) -> str | None:
    """The field a find groups its records by, if any."""
    names = names_in(value)
    return names[0] if names else None


def _sorted_by(sort: list[dict]) -> str:
    """A sort as find names it in a listing's first line."""
    return ", then ".join(f"{key['field']}{', largest first' if key['descending'] else ''}" for key in sort)


_JSON_FIELD = re.compile(r'^"([^"]{1,80})"\s*:\s*(.*?),?$')
_TEXT_FIELD = re.compile(r'^([^:"{}\[\]]{1,60}):\s*(.+)$')


def pairs_of(start: int, end: int, text, table: dict | None = None) -> list[tuple[str, str]]:
    """A record's fields and values as it names them (core's pairsOf): a table row's columns, a JSON object's keys, a
    text's words before a colon. `text(line)` is a line of the document, trimmed."""
    if table is not None and start == end:
        cells = cells_of(text(start), table["delimiter"])
        return [(name, cells[index] if index < len(cells) else "") for index, name in enumerate(table["cells"])]
    pairs = []
    for line in range(start, end + 1):
        found = _JSON_FIELD.match(text(line)) or _TEXT_FIELD.match(text(line))
        if found:
            pairs.append((found.group(1), re.sub(r'^"(.*)"$', r"\1", found.group(2))))
    return pairs


def fields_of_record(start: int, end: int, text, table: dict | None = None):
    """A record's fields by name (core's fieldsOfRecord): a JSON key, the words before a text line's colon or a table
    row's column, in any case, else the first key that holds the name, else in a text the number its words stand
    next to in a sentence; "a-b" is the days from one date to the other, or one number less the other. None for a
    field the record lacks."""
    pairs = [(key.lower(), value) for key, value in pairs_of(start, end, text, table)]

    def own(field: str) -> str | None:
        wanted = field.lower()
        pair = next((pair for pair in pairs if pair[0] == wanted), None) or next(
            (pair for pair in pairs if wanted in pair[0]), None)
        if pair is not None and pair[1] != "":
            return pair[1]
        # A table's row names its fields in its header; a text may name them only in its sentences.
        return in_sentence(field, start, end, text) if table is None else None

    def value(field: str) -> str | None:
        found = own(field)
        if found is not None or "-" not in field:
            return found
        left, right = (own(part.strip()) for part in field.split("-")[:2])
        return _difference(left, right, table)

    return value


def _paths_in(value: dict, keys: tuple = (), out: list | None = None) -> list[tuple[str, tuple, object]]:
    """The keys inside a JSON object, below its own, each with the path of keys to it in lower case, the keys as
    written, and its value (core's pathsIn); a list's items are left out, as len and sum count and add them up."""
    out = [] if out is None else out
    if len(keys) >= 8:
        return out
    for key, item in json_records.entries(value):
        path = (*keys, key)
        if keys:
            out.append((".".join(path).lower(), path, item))
        if isinstance(item, dict):
            _paths_in(item, path, out)
    return out


def json_names(value: dict) -> list[str]:
    """The fields a JSON record names, as it writes them: its keys, and the paths to the values inside its objects
    (core's jsonNames)."""
    inside = [".".join(keys) for _, keys, item in _paths_in(value) if not isinstance(item, (dict, list))]
    return [*(key for key, _ in json_records.entries(value)), *inside][:40]


def object_reader(value: dict) -> tuple:
    """One JSON object's values by name, worked out when first asked for (core's readerOf): `read(wanted)` gives its
    own key in any case, else the first key that holds the name, else a key inside it by its path, else the only one
    whose path ends so; `own(wanted)` its own key alone; `path(wanted)` a key inside it by its whole path alone. Each
    gives the keys to the value from the object, and the value, or None. Built anew for each record that stands in
    the object, 4,000 records under one object's keys took 44 s and 4.4 GB (2026-11-16)."""
    keys: dict | None = None
    inside: list | None = None
    paths: dict | None = None

    def path(wanted: str) -> tuple | None:
        nonlocal inside, paths
        if inside is None:
            inside = _paths_in(value)
        if paths is None:
            paths = {}
            for lowered, written, item in inside:
                paths.setdefault(lowered, (written, item))
        return paths.get(wanted)

    def own(wanted: str) -> tuple | None:
        nonlocal keys
        if keys is None:
            keys = {}
            for key, item in json_records.entries(value):
                keys.setdefault(key.lower(), ((key,), item))
        return keys.get(wanted)

    def read(wanted: str) -> tuple | None:
        nonlocal inside
        mine = own(wanted)
        if mine is not None:
            return mine
        for key, held in keys.items():
            if wanted in key:
                return held
        exact = path(wanted)
        if exact is not None:
            return exact
        ending = [(written, item) for lowered, written, item in inside if lowered.endswith("." + wanted)]
        return ending[0] if len(ending) == 1 else None

    return read, own, path


def fields_of_json(found: dict, whole: tuple = (), reader_of=lambda entry: object_reader(entry["value"])):
    """A JSON object's fields by name, each as its text, its JSON kind and the place its value stands at in the
    document (core's fieldsOfJson): its key in any case, else the first key that holds the name; a key inside an
    object by its path ("detaljer.pris_kr"), else the only key inside it whose path ends so; else the key itself, or
    the whole path to one inside, in the objects it is a part of (`whole`, nearest first): a sum of
    "detaljer.reservedel.pris_kr" over entries and parts the words found left out the entries found by their part
    alone (2026-11-16). Never a number from a sentence, as a JSON value says what
    it is. "a-b" is the days from one date to the other, or one number less the other, and no difference with a
    null. None for a field the object lacks. `reader_of` gives an object's reader, kept for the find."""
    chain = (found, *whole)

    def placed(held: tuple, depth: int) -> tuple[str, str, str]:
        # A value's place: the outermost object it is a part of, and the keys to it from there, whichever record
        # reads it.
        keys, item = held
        return (*json_records.field(item),
                json.dumps([chain[-1]["offset"], *(entry["key"] for entry in reversed(chain[depth:-1])), *keys]))

    def own(field: str) -> tuple[str, str, str] | None:
        wanted = field.lower().strip()
        mine = reader_of(found)[0](wanted)
        if mine is not None:
            return placed(mine, 0)
        for index, entry in enumerate(whole):
            held = reader_of(entry)[1](wanted)
            if held is None and "." in wanted:
                # A path has a point in it: no object is gone through for a plain name.
                held = reader_of(entry)[2](wanted)
            if held is not None:
                return placed(held, index + 1)
        return None

    def value(field: str) -> tuple[str, str, str] | None:
        found_here = own(field)
        if found_here is not None or "-" not in field:
            return found_here
        left, right = (own(part.strip()) for part in field.split("-")[:2])
        if (left is not None and left[1] == "null") or (right is not None and right[1] == "null"):
            return None
        result = _difference(None if left is None else left[0], None if right is None else right[0])
        return None if result is None else (result, "number", f"{left[2]}-{right[2]}")

    return value


def added_once(places: list) -> list[bool]:
    """Which of the values read for a sum are added, by the place each stands at (core's addedOnce): each value of a
    document once, whichever records read it. The words of a find stood in a log entry, in its details and in its
    part, three records that all read the part's price: two prices, 1,000 and 500, came to 4,500 (2026-11-16)."""
    seen: set = set()
    once = []
    for place in places:
        once.append(place is None or place not in seen)
        if place is not None:
            seen.add(place)
    return once


class _Folds(dict):
    """Characters as core's expressions compare them in any case (JavaScript's /iu, simple case folding): one
    character for one, so a place in a folded text is the same place in the text. Python's re.I would take the
    Turkish dotted and dotless i for an i, which core does not."""

    def __missing__(self, code: int) -> str:
        char = chr(code)
        self[code] = next((form for form in (char.casefold(), char.lower()) if len(form) == 1), char)
        return self[code]


_FOLDS = _Folds()


def in_sentence(field: str, start: int, end: int, text) -> str | None:
    """The number a record's sentences give a field, when no line names it before a colon (core's inSentence): the
    one just before its words ("21 rejected units"), else the first after them in the same sentence ("an upper
    acceptance limit of 42 kPa", "the agreed net unit price is EUR 484.50", "en bøde på kr. 12.500"). An archive
    that wrote "The signed result is 1069 accepted units and 21 rejected units." in each of 180 reports gave fields
    and sum nothing to read, and the model, asked for the rejected units in all, opened the reports page by page
    until it was stopped at 10 minutes. Matched as core matches, in any case, with JavaScript's \\w and \\d (ASCII
    only); Python's re has no \\p{L}, so a letter of any script just around the words is looked at here."""
    spaces = f"[{_JS_SPACE}]+"
    words = spaces.join(_literal(word) for word in re.split(spaces, field.translate(_FOLDS)) if word)
    if not words:
        return None
    prose = "".join(f"{text(line)} " for line in range(start, end + 1)).translate(_FOLDS)

    def apart(at: int) -> bool:
        # Neither a letter of any script nor a digit here (core's [\p{L}\d]), or the start or end of the text.
        return not 0 <= at < len(prose) or not (prose[at].isalpha() or "0" <= prose[at] <= "9")

    before = re.compile(rf"(?<![a-z0-9_.,-])(-?[0-9][0-9.,]*){spaces}(?:{words})")
    at = 0
    while (found := before.search(prose, at)) is not None:
        if apart(found.end()):
            return re.sub(r"[.,]\Z", "", found.group(1))
        at = found.start() + 1
    named = re.compile(words)
    at = 0
    while (found := named.search(prose, at)) is not None:
        after = _amount_after(prose, found.end()) if apart(found.start() - 1) and apart(found.end()) else None
        if after is not None:
            return after
        at = found.start() + 1
    return None


_AMOUNT = re.compile(r"-?(?:[0-9][0-9.,]*[0-9]|[0-9])")


def _amount_after(prose: str, at: int) -> str | None:
    """The first number within 40 steps of `at` in a folded text, in the same sentence (core's inSentence): a step is
    a character other than a point, a digit or a line break, or a "kr." (Dkr., Nkr., Skr.) or "ca." after no letter,
    as the sentence goes on past them before an amount. Core's expression steps over a "kr." whole only where
    stepping over its letters stops at its point, so a walk finds the number it finds."""
    for steps in range(41):
        amount = _AMOUNT.match(prose, at)
        if amount is not None:
            return amount.group()
        if steps == 40 or at >= len(prose):
            return None
        step = 4 if prose.startswith(("dkr.", "nkr.", "skr."), at) else 3 if prose.startswith(("kr.", "ca."), at) else 0
        if step:
            if prose[at - 1].isalpha():
                return None
            at += step
        elif prose[at] in ".\n" or "0" <= prose[at] <= "9":
            return None
        else:
            at += 1
    return None


def field_line(path: str, hit: int, around: dict | None, text, fields: list[str], table: dict | None = None,
               read=None) -> str:
    """One record as one line, holding only the fields asked for (core's fieldLine): a JSON key ("forfald":
    "2041-06-19") or the words before the colon of a text record's line ("Afsendelsesdato: 2041-04-02"). A field
    the record lacks shows as "-"; a value's line breaks show as \\n, and a long one is cut so the fields fit a page.
    `read(field)` reads the record's fields when they are read already."""
    start = around["start"] if around is not None else hit
    end = around["end"] if around is not None else hit
    value = read or fields_of_record(start, end, text, table)
    budget = min(LINE_CHARS, (knowledge.RECORD_CHARS - 500) // max(1, len(fields)))

    def shown(field: str) -> str:
        found = value(field)
        raw = ("-" if found is None else found).replace("\r", "\\r").replace("\n", "\\n")
        return raw if len(raw) <= budget else f"{raw[:budget]}…"

    return f"- {path}:{start}{f'-{end}' if end > start else ''}: {'; '.join(f'{field}={shown(field)}' for field in fields)}"


_DATE = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})")
_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _day(found: re.Match) -> int | None:
    """The day a YYYY-MM-DD date is (core's difference), counted from any day; None for a day no calendar has
    ("2041-02-30", "2041-13-01"), as JavaScript's Date gives another day for it."""
    year, month, day = (int(part) for part in found.groups())
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    if not 1 <= month <= 12 or not 1 <= day <= (29 if month == 2 and leap else _MONTH_DAYS[month - 1]):
        return None
    # Days from 0000-03-01 (Howard Hinnant's days_from_civil), which takes any year.
    shifted = year - (month <= 2)
    era, of_era = divmod(shifted, 400)
    of_year = (153 * (month - 3 if month > 2 else month + 9) + 2) // 5
    return era * 146_097 + of_era * 365 + of_era // 4 - of_era // 100 + of_year + day - 1


def _number_text(value: float) -> str:
    """A number as JavaScript writes it (String(number)): a whole one without ".0"."""
    return json_records.number_text(value)


# Enough digits for toFixed's 100 decimals of a number below 10^21.
_FIXED = Context(prec=160)


def _to_fixed(value: float, digits: int) -> str:
    """A number with this many decimals, as JavaScript's toFixed rounds it: half away from zero, from the number's
    exact value; from 10^21 on, as String(number) writes it."""
    if not math.isfinite(value) or abs(value) >= 1e21:
        return _number_text(value)
    return format(Decimal(value if value != 0 else 0.0).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP,
                                                                 context=_FIXED), "f")


# A date, or a date and a time, as ISO writes it: "2043-03-14", "2043-03-14T08:12", "2043-03-14 08:12:05+01:00"
# (core's MOMENT).
_MOMENT = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})(?:[T ]([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:[.,]([0-9]+))?)?)?"
                     r"\s*(Z|[+-][0-9]{2}:?[0-9]{2})?")


def _moment_of(text: str) -> tuple[float, bool] | None:
    """A date's or a time's minutes, at its offset if it names one, else as UTC, and whether it has a time of day
    (core's momentOf); None when it is no real date."""
    found = _MOMENT.fullmatch(text.strip(_JS_TRIMMED))
    if found is None:
        return None
    day = _day(_DATE.fullmatch(f"{found.group(1)}-{found.group(2)}-{found.group(3)}"))
    hour, minute, second = (int(found.group(at) or 0) for at in (4, 5, 6))
    if day is None or hour > 23 or minute > 59 or second > 59:
        return None
    zone = found.group(8)
    offset = (0 if zone is None or zone == "Z"
              else (-1 if zone.startswith("-") else 1) * (int(zone[1:3]) * 60 + int(zone[-2:])))
    fraction = float(f"0.{found.group(7)}") if found.group(7) else 0.0
    return day * 1440 + hour * 60 + minute + (second + fraction) / 60 - offset, found.group(4) is not None


def _difference(left: str | None, right: str | None, table: dict | None = None) -> str | None:
    """One field's value less another's (core's difference): the days between two dates (each all of its value,
    YYYY-MM-DD, a day the calendar has), the minutes between two times (each a date with a time of day: an alarm
    acknowledged 17 minutes after it was raised), or the difference of two numbers. None when the two are not both
    dates or both numbers."""
    if left is None or right is None:
        return None
    first, second = _moment_of(left), _moment_of(right)
    if first is not None and second is not None:
        # A date and a time of day are not alike: a day is no moment of it, so they give no difference.
        if first[1] != second[1]:
            return None
        if first[1]:
            return _number_text(float(f"{first[0] - second[0]:.15g}"))
        return str(round((first[0] - second[0]) / 1440))
    if table is not None:
        minuend, subtrahend = _table_number(left, table["delimiter"]), _table_number(right, table["delimiter"])
        return (None if minuend is None or subtrahend is None
                else _number_text(float(_to_fixed(minuend - subtrahend, 6))))
    number = re.compile(r"-?[0-9]+(?:[.,][0-9]+)?")
    if number.fullmatch(left) and number.fullmatch(right):
        return _number_text(float(left.replace(",", ".", 1)) - float(right.replace(",", ".", 1)))
    return None


def _dates_in(row: str) -> list[str]:
    """The fields of a one-line record that hold a date (YYYY-MM-DD), in their order."""
    return [found.group(1) for found in
            re.finditer(r"(?:: |; )([^=;]+)=[0-9]{4}-[0-9]{2}-[0-9]{2}(?=;|\Z)", row)]


# A number at the start of a value, and what may follow it (a unit): "6,375 m³", "1.150,00 kr.", "-12"; not a date
# or a time.
_NUMBER_AT_START = re.compile(r"(-?[0-9][0-9.,]*)(?:\s*[^0-9\s.,:/-].*)?")


def _mark_of(token: str) -> str | None:
    """What one number's marks say its decimal mark is (core's markOf): the last of two kinds; one kind followed by
    other than three digits, or after a lone 0 or more than three digits; the other kind when one stands several
    times. "1.150" or "6,375" alone says nothing."""
    comma, point = token.rfind(","), token.rfind(".")
    if comma >= 0 and point >= 0:
        return "," if comma > point else "."
    mark = "," if comma >= 0 else "." if point >= 0 else None
    if mark is None:
        return None
    parts = token.removeprefix("-").split(mark)
    if len(parts) > 2:
        return "." if mark == "," else ","
    return mark if len(parts[1]) != 3 or parts[0] == "0" or len(parts[0]) > 3 else None


# "kroner" before "kr": core's expression takes "kr" only when no letter follows it, and "kroner" when "kr" is not.
# Dkr., Nkr. and Skr. are kroner too.
_KRONER = re.compile(
    rf"(?<![0-9.,])[0-9]{{1,3}}(?:\.[0-9]{{3}})+(?:,-)?(?![0-9,])[{_JS_SPACE}]?(?:kroner|[dns]?kr|dkk|nok|sek)")
_KRONER_FIRST = re.compile(rf"(?:dkk|nok|sek|[dns]?kr\.)[{_JS_SPACE}]?[0-9]{{1,3}}(?:\.[0-9]{{3}})+(?![0-9,])")


def _kroner(text: str) -> int:
    """How many amounts in kroner a text writes with a point before thousands ("12.500 kr.", "12.500 kroner",
    "12.500,- kr", "kr. 12.500", "Dkr. 12.500", "DKK 12.500"), matched as core matches them: in any case, the
    currency after the amount not followed by a letter of any script, and before it not preceded by one (its /iu
    with \\p{L}, which Python's re lacks)."""
    folded = text.translate(_FOLDS)
    count = at = 0
    while (found := _KRONER.search(folded, at)) is not None:
        if found.end() < len(folded) and folded[found.end()].isalpha():
            at = found.start() + 1
        else:
            count += 1
            at = found.end()
    at = 0
    while (found := _KRONER_FIRST.search(folded, at)) is not None:
        if found.start() > 0 and folded[found.start() - 1].isalpha():
            at = found.start() + 1
        else:
            count += 1
            at = found.end()
    return count


def _document_mark(lines: list[str]) -> str | None:
    """The decimal mark a document's amounts use, for values that do not say (core's documentMark): amounts with
    both marks ("1.150,00") or kroner with a point before thousands ("12.500 kr.": fines in a conversation's notes
    were not added up), else the more frequent of a comma or a point before one or two decimals."""
    text = "\n".join(lines)
    comma = len(re.findall(r"[0-9]\.[0-9]{3},[0-9]", text)) + _kroner(text)
    point = len(re.findall(r"[0-9],[0-9]{3}\.[0-9]", text))
    if comma != point:
        return "," if comma > point else "."
    commas = len(re.findall(r"(?<![0-9.,])[0-9]+,[0-9]{1,2}(?![0-9.,])", text))
    points = len(re.findall(r"(?<![0-9.,])[0-9]+\.[0-9]{1,2}(?![0-9.,])", text))
    return None if commas == points else "," if commas > points else "."


def numbers_of(entries: list[tuple]) -> list[dict | None]:
    """One field's numbers over found records, each `(document, table, value, kind)` (core's numbersOf), each read as
    its document writes numbers: a JSON number as the number it is (kind "number"), never a JSON true, null, list
    or object, a table's as its delimiter says (a decimal comma where semicolons cut the rows), otherwise as the
    field's values say, else a JSON number with a point, else as the document's amounts say. A value that is no
    number, or one whose mark nothing tells (6,375 alone), is None; a number is {value, decimals, mark}. `kind` is
    None for a value read from text."""
    tokens = []
    for _, _, value, kind in entries:
        if kind is not None and kind != "string":
            tokens.append(None)
            continue
        found = None if value is None else _NUMBER_AT_START.fullmatch(value.strip())
        token = None if found is None else found.group(1)
        tokens.append(token[:-1] if token is not None and token[-1] in ".," else token)
    marks: dict[int, str | None] = {}

    def mark_in(doc: dict, table: dict | None) -> str | None:
        # A CSV or TSV table's delimiter says it; a Markdown table's cells say no more than a text's.
        if table is not None and table["delimiter"] != "|":
            return "," if table["delimiter"] == ";" else "."
        if id(doc) in marks:
            return marks[id(doc)]
        votes = {",": 0, ".": 0}
        for (other, _, _, _), token in zip(entries, tokens):
            own = _mark_of(token) if other is doc and token is not None else None
            if own is not None:
                votes[own] += 1
        mark = (("," if votes[","] > votes["."] else ".") if votes[","] != votes["."]
                else "." if doc["json"] else _document_mark(doc["lines"]))
        marks[id(doc)] = mark
        return mark

    numbers: list[dict | None] = []
    for (doc, table, value, kind), token in zip(entries, tokens):
        if kind == "number" and value is not None:
            number = float(value)
            if not math.isfinite(number) or (number == int(number) and abs(number) > json_records.SAFE):
                numbers.append(None)
                continue
            base, _, exponent = value.lower().partition("e")
            decimals = max(0, len(base.partition(".")[2]) - int(exponent or "0"))
            numbers.append({"value": number, "decimals": decimals, "mark": None if decimals == 0 else "."})
            continue
        if token is None:
            numbers.append(None)
            continue
        if re.fullmatch(r"-?[0-9]+", token):
            numbers.append({"value": float(token), "decimals": 0, "mark": None})
            continue
        mark = _mark_of(token) or mark_in(doc, table)
        parts = None if mark is None else token.replace("." if mark == "," else ",", "").split(mark)
        if parts is None or len(parts) > 2 or not re.fullmatch(r"-?[0-9]+", parts[0]):
            numbers.append(None)
            continue
        fraction = parts[1] if len(parts) > 1 else ""
        numbers.append({"value": float(f"{parts[0]}.{fraction or '0'}"), "decimals": len(fraction),
                        "mark": mark if fraction else None})
    return numbers


def sum_text(numbers: list[dict], named: bool = True) -> str:
    """A sum written as its numbers are (core's sumText): with their most decimals, after their decimal mark, which
    it names unless told not to, and with the other mark between thousands, so it can be copied as the document
    writes amounts. Given "22558,0", a model wrote "22.558,5" when it put in the point itself. A sum past the largest
    number is said to be unavailable; one with more than 100 decimals is written as JavaScript writes the number."""
    decimals = max([0, *(number["decimals"] for number in numbers)])
    total = 0.0
    for number in numbers:  # one at a time, as JavaScript adds them: Python's sum() compensates
        total += number["value"]
    if not math.isfinite(total):
        return "unavailable (sum exceeds the numeric range)"
    if decimals > 100:
        return _number_text(total)
    marks = list(dict.fromkeys(number["mark"] for number in numbers if number["mark"] is not None))
    mark = None if decimals == 0 else marks[0] if len(marks) == 1 else "."
    whole, _, fraction = _to_fixed(total, decimals).partition(".")
    if mark is None:
        return whole
    text = f"{re.sub(r'(?<=[0-9])(?=(?:[0-9]{3})+$)', '.' if mark == ',' else ',', whole)}{mark}{fraction}"
    if not named:
        return text
    return (f"{text} (a decimal comma, a point between thousands)" if mark == ","
            else f"{text} (a decimal point, a comma between thousands)")


# Letters that are not a letter and an accent, as JavaScript's default collation (en-US) sorts them: Ærø by "aero",
# Ørsted by "orsted".
_LETTERS = str.maketrans({"æ": "ae", "ø": "o", "œ": "oe", "đ": "d", "ð": "d", "ł": "l", "þ": "th"})
# Punctuation and symbols in that order, before digits.
_MARKS = {char: index for index, char in enumerate("_-–—,;:!?.·'\"«»()[]{}§@*/\\&#%`^°+±÷×<=>|~¤$£€")}


def _collation(text: str) -> tuple[list, str]:
    """A value's place in a sort, close to JavaScript's localeCompare in its default locale (en-US): spaces, then
    punctuation and symbols, then digits, then letters; by the letters first, then their accents, then their case,
    lower before upper. Core's order follows the locale of the machine it runs on: with da-DK, æ, ø and å come after
    z."""
    base = "".join(char for char in unicodedata.normalize("NFD", text.casefold()) if not unicodedata.combining(char))
    return ([(0, 0, char) if char.isspace() else (2, 0, char) if char.isdigit() else (3, 0, char) if char.isalpha()
             else (1, _MARKS.get(char, len(_MARKS)), char) for char in base.translate(_LETTERS)], text.swapcase())


def named_documents(documents: list[dict], wanted: str) -> tuple[list[dict], str, list[str]]:
    """The documents a find names (core's namedDocuments): one by its path or the end of it, those a pattern matches
    (0?_Afregning.json), else those whose paths start with it, else those whose paths hold it ("Afregning":
    01_Afregning.json to 04_Afregning.json); several such names with commas between name them all. The documents,
    how they were named ("one", "list", "start" or "part"), and the names in a list that no document has."""
    def one(name: str) -> dict | None:
        exact = next((entry for entry in documents if entry["path"] == name), None)
        if exact is not None:
            return exact
        ends = [entry for entry in documents if entry["path"].endswith("/" + name)]
        return ends[0] if len(ends) == 1 else None

    def named(name: str) -> tuple[list[dict], str]:
        exact = one(name)
        if exact is not None:
            return [exact], "one"
        lower = name.lower()
        # A pattern, as a shell writes it: Forsendelser_2041_K*.csv.
        if re.search(r"[*?]", name):
            return _by_pattern(documents, name), "part"
        starting = [entry for entry in documents if entry["path"].lower().startswith(lower)]
        if starting:
            return starting, "start"
        return [entry for entry in documents if lower in entry["path"].lower()], "part"

    exact = one(wanted)
    if exact is not None:
        return [exact], "one", []
    parts = [part.strip() for part in wanted.split(",") if part.strip()]
    if len(parts) > 1:
        each = [(part, named(part)[0]) for part in parts]
        files = list({id(entry): entry for _, found in each for entry in found}.values())
        return files, "list", [part for part, found in each if not found]
    files, how = named(wanted)
    return files, how, []


def _by_pattern(documents: list[dict], name: str) -> list[dict]:
    """The documents a pattern names (a * any characters, a ? one), by their whole path or its last part, in any case
    (core's byPattern)."""
    pattern = re.compile("".join(".*" if char == "*" else "." if char == "?" else re.escape(char)
                                 for char in name.lower()))
    return [entry for entry in documents if pattern.fullmatch(entry["path"].lower())
            or pattern.fullmatch(entry["path"].split("/")[-1].lower())]


def _alike(path: str) -> str:
    """A path but for its numbers: documents named alike (01_Afregning.json, 02_Afregning.json) are volumes, months."""
    return re.sub(r"[0-9]+", "0", path)


def _kin_pattern(documents: list[dict], family: list[str]) -> str | None:
    """Patterns that name exactly these documents of one kind, for document (core's kinPattern): a ? for each digit,
    one pattern for each count of digits (Rapport_?.txt, Rapport_??.txt, Rapport_???.txt for Rapport_1.txt to
    Rapport_120.txt), else a * for each number; None when neither names exactly them. A * also took the replies
    beside 300 letters (Brev_FB-41-00007-svar.txt), and a ? for each digit alone named 9 of 120 reports numbered
    from 1."""
    wanted = set(family)

    def exactly(names: list[str]) -> bool:
        return {entry["path"] for name in names for entry in _by_pattern(documents, name)} == wanted

    # Documents named alike differ only in their numbers, so the shorter shape is the one with fewer digits.
    shapes = sorted(dict.fromkeys(re.sub(r"[0-9]", "?", path) for path in family), key=len)
    candidates = ([shapes] if len(shapes) <= 3 else []) + [[re.sub(r"[0-9]+", "*", family[0])]]
    return next((", ".join(names) for names in candidates if exactly(names)), None)


def _holder_lines(holders: list[dict]) -> list[str]:
    """Each holder's first line with the words, and the line after it, a line each, the rest by name (core's
    holderLines): told a directory held an employee's id, an agent answered with the id; the name stood on the line
    after it."""
    lines = []
    for match in holders[:HOLDER_LINES]:
        after = next((text for text in match["after"] if text.strip()), None)
        line = f"- {match['path']}:{match['line']}: {match['text'].strip()}"
        lines.append((line if after is None else f"{line} / {match['line'] + 1}: {after.strip()}")[:240])
    if len(holders) > HOLDER_LINES:
        rest = ", ".join(match["path"] for match in holders[HOLDER_LINES:HOLDER_LINES + LISTED_DOCUMENTS])
        lines.append(f"- and {rest}{', …' if len(holders) > HOLDER_LINES + LISTED_DOCUMENTS else ''}")
    return lines


def _listed(path: str, read: dict) -> dict:
    """A document find lists from: its lines, in lower case too, its header row when it is a table (CSV, TSV), the
    data rows of its Markdown tables with their header rows, the header and rule lines of those (no hits), whether it
    is JSON, and a JSON, JSON Lines or NDJSON document's objects, its records (core's listedIn). One that is not valid
    JSON, or whose numbers find can't report exactly, is refused with what to do instead."""
    table = read.get("table")
    is_json = record.is_json_document(path)
    rows = markdown_rows(read["lines"]) if table is None and not is_json else {}
    heads = {line for header in rows.values() for line in (header["line"], header["line"] + 1)}
    listed = {"path": path, "lines": read["lines"], "lower": [line.lower() for line in read["lines"]], "table": table,
              "rows": rows, "heads": heads, "json": is_json}
    if json_records.is_structured_json(path):
        try:
            listed["objects"] = json_records.json_records(read["lines"], path)
        except json_records.JsonRecordError as error:
            _fail(str(error))
    return listed


def _holding(doc: dict, needles: list[str]) -> list[int]:
    """The lines of a listed document that hold every needle, in any case: not a table's header row. In a JSON
    document, the objects that hold them all on one line of their own fields: their places in its records."""
    if "objects" in doc:
        return [index for index, found in enumerate(doc["objects"])
                if any(all(needle in line for needle in needles) for line in found["search"].split("\n"))]
    table = doc["table"]
    found = []
    for index, lower in enumerate(doc["lower"]):
        if (table is not None and index + 1 <= table["line"]) or index + 1 in doc["heads"]:
            continue
        if all(needle in lower for needle in needles):
            found.append(index + 1)
    return found


def _table_at(doc: dict, line: int) -> dict | None:
    """The header row of the table a line of a listed document is a row of, if any."""
    if doc["table"] is not None:
        return doc["table"] if line > doc["table"]["line"] else None
    return doc["rows"].get(line)


def _as_written(match: dict, read: dict | None) -> dict:
    """A row the engine's search found in a table, as the table's original writes it, alone (core's find): the rows
    beside it are other records. Asked for one customer's rentals, an agent got 25 rows and 50 rows of other
    customers, 16,756 characters in all, and the turn no longer fitted the model."""
    if read is None:
        return match
    lines = read["lines"]
    line = match["line"]
    return {**match, "text": lines[line - 1].strip() if 1 <= line <= len(lines) else match["text"],
            "before": [], "after": []}


# What JavaScript's trim takes off a text's ends, as core trims a line: a byte order mark too, never a file separator
# as Python's strip does.
_JS_TRIMMED = record.JS_TRIMMED


def _text_of(doc: dict):
    return lambda line: doc["lines"][line - 1].strip(_JS_TRIMMED)


def written_lines(written: list[str]) -> list[str]:
    """What the person wrote in a conversation, as the lines find_in_conversation reads (core's writtenLines): each
    of their messages headed with its number, its text a line each, an empty line after it; an empty message is
    left out. Only the person's own messages: counted together with the agent's replies, which repeat figures, one
    typed fine became three."""
    lines: list[str] = []
    number = 0
    for text in written:
        said = text.strip(_JS_TRIMMED)
        if said:
            number += 1
            lines += [f"[message {number}, the person]", *re.split(r"\r?\n", said), ""]
    return lines


def _records(docs: list[dict], hits: list[list[int]], query: str) -> list[dict]:
    """The records the found lines are in, each with what reads its fields: a line in a record already taken is
    left out, and a table's record is its row, as is each line of a conversation (two fines in one note are two
    records). A JSON document's record is the object found, shown from its first line that holds the words, and its
    fields are read as JSON types them."""
    records = []
    needle = query.lower()
    for doc, found in zip(docs, hits):
        last = None
        by_offset = {entry["offset"]: entry for entry in doc.get("objects") or []}
        readers: dict[int, tuple] = {}

        def reader_of(entry: dict, readers: dict = readers) -> tuple:
            """An object's reader, one for the find: a record and every part of it read the same object."""
            if entry["offset"] not in readers:
                readers[entry["offset"]] = object_reader(entry["value"])
            return readers[entry["offset"]]

        def whole_of(found_object: dict, by_offset: dict = by_offset) -> tuple:
            """The objects a JSON record is a part of, nearest first (core's wholeOf): those it stands under a key
            of, out to the first that is a list's item or stands at the top. The fields it lacks are read there, and
            no further out: an entry of a report's list read the report's own status as its own (2026-11-16)."""
            chain, at = [], found_object
            while at["key"] != "" and at["parent"] in by_offset:
                at = by_offset[at["parent"]]
                chain.append(at)
            return tuple(chain)

        for hit in found:
            if "objects" in doc:
                found_object = doc["objects"][hit]
                start, end = found_object["start"], found_object["end"]
                at = next((index for index, line in enumerate(doc["lower"][start - 1:end]) if needle in line), 0)
                whole = whole_of(found_object)
                typed = fields_of_json(found_object, whole, reader_of)
                records.append({"doc": doc, "hit": start + at, "table": None, "typed": typed, "object": found_object,
                                # The outermost object it is a part of: itself, or the entry its details and its
                                # part stand in (core's outermost).
                                "outermost": (id(doc), (whole[-1] if whole else found_object)["offset"]),
                                "record": {"start": start, "end": end, "whole": True, "lines": {}},
                                "names": json_names(found_object["value"]),
                                "read": lambda field, typed=typed: (lambda value: None if value is None else value[0])(
                                    typed(field))})
                continue
            if last is not None and last["start"] <= hit <= last["end"]:
                continue
            table = _table_at(doc, hit)
            last = ({"start": hit, "end": hit, "whole": True, "lines": {hit: doc["lines"][hit - 1].strip(_JS_TRIMMED)}}
                    if table is not None or doc.get("line_records")
                    else record.record_in(doc["lines"], hit, doc["json"]))
            start, end = (last["start"], last["end"]) if last is not None else (hit, hit)
            # Each record's fields are read once a find: sort and sum ask for several over thousands of records.
            records.append({"doc": doc, "hit": hit, "record": last, "table": table, "typed": None,
                            "read": fields_of_record(start, end, _text_of(doc), table)})
    return records


def sorted_records(records: list[dict], sort: list[dict], values_of) -> list[dict]:
    """Records sorted by their fields (core's listedIn): as numbers when every record that has the field has a number
    there (a JSON null counts as none), else as text; records without the field last, whichever way; ties by the
    next field, then in their documents' order. `values_of(field)` is each record's `(document, table, value,
    kind)`."""
    keys = []
    for entry in sort:
        values = values_of(entry["field"])
        numbers = numbers_of(values)
        numeric = all(value is None or kind == "null" or number is not None
                      for (_, _, value, kind), number in zip(values, numbers))
        texts = [None if kind == "null" else value for _, _, value, kind in values]
        keys.append((entry["descending"], texts,
                     [0.0 if number is None else number["value"] for number in numbers] if numeric
                     else [None if value is None else _collation(value) for value in texts]))

    def compare(a: int, b: int) -> int:
        for descending, values, order in keys:
            if values[a] is None or values[b] is None:
                if values[a] is None and values[b] is None:
                    continue
                return 1 if values[a] is None else -1
            by = (order[a] > order[b]) - (order[a] < order[b])
            if by:
                return -by if descending else by
        return 0

    return [records[index] for index in sorted(range(len(records)), key=functools.cmp_to_key(compare))]


def holding_all(lines: list[str], words: list[str], is_json: bool) -> list[int]:
    """The records of a document that hold every word, on any of their lines (core's holdingAll): one line of each
    (the one with the word found on fewest lines), in document order. A record is what find shows with a line: the
    JSON object around it, or a text document's paragraph; one longer than a record is shown is left out. `words`
    are in lower case."""
    lower = [line.lower() for line in lines]
    counts = [sum(1 for line in lower if word in line) for word in words]
    if not counts or min(counts) == 0:
        return []
    anchor = words[counts.index(min(counts))]
    found: list[int] = []
    taken: set[int] = set()
    for index, line in enumerate(lower):
        if anchor not in line:
            continue
        around = record.record_in(lines, index + 1, is_json)
        if around is None or not around["whole"] or around["start"] in taken:
            continue
        text = "\n".join(lower[around["start"] - 1:around["end"]])
        if not all(word in text for word in words):
            continue
        taken.add(around["start"])
        found.append(index + 1)
    return found


def _object_row(row: str) -> bool:
    """Whether a line is a JSON object, as a line of JSON Lines is: read as JavaScript's JSON.parse reads it."""
    try:
        return isinstance(json_records._parse(row), dict)
    except (ValueError, RecursionError):
        return False


def json_units(lines: list[str], path: str) -> list[dict] | None:
    """The objects of a JSON or JSON Lines document, each {first, last, depth, above, list}: its lines, how deep it
    stands, the containers it stands in (outermost first, the document's outermost left out: a key's, or a record's,
    each {line, key, record}), and its place in the list that holds it ({index, count, line, key}, or None). Read by a
    scan of its brackets outside its strings, once its text is known to be valid JSON: in anything else (JSON with
    comments) a bracket in a comment would move them. The engine's heading, and a repository's fence around the
    file, are left out. None when it is no valid JSON (core's jsonUnits)."""
    heading = next((index for index, row in enumerate(lines[:8]) if row.startswith("Source type: ")), -1)
    begin = heading + 1
    while begin < len(lines) and (lines[begin].strip(_JS_TRIMMED) == ""
                                  or (heading >= 0 and begin < heading + 5 and SOURCE_LINE.match(lines[begin]))):
        begin += 1
    end = len(lines) - 1
    if (lines[begin] if begin < len(lines) else "").startswith("```") and end > begin and CLOSER.match(lines[end]):
        begin += 1
        end -= 1
    body = [row[:-1] if row.endswith("\r") else row for row in lines[begin:end + 1]]
    text = "\n".join(body).removeprefix("\U0000feff")
    # JSON Lines, or an export of it named .json: every line its own record.
    jsonl = bool(JSON_LINES.search(path))
    if not jsonl:
        try:
            json_records._parse(text)
        except (ValueError, RecursionError):
            rows = [row for row in body if row.strip(_JS_TRIMMED) != ""]
            if len(rows) < 2 or not all(map(_object_row, rows)):
                return None
            jsonl = True
    if jsonl:
        rows = [begin + index + 1 for index, row in enumerate(body) if row.strip(_JS_TRIMMED) != ""]
        if not all(_object_row(body[line - begin - 1]) for line in rows):
            return None
        return [{"first": line, "last": line, "depth": 1, "above": [],
                 "list": {"index": index + 1, "count": len(rows), "line": rows[0], "key": ""}}
                for index, line in enumerate(rows)]
    units: list[dict] = []
    stack: list[dict] = []
    line = begin + 1
    last_string = ""
    key = ""
    for token in JSON_TOKEN.finditer(text, 1 if text.startswith("\U0000feff") else 0):
        char = token.group()
        if char == "\n":
            line += 1
        elif char[0] == '"':
            last_string = char[1:-1]
        elif char == ":":
            key = last_string
        elif char in "{[":
            parent = stack[-1] if stack else None
            stack.append({"char": char, "line": line, "key": key if parent and parent["char"] == "{" else "",
                          "record": char == "{" and parent is not None and parent["char"] == "[", "members": []})
            key = ""
        elif char in "}]":
            if not stack:
                return None
            opened = stack.pop()
            if char == "}":
                if stack and stack[-1]["char"] == "[":
                    stack[-1]["members"].append(len(units))
                units.append({"first": opened["line"], "last": line, "depth": len(stack),
                              "above": [{"line": held["line"], "key": held["key"], "record": held["record"]}
                                        for held in stack[1:] if held["key"] != "" or held["record"]],
                              "list": None})
            else:
                for index, member in enumerate(opened["members"]):
                    units[member] = {**units[member], "list": {"index": index + 1, "count": len(opened["members"]),
                                                               "line": opened["line"], "key": opened["key"]}}
            key = ""
        else:
            key = ""
    return units if not stack else None


_BRACKETS_ONLY = re.compile(rf"[{_JS_SPACE}{{}}\[\],]*")


def record_at(units: list[dict], line: int, lines: list[str]) -> dict | None:
    """The record a line of JSON stands in: the innermost object that is an element of a list (an order in a
    customer's list of orders), else the innermost object short of the document's own outermost one. Of objects on
    the same lines, the outermost: a record on one line holds its own lists. A record shares no line with another:
    where two stand on one line ('{"a": 1}, {"a": 2},'), the object that holds them both is the record, and where
    none does, there is none. A line of brackets alone ('}, {') is no such line (core's recordAt)."""
    holding = [unit for unit in units if unit["first"] <= line <= unit["last"] and unit["depth"] >= 1]
    listed = [unit for unit in holding if unit["list"] is not None]

    def nested(inner: dict, outer: dict) -> bool:
        return outer["first"] <= inner["first"] and inner["last"] <= outer["last"] and outer["depth"] < inner["depth"]

    def brackets(number: int) -> bool:
        return _BRACKETS_ONLY.fullmatch(lines[number - 1] if 0 < number <= len(lines) else "") is not None

    def shared(unit: dict, other: dict) -> bool:
        return any(not brackets(number)
                   for number in range(max(unit["first"], other["first"]), min(unit["last"], other["last"]) + 1))

    def alone(unit: dict) -> bool:
        return not any(other is not unit and other["first"] <= unit["last"] and unit["first"] <= other["last"]
                       and not nested(other, unit) and not nested(unit, other) and shared(unit, other) for other in units)

    return next(filter(alone, sorted(listed or holding, key=lambda unit: (unit["last"] - unit["first"], unit["depth"]))),
                None)


def headings_of(lines: list[str], start: int = 2, to: int | None = None, setext: bool = False) -> list[dict]:
    """The Markdown headings of a text ("## Title"), in line order, each {line, level, title}, from line `start` to
    `to` (a repository's file starts under the engine's heading). A line of a fenced block is no heading, and
    neither is the document's first line: the engine heads every document with its name there. With `setext`, a
    paragraph underlined with "===" or "---" is a heading too: in a Markdown file it is, in what the gateway writes
    it never is, and a line of dashes there is a rule (core's headingsOf)."""
    to = len(lines) if to is None else to

    def text(line: int) -> str:
        row = lines[line - 1]
        return row[:-1] if row.endswith("\r") else row

    headings: list[dict] = []
    # A fence is closed by its own character, at least as many of them, and nothing after: a ```` block holds ```
    # lines.
    fence = None
    # Front matter ("---" first, then YAML, then "---" or "..."), whose closing line would underline its last key as
    # a heading. A rule ("---") over a text is no front matter: its next line is no key.
    while start <= to and text(start).strip(_JS_TRIMMED) == "":
        start += 1
    if setext and start < to and text(start).strip(_JS_TRIMMED) == "---" and YAML_KEY.match(text(start + 1)):
        end = start + 1
        while end <= to and not FRONT_END.match(text(end)) and YAML_LINE.match(text(end)):
            end += 1
        if end <= to and FRONT_END.match(text(end)):
            start = end + 1
    paragraph = None
    block = False
    for line in range(start, to + 1):
        row = text(line)
        mark = FENCE.match(row)
        if mark and (fence is None or (mark.group(1)[0] == fence[0] and len(mark.group(1)) >= fence[1]
                                       and mark.group(2).strip(_JS_TRIMMED) == "")):
            fence = (mark.group(1)[0], len(mark.group(1))) if fence is None else None
            paragraph = None
            continue
        if fence is not None:
            continue
        found = ATX.match(row)
        if found:
            headings.append({"line": line, "level": len(found.group(1)), "title": found.group(2)})
            paragraph = None
            block = False
            continue
        under = SETEXT.match(row) if setext and paragraph is not None else None
        if under and paragraph is not None:
            title = " ".join(text(at).strip(_JS_TRIMMED) for at in range(paragraph, line))
            headings.append({"line": paragraph, "level": 1 if under.group(1)[0] == "=" else 2, "title": title})
            paragraph = None
            continue
        if row.strip(_JS_TRIMMED) == "":
            paragraph = None
            block = False
        elif BLOCK.match(row):
            paragraph = None
            block = True
        elif paragraph is None and not block and not INDENTED.match(row):
            paragraph = line
    return headings


def headings_in(path: str, kind: str, lines: list[str]) -> list[dict]:
    """A document's headings, where its text is Markdown: a Markdown file, one the gateway made Markdown of, and a
    text whose name says no kind (no ending, or .txt) whose every "#" line stands apart as Markdown writes a
    heading, a blank line under it. A Containerfile's, a Makefile's or a log's "# comment" has its line of code
    there, and was read as a heading. Code, tables and other endings have none. A repository's file stands in a
    fenced block under the engine's heading, so its headings are those inside it (core's headingsIn)."""
    # A rotated log (app.log.1) is a log: the numbers after its ending are no ending.
    name = ROTATED.sub("", path.split("/")[-1], count=1)
    suffix = ENDING.search(name)
    ending = suffix.group(1) if suffix else ""
    if kind == "repository_file":
        opener = next((index for index, row in enumerate(lines[:12]) if row.startswith("```")), -1)
        if opener < 0 or not CLOSER.match(lines[-1] if lines else "") or not MARKDOWN_ENDINGS.match(ending):
            return []
        return headings_of(lines, opener + 2, len(lines) - 1, True)
    if kind == "code" or kind in TABLE_TYPES:
        return []
    # The document's own text starts under the engine's heading ("# name", "", "Source type: …", "").
    body = next((index for index, row in enumerate(lines[:8]) if row.startswith("Source type: ")), -1) + 3
    if MARKDOWN_ENDINGS.match(ending):
        return headings_of(lines, max(2, body), len(lines), True)
    if WRITTEN_AS_MARKDOWN.match(ending):
        return headings_of(lines)
    if ending != "" and ending.lower() != "txt":
        return []
    # A build file, a dotfile and a script say their kind by their name or their first line, and are no Markdown.
    first = lines[max(0, body - 1)] if max(0, body - 1) < len(lines) else ""
    if NOT_MARKDOWN_NAMES.match(name) or name.startswith(".") or first.startswith("#!"):
        return []
    found = headings_of(lines)
    return found if all((lines[heading["line"]] if heading["line"] < len(lines) else "").strip(_JS_TRIMMED) == ""
                        for heading in found) else []


def cut_text(text: str, most: int) -> str:
    """A text cut to `most` characters with "…", never between the two halves of a character outside the basic
    plane, counted as the core counts them (core's cutText)."""
    wide = _wide(text)
    if (len(text) if wide is None else len(wide) // 2) <= most:
        return text
    end = most - 1 if _high(text, wide, most - 1) else most
    return (text[:end] if wide is None else wide[:2 * end].decode("utf-16-le", "surrogatepass")) + "…"


def title_of(heading: dict) -> str:
    """A heading's words as open names them."""
    return cut_text(heading["title"], HEADING_CHARS)


def headings_above(headings: list[dict], line: int) -> list[dict]:
    """The headings a line stands under, outermost first: the chapter, the agreement and the clause a paragraph is
    in. A heading stands under the ones above its own level (core's headingsAbove)."""
    above: list[dict] = []
    for heading in headings:
        if heading["line"] > line:
            break
        while above and above[-1]["level"] >= heading["level"]:
            above.pop()
        if heading["line"] < line:
            above.append(heading)
    return above


def section_at(read: dict, line: int) -> dict | None:
    """The section a line stands in, {first, last, at, leaf, own}: from the heading above it to the next heading of
    that level or higher; `leaf` when it has no sections of its own, `own` the last of its own lines (the one above
    its first section, or its last). None when the document has no heading above the line, or is known by its start
    only and the heading that ends the section is not (core's sectionAt)."""
    headings = read.get("headings") or []
    at = -1
    while at + 1 < len(headings) and headings[at + 1]["line"] <= line:
        at += 1
    if at < 0:
        return None
    heading = headings[at]
    after = next((entry for entry in headings[at + 1:] if entry["level"] <= heading["level"]), None)
    if after is None and not read["whole"]:
        return None
    last = len(read["lines"]) if after is None else after["line"] - 1
    leaf = at + 1 >= len(headings) or headings[at + 1]["level"] <= heading["level"]
    return {"first": heading["line"], "last": last, "at": at, "leaf": leaf,
            "own": last if leaf else headings[at + 1]["line"] - 1}


def names_of(named: list[dict]) -> str:
    """Headings by their words and line, as open names sections."""
    return "; ".join(f"{title_of(heading)} (line {heading['line']})" for heading in named)


def beside_line(headings: list[dict], at: int) -> list[str]:
    """The line that names the sections beside one: the other clauses of its agreement, the other parts of its
    chapter. Shown a clause alone, a reader does not see the addendum that changes it, eight headings further
    down. Among more than open names, only the one before it and the one after it, and how many there are: in a
    report of 500 small sections, 24 names were more than the section they stood beside. None when the section
    stands alone (core's besideLine)."""
    heading = headings[at]
    before: list[dict] = []
    after: list[dict] = []
    up = at - 1
    while up >= 0 and headings[up]["level"] >= heading["level"]:
        if headings[up]["level"] == heading["level"]:
            before.insert(0, headings[up])
        up -= 1
    down = at + 1
    while down < len(headings) and headings[down]["level"] >= heading["level"]:
        if headings[down]["level"] == heading["level"]:
            after.append(headings[down])
        down += 1
    count = len(before) + len(after)
    if count == 0:
        return []
    named = before[-1:] + after[:1] if count > BESIDE_SHOWN else before + after
    where = f"under line {headings[up]['line']}" if up >= 0 else "in this document"
    nearest = f" (the nearest {len(named)} of {count})" if len(named) < count else ""
    return [f"Beside it {where}{nearest}: {names_of(named)}."]


def sections_line(headings: list[dict], at: int) -> list[str]:
    """The line that names a section's own sections, for one too long to show whole: a binder's agreements, a
    chapter's clauses. Each is opened at its line. Its own are the headings right under it, whatever their levels
    (core's sectionsLine)."""
    own: list[dict] = []
    level = math.inf
    down = at + 1
    while down < len(headings) and headings[down]["level"] > headings[at]["level"]:
        if headings[down]["level"] <= level:
            level = headings[down]["level"]
            own.append(headings[down])
        down += 1
    named = own[:BESIDE_SHOWN]
    first = f" (the first {len(named)} of {len(own)})" if len(named) < len(own) else ""
    return [f"Its sections{first}: {names_of(named)}."]


def reading_note(item: dict) -> str:
    """How a ready collection is read, for a status line: its size, and whether a reply reads it whole or in
    parts; nothing when its size is unknown (core's readingNote)."""
    tokens = _count(item.get("logical_tokens"))
    if tokens is None:
        return ""
    if tokens > READ_PER_REPLY:
        return (f", {format_tokens(tokens)} tokens, read in parts: each reply reads what matches the conversation; "
                "contents, find and open reach the rest")
    return f", {format_tokens(tokens)} tokens, read whole"


def conversations_note(items: list[dict]) -> str | None:
    """What the agent is told about the conversations this one reads: the ready ones are in its context, the
    others not yet."""
    if not items:
        return None
    ready = [item for item in items if item["readiness"] == "ready"]
    pending = [item for item in items if item["readiness"] != "ready"]
    parts = []
    if ready:
        parts.append(f"Earlier conversations added here: {_names(ready)}. {ADDED_READ}")
    if pending:
        parts.append(f"Not in your context yet: {_names(pending)}; "
                     f"{'it arrives' if len(pending) == 1 else 'they arrive'} with a later reply.")
    return " ".join(parts)


def _memory_view(raw) -> dict | None:
    """The workspace memory from the account summary (core's memoryView): None when its size is not known."""
    if not isinstance(raw, dict):
        return None
    if raw.get("available") is False:
        return {"available": False}
    used, limit = _count(raw.get("used_tokens")), _count(raw.get("limit_tokens"))
    if used is None or not limit:
        return None
    history_limit, history = _count(raw.get("session_limit_tokens")), _count(raw.get("session_tokens"))
    return {"available": True, "used": used, "limit": limit, "full": raw.get("full") is True or used >= limit,
            "history": {"used": history, "limit": history_limit} if history_limit and history is not None else None}


# The longest pattern AdaptiqAI's search takes, in characters (the engine's MAX_PATTERN_CHARS; core's PATTERN_CHARS).
PATTERN_CHARS = 200


def _all_words(words: list[str]) -> tuple[str, list[str], list[str]]:
    """The search for lines that hold every word, in any order and any case, the words it searches and the words it
    leaves out (core's allWords): a lookahead for each word, from the line's start. Without the "^" the engine tries
    the lookaheads again from every character of a line that lacks a word, 1.1 s on a line of 60,000 characters, and
    refuses the pattern past the 2 s it gives a line (400 invalid_pattern, MiMo eval 2026-11-15); from the start
    alone the same lines take under a millisecond (measured in the engine's collection_search, 2026-11-16). Each
    order of the words one after another ("a.*b|b.*a") is past PATTERN_CHARS with three words of 27 letters in all.
    A query's 120 characters fit with up to 14 words; with more, the longest words that fit are searched, in the
    order they came. A word asked several times, in any case, is searched once."""
    def pattern(searched: list[str]) -> str:
        return "(?i)^" + "".join(f"(?=.*{_literal(word)})" for word in searched)

    asked = [word for at, word in enumerate(words) if word.lower() not in (other.lower() for other in words[:at])]
    searched = list(asked)
    while len(searched) > 1 and len(pattern(searched)) > PATTERN_CHARS:
        shortest = min(len(word) for word in searched)
        del searched[max(index for index, word in enumerate(searched) if len(word) == shortest)]
    return pattern(searched), searched, [word for word in asked if word not in searched]


def _fail(message: str):
    raise knowledge.KnowledgeError(message)


def _run(work) -> str:
    """A tool's answer: its sentences, or Hermes' tool error with the refusal's sentence. Passing trouble is said for
    the agent, and counted by the action of the guarded handler it runs in."""
    session, tool, action = getattr(_calling, "call", None) or ("", "", "")
    try:
        return FAILURES.run(session, tool, action, work)
    except knowledge.KnowledgeError as error:
        return json.dumps({"error": str(error)}, ensure_ascii=False)


def _connected() -> None:
    if core.key() is None:
        _fail(f"AdaptiqAI is not connected. Ask the person to sign in from {SIGN_IN}.")


def _session(session_id: str) -> str:
    if not session_id:
        _fail("This tool works inside a conversation.")
    return session_id


def _unreadable(item: dict, refused: str, error: knowledge.KnowledgeError,
                instead: str = "") -> knowledge.KnowledgeError:
    """A failed read of a collection's text never reads as "not there" (core's reading.ts). Passing trouble was
    asked again inside the call already, so the guard says not to call again in this turn; a refusal says what
    AdaptiqAI answered. Told to try again, an agent called contents and open nine times against a 403 for the file
    list, and the cause never reached the person."""
    if _passing(error) or error.status is None:
        return _OutOfReach(f"The text of {_quoted(item['name'])} could not be read just now; this does not mean "
                           "anything is missing.")
    answer = f" ({error.answer})" if error.answer else ""
    return knowledge.KnowledgeError(
        f"AdaptiqAI refused {refused} {_quoted(item['name'])}{answer}, and another try gets the same answer. This "
        f"does not mean anything is missing{instead}. Tell the person once, with that answer.")


def handlers(model_of, remember_of, set_remember, conversation_of=lambda session: session,
             preparing_of=lambda session: [], open_url=lambda url: False, elsewhere=lambda session: False,
             written_of=lambda session: None):
    """The knowledge and memory tools. `model_of(session)` is the model a conversation runs on; `remember_of` and
    `set_remember` read and set whether its new messages are saved; `conversation_of(session)` is the AdaptiqAI
    conversation a Hermes session belongs to (the one a compaction continued); `preparing_of(session)` lists the
    collections saved there that are used as soon as they are ready; `open_url(url)` opens a page in the person's
    browser, false where it can't; `elsewhere(session)` says a conversation's latest request went to another
    provider; `written_of(session)` is every message the person wrote in a conversation, oldest first, None when
    none is at hand, and raises when they could not be read."""
    # Where the next page of a search, an inventory or a document listing starts, by conversation.
    pages: OrderedDict[str, object] = OrderedDict()
    # The lines of the documents opened last, by collection, its version and document, and their characters.
    known: OrderedDict[str, dict] = OrderedDict()
    known_chars: dict[str, int] = {}
    # The documents of the last finds over many, least recently used first, with their characters.
    known_many: OrderedDict[str, tuple[dict, int]] = OrderedDict()
    many_chars = 0
    # The collections each earlier conversation added to another reads, by its id, and when that was read.
    theirs_read: OrderedDict[str, tuple[float, list[str]]] = OrderedDict()
    # What each conversation's reads returned last, and how many reads in a row returned nothing new.
    returned: OrderedDict[str, dict] = OrderedDict()
    lock = threading.Lock()

    def told(session: str, action: str, args: dict, result: str, scope: str = "") -> str:
        """A read's result, with a last line when it was asked again and nothing is new: the same read with the same
        result as before, for the second time in a row (core's told, plugins 2e335ea). A result alone gives no sign
        that it was there before; in a long test a model asked one find 117 times with its answer in front of it.
        The read still runs and all of its result is returned: what to do with it is left to the model. A read with
        other words, another document or page, or a changed result starts the count again. `scope` is what the read
        is of besides its arguments: the conversation a find_in_conversation reads."""
        now = time.monotonic()
        key = json_records.stringify([action, scope, *(args.get(name) if isinstance(args.get(name), str) else None
                                                       for name in ("name", "query", "document")),
                                      *(args.get(name) for name in ("line", "more", "fields", "sort", "sum", "group",
                                                                    "summary"))])
        digest = hashlib.sha256(result.encode("utf-8", "surrogatepass")).hexdigest()
        with lock:
            state = returned.get(session)
            if state is None or now - state["at"] > READS_IDLE:
                state = {"seen": OrderedDict(), "streak": 0, "at": now}
            state["at"] = now
            _keep(returned, session, state, READERS_KEPT)
            state["streak"] = state["streak"] + 1 if state["seen"].get(key) == digest else 0
            _keep(state["seen"], key, digest, READS_KEPT)
            streak = state["streak"]
        if streak < NOTHING_NEW_AFTER:
            return result
        return (f"{result}\nNothing new: this is the same read as before with the same result, and the last {streak} "
                "reads here returned only what they had returned already. Asking again returns the same. Answer from "
                "what you have, or read something different: other words, another document, or the next page.")

    def page_at(key: str, more) -> object:
        with lock:
            return pages.get(key) if more is True else None

    def page_next(key: str, value) -> None:
        with lock:
            if value is None:
                pages.pop(key, None)
            else:
                _keep(pages, key, value, PAGES_KEPT)

    def library(model: str, query: str | None, cursor, selected: frozenset) -> dict:
        """The library, or a plain failure: an unreadable list never reads as "nothing saved"."""
        try:
            return knowledge.library(model, query, cursor, selected)
        except knowledge.KnowledgeError as error:
            if not _passing(error):
                raise
            raise _OutOfReach("Saved knowledge could not be read just now; this does not mean there is none.") \
                from error

    def model_here(session: str, view: dict) -> str:
        model = view.get("model") or model_of(session)
        if not model:
            _fail("Send a message on an AdaptiqAI model first.")
        return model

    def named(model: str, view: dict, name: str, in_use: bool, selected: frozenset) -> dict:
        """The collection a name means: an exact name first (in use, then newest), else the only one whose name
        contains it."""
        wanted = same_name(name)
        if not wanted:
            _fail("Say which knowledge: use its name as search shows it.")
        here = [item for item in view["knowledge"] if same_name(item["name"]) == wanted]
        # Stopping names one used here, exactly: never another that only contains the name, and never one of two
        # with the same name without asking.
        if in_use:
            if len(here) > 1:
                _fail(f"Two collections used in this conversation are called {_quoted(name)}. Ask the person which "
                      f"one, or stop it from {STOP_KNOWLEDGE}.")
            if here:
                return here[0]
            _fail(f"No knowledge used in this conversation is called {_quoted(name)}. Nothing was changed.")
        page = library(model, name, None, selected)
        usable = [item for item in page["items"] if item["fits"] and item["kind"] == "collection"]
        exact = sorted((item for item in usable if same_name(item["name"]) == wanted),
                       key=lambda item: (item["selected"], item["at"]), reverse=True)
        if exact:
            return exact[0]
        partial = [item for item in usable if wanted in same_name(item["name"])]
        if len(partial) == 1:
            return partial[0]
        if partial:
            _fail(f"Several saved collections match {_quoted(name)}: {_names(partial[:6])}. Ask the person which one.")
        if any(not item["fits"] and same_name(item["name"]) == wanted for item in page["items"]):
            _fail(f"{_quoted(name)} is made for another model. The person can make it available here from "
                  f"{ADD_KNOWLEDGE}.")
        return _fail(f"No saved knowledge is called {_quoted(name)}. Search first and use the name exactly as it shows.")

    def knowledge_line(item: dict) -> str:
        if not item["fits"]:
            return (f"- {_quoted(item['name'])}: made for another model; the person can make it available here from "
                    f"{ADD_KNOWLEDGE}")
        return f"- {_quoted(item['name'])}: {_state(item)}{', used in this conversation' if item['selected'] else ''}"

    def described_line(item: dict) -> str:
        """A collection's line with its description: what an agent chooses by, since names alone left it guessing."""
        return knowledge_line(item) + (" — " + " ".join(item["preview"].split())[:200] if item.get("preview") else "")

    def usable_here(model: str | None, selected: frozenset) -> list[str]:
        """The collections the conversation's model can use, newest first, each with its description, up to
        COLLECTIONS_LISTED (core's usableHere): none when the library has none or could not be read."""
        if not model:
            return []
        try:
            page = library(model, "", None, selected)
        except Exception:  # noqa: BLE001 - what was asked is still said
            return []
        usable = [item for item in page["items"] if item["kind"] == "collection" and item["fits"]]
        more = len(usable) > COLLECTIONS_LISTED or page["has_more"]
        return [*map(described_line, usable[:COLLECTIONS_LISTED]), *(["More: search with no words."] if more else [])]

    def start(conversation: str, model: str, item: dict) -> str:
        """Starts using a collection here, or says why it can't be: what use answers."""
        if item["readiness"] == "failed":
            why = knowledge.failure(item["memory_id"], model)
            _fail(f"{_quoted(item['name'])} could not be built, so it cannot be used"
                  f"{'' if why is None else ': ' + build_failure(why)}.")
        if item["readiness"] != "ready":
            _fail(f"{_quoted(item['name'])} is {_state(item)}; it can be used once it is ready.")
        knowledge.use(conversation, model, item["memory_id"], True)
        return f"{_quoted(item['name'])} is now used in this conversation."

    def collections_of_conversations(view: dict) -> list[str]:
        """The collections the earlier conversations added to this one read and this one does not: they are not
        read here until it uses them (core's collectionsOfConversations). Told only that two earlier conversations
        were added, an agent asked about a shipment in the collection one of them read said it was not there
        (2026-11-08); told which collections they read, it used them and answered. Their names, each once, in the
        order the conversations were added; each conversation's are read at most once a minute, and one that can't
        be read now is left out."""
        here = {same_name(item["name"]) for item in view["knowledge"]}
        names: list[str] = []
        for conversation in view["conversations"]:
            conversation_id = conversation["conversation_id"]
            if conversation["readiness"] != "ready" or not isinstance(conversation_id, str) or not conversation_id:
                continue
            with lock:
                kept = theirs_read.get(conversation_id)
            if kept is None or time.monotonic() - kept[0] > THEIRS_FOR:
                read = knowledge.collections_of(conversation_id, view.get("model"))
                if read is None:
                    continue
                kept = (time.monotonic(), read)
                with lock:
                    _keep(theirs_read, conversation_id, kept, THEIRS_KEPT)
            for name in kept[1]:
                if same_name(name) not in here and all(same_name(entry) != same_name(name) for entry in names):
                    names.append(name)
        return names

    def used_here(view: dict, name: str) -> list[dict]:
        """The ready collections used here that a name means, or all of them."""
        used = ready_here(view["knowledge"])
        asked = name if name.strip() else None
        if not used:
            theirs = collections_of_conversations(view)
            raise _NotUsedHere(f"This conversation uses no ready collection. {their_collections(theirs)}" if theirs
                               else "This conversation uses no ready collection. Find one with search (with no words it lists them all) and use it first.",
                               asked, not theirs)
        if asked is None:
            return used
        wanted = same_name(name)
        found = ([item for item in used if same_name(item["name"]) == wanted]
                 or [item for item in used if wanted in same_name(item["name"])])
        if not found:
            raise _NotUsedHere(f"No collection used here is called {_quoted(name)}. Used here: {_names(used)}.",
                               asked, False)
        return found

    def listing(model: str, item: dict, offset: int) -> dict:
        try:
            return knowledge.files(model, item["memory_id"], offset)
        except knowledge.KnowledgeError as error:
            raise _unreadable(item, "to list the documents in", error,
                              ": find still searches all of its text") from error

    def status(session: str, view: dict) -> str:
        used = view["knowledge"]
        here = {item["memory_id"] for item in used}
        preparing = [item for item in preparing_of(session) if item["memory_id"] not in here]
        lines = []
        if not used and not preparing:
            lines.append("This conversation uses no saved knowledge.")
            # Told only "uses none", agents searched next, in 154 of 325 captured conversations (2026-11-14): the ones
            # there are follow.
            there = usable_here(view.get("model") or model_of(session), frozenset())
            if there:
                lines += ["The collections there are, newest first (contents, find and open with a name use it here):",
                          *there]
        else:
            parts = [f"{_quoted(item['name'])} ({_state(item)}{reading_note(item) if item['readiness'] == 'ready' else ''})"
                     for item in used]
            parts += [f"{_quoted(item['name'])} (getting ready; used by itself when ready)" for item in preparing]
            lines.append(f"This conversation uses: {', '.join(parts)}.")
        note = conversations_note(view["conversations"])
        if note:
            lines.append(note)
        theirs = collections_of_conversations(view)
        if theirs:
            lines.append(their_collections(theirs))
        if not remember_of(session):
            lines.append("This conversation is private: new messages are not saved.")
        return "\n".join(lines)

    def search(session: str, conversation: str, model: str, view: dict, args: dict, selected: frozenset) -> str:
        query = str(args.get("query") or "").strip()
        key = f"{session}\nsearch\n{query}"
        page = library(model, query, page_at(key, args.get("more")), selected)
        page_next(key, page["next_cursor"] if page["next_cursor"] and page["has_more"] else None)
        items = [item for item in page["items"] if item["kind"] == "collection"]
        if not items:
            if not query:
                return "No knowledge is saved yet."
            if args.get("more") is True:
                return f"No saved knowledge matches {_quoted(query)}."
            if view["stored"]:
                used = view["knowledge"]
            else:  # not stored yet: its first request is given the collections whose rule is always
                try:
                    used = knowledge.read_here(conversation, model)
                except knowledge.KnowledgeError:
                    used = []
            # A miss that names no collection left agents guessing names until they gave up (2026-11-14).
            try:
                there = [item for item in library(model, "", None, selected)["items"] if item["kind"] == "collection"]
            except Exception:  # noqa: BLE001 - the miss is still said
                there = []
            miss = search_miss(query, used)
            return "\n".join([miss, "The collections there are, newest first:", *map(described_line, there)]) if there else miss
        return "\n".join([f"Saved knowledge{f' matching {_quoted(query)}' if query else ''}, newest first:",
                          *map(knowledge_line, items), *(["More: search again with more: true."] if page["has_more"] else [])])

    def contents(session: str, model: str, view: dict, args: dict) -> str:
        """A page of each collection's documents (core's contents). Where the next page starts is kept with the
        listing's version and size: asked on after the last page, contents says so instead of starting again, and a
        collection that changed is listed again from its first document. The list is always asked afresh, as access
        and versions can change."""
        lines = []
        for item in used_here(view, str(args.get("name") or "")):
            key = f"{session}\ncontents\n{item['memory_id']}"
            held = page_at(key, args.get("more"))
            offset = held["offset"] if held else 0
            page = listing(model, item, offset)

            def version_of(page: dict) -> object:
                # A document's version is the listing's (core's page.files[0].version).
                return page["version"] if page["version"] is not None else item.get("changed")

            if held and offset > 0 and (held["version"] != version_of(page) or held["total"] != page["total"]):
                offset = 0
                page = listing(model, item, offset)
            page_next(key, {"offset": offset + len(page["documents"]) if page["next_offset"] is None
                            else page["next_offset"], "version": version_of(page), "total": page["total"]})
            if not page["documents"] and offset >= page["total"]:
                lines.append(f"{_quoted(item['name'])}: All {page['total']} documents were shown already. contents "
                             "without more: true starts again from the first.")
                continue
            lines.append(f"{_quoted(item['name'])}: {page['total']} documents{reading_note(item)}. "
                         f"Documents {offset + 1}-{offset + len(page['documents'])}:")
            lines += with_columns(model, item, page["documents"], series.contents_lines(
                page["documents"], lambda family, documents=page["documents"]: _kin_pattern(documents, family)))
            if page["next_offset"] is not None:
                lines.append("More: contents again with more: true.")
        return "\n".join(lines)

    def with_columns(model: str, item: dict, documents: list[dict], listed: list[str]) -> list[str]:
        """contents' lines, each table that leads a line with its columns (core's withColumns): agents opened a table
        only to see its columns before they could find in it (28 of 325 conversations, MiMo eval 2026-11-14). A
        series' line shows the columns of its first document. Columns already read for open or find are reused."""
        leads = [next((document for document in documents if DELIMITED_NAME.search(document["path"])
                       and (line.startswith(f"- {document['path']} (") or line.startswith(f"- {document['path']} to "))),
                      None) for line in listed]
        wanted = list({document["path"]: document for document in leads if document is not None}.values())[:COLUMN_READS]
        if not wanted:
            return listed
        try:
            sources = knowledge.source_list(model, item["memory_id"])
        except knowledge.KnowledgeError:
            return listed

        def columns_of(document: dict) -> list[str]:
            source = next((entry for entry in sources if entry["name"] == document["path"]), None)
            if source is None or source["type"] not in TABLE_TYPES:
                return []
            key = key_of(item, document)
            with lock:
                held = known_many.get(key)
                read = known.get(key) or (held[0] if held is not None else None)
            if read is not None and read.get("table"):
                return read["table"]["cells"]
            try:
                return columns_in(knowledge.source_text(model, item["memory_id"], source["id"])["text"])
            except knowledge.KnowledgeError:
                return []

        with ThreadPoolExecutor(max_workers=min(len(wanted), 4)) as pool:
            columns = dict(zip((document["path"] for document in wanted), pool.map(columns_of, wanted)))
        out = []
        for line, document in zip(listed, leads):
            cells = columns.get(document["path"]) if document is not None else None
            if not cells or len(cells) < 2:
                out.append(line)
                continue
            named = ", ".join(cells)
            shown = named[:600] + "…" if len(named) > 600 else named
            out.append(f"{line[:-1]}; columns: {shown})" if line.endswith(" lines)")
                       else f"{line} Columns of {document['path']}: {shown}.")
        return out

    def documents_of(model: str, item: dict) -> list[dict]:
        """The documents of a collection, up to twenty pages of its list (core's documentsOf)."""
        documents, offset = [], 0
        for _ in range(20):
            page = listing(model, item, offset)
            documents += page["documents"]
            offset = page["next_offset"]
            if offset is None:
                break
        return documents

    def read_all(model: str, item: dict, documents: list[dict]) -> list[dict | None]:
        """The lines of several documents, read sixteen at a time (core's listedIn), with one list of the collection's
        sources for all of them, asked for only when one is not kept. None for one that could not be read: when no
        more than sixteen could not, each is asked once more. The documents of a find over more than twelve are kept
        by their characters; those already kept move to the back of their shelves first."""
        many = len(documents) > LISTED_DOCUMENTS
        touch(item, documents, many)
        asked: list = []
        asking = threading.Lock()

        def sources() -> list[dict]:
            with asking:
                if not asked:
                    try:
                        asked.append(knowledge.source_list(model, item["memory_id"]))
                    except knowledge.KnowledgeError as error:
                        asked.append(error)
                if isinstance(asked[0], knowledge.KnowledgeError):
                    raise asked[0]
                return asked[0]

        def read(document: dict) -> dict | None:
            try:
                return lines_of(model, item, document, sources, many, keep=False)
            except knowledge.KnowledgeError:
                return None

        def read_each(chosen: list[dict]) -> list[dict | None]:
            if len(chosen) == 1:
                return [read(chosen[0])]
            with ThreadPoolExecutor(max_workers=min(len(chosen), READ_AT_ONCE)) as pool:
                return list(pool.map(read, chosen))

        reads = read_each(documents)
        order = list(range(len(documents)))
        # A read that failed once (a 503) is asked again before a listing of many gives up on all of them.
        failed = [index for index, lines in enumerate(reads) if lines is None]
        if failed and len(failed) <= READ_AT_ONCE:
            for index, lines in zip(failed, read_each([documents[index] for index in failed])):
                reads[index] = lines
            again = set(failed)
            order = [index for index in order if index not in again] + failed
        # The new reads are kept in the order the find lists them, as core keeps each as its read starts (those asked
        # again last). Kept as they finished, the small ones came first and were left out first, so later finds read
        # more of them again.
        with lock:
            for index in order:
                document, lines = documents[index], reads[index]
                key = key_of(item, document)
                if lines is not None and lines["whole"] and key not in known and key not in known_many:
                    shelve(key, document, lines, many)
        return reads

    def holders_of(model: str, item: dict, pattern: str) -> dict | None:
        """The first line with a pattern in each document of a collection that holds it, with the line after it,
        searched a document at a time, sixteen at a time (core's holdersOf): {matches, complete}. Not complete when
        a search failed, or when a longer path beside a document's (staff.json beside staff.js) took the lines its
        search gave: then no note may say the lines shown are the only ones. None past PROBE_DOCUMENTS documents,
        or when they can't be listed."""
        try:
            # The first page says how many there are: past PROBE_DOCUMENTS, no other page is read.
            first = knowledge.files(model, item["memory_id"], 0)
        except knowledge.KnowledgeError:
            return None
        documents = first["documents"]
        if not documents or first["total"] > PROBE_DOCUMENTS or len(documents) < first["total"]:
            return None
        complete = True

        def own(document: dict) -> list[dict]:
            nonlocal complete
            try:
                found, more = knowledge.first_lines(model, item["memory_id"], pattern, document["path"])
            except knowledge.KnowledgeError:
                complete = False
                return []
            # The search's path is a start: a longer path beside this one may answer first.
            mine = [match for match in found if match["path"] == document["path"]]
            if not mine and found and more:
                complete = False
            return mine[:1]

        with ThreadPoolExecutor(max_workers=min(len(documents), READ_AT_ONCE)) as pool:
            matches = [match for found in pool.map(own, documents) for match in found]
        return {"matches": matches, "complete": complete}

    def listed_in(session: str, model: str, used: list[dict], wanted: str, query: str, words: list[str],
                  more: bool, fields: list[str], sort: list[dict], sums: list[str],
                  group: str | None, summary: bool = False) -> str | list[str] | None:
        """Every line of the named documents that holds the words, each with its record, page by page (core's
        listedIn). The engine's search gives a find its first 25 lines and no more, with a line on each side: asked
        for the lots of one depot in four volumes, an agent saw neither the rest nor which lot a "Depot:" line
        belonged to, and made 47 finds in 26 minutes without an answer. Read from the documents themselves, nothing
        is cut off, and records of several documents (four volumes' payments) sort as one list: sorted a volume at a
        time, an agent joined four lists of 90 in its head and passed 180k prompt tokens. A sum is taken over every
        record found. The page; the documents to leave to the engine's search, when one of them can't be read whole
        (a PDF, a part of a repository) and they are no more than twelve; or None when no collection used here has
        documents by that name."""
        for item in used:
            try:
                documents = documents_of(model, item)
            except knowledge.KnowledgeError:
                documents = []
            files, how, unknown = named_documents(documents, wanted)
            if not files:
                continue
            if unknown:
                _fail(f"No document in {_quoted(item['name'])} is called {', '.join(map(_quoted, unknown))}. Name "
                      "each as contents shows it, with commas between.")
            chars = sum(document.get("chars") or 0 for document in files)
            if len(files) > LISTED_DOCUMENTS and (len(files) > LISTED_SMALL or chars > LISTED_CHARS):
                # The engine's search still finds lines in them, but no fields, order or sums.
                if how == "start" and not fields and not sort and not sums and group is None and not summary:
                    return None
                _fail(f"{_quoted(wanted)} names {len(files):,} documents in {_quoted(item['name'])}, "
                      f"{format_tokens(chars)} characters; find lists at most {LISTED_DOCUMENTS} together, or up to "
                      f"{LISTED_SMALL:,} small ones of {format_tokens(LISTED_CHARS)} characters in all. Name fewer, "
                      "with commas between, or more of their names.")
            reads = read_all(model, item, files)
            unread = sum(1 for read in reads if read is None or not read["whole"])
            if unread:
                if how == "start":
                    return None
                # The engine's search goes a document at a time: a few, never hundreds at once.
                if len(files) > LISTED_DOCUMENTS:
                    _fail(f"{unread:,} of the {len(files):,} documents {_quoted(wanted)} names in "
                          f"{_quoted(item['name'])} can't be read whole (part of a larger source, or not answering "
                          f"just now), and find leaves at most {LISTED_DOCUMENTS} such documents to the engine's "
                          "search. Name fewer, or find without document.")
                return [document["path"] for document in files]
            docs = [_listed(document["path"], read) for document, read in zip(files, reads)]
            return records_page({
                "title": _quoted(item["name"]),
                # Kept by the documents' versions: once they change, a find starts again from the first record.
                "key": "\n".join([session, "find", *(key_of(item, document) for document in files)]),
                "wanted": wanted, "documents": documents, "first_chars": files[0].get("chars") or 0,
                "again": "the same words, document and more: true",
                "miss": lambda where, words: f"No line or record of {where} in {_quoted(item['name'])} holds "
                                             f"{_quoted(words)}.",
                "closing": FOUND}, docs, query, words, more, fields, sort, sums, group, summary)
        if "," in wanted:
            _fail(f"No collection used here has the documents {_quoted(wanted)}. Name each as contents shows it, with "
                  "commas between.")
        return None

    def records_page(scope: dict, docs: list[dict], query: str, words: list[str], more: bool, fields: list[str],
                     sort: list[dict], sums: list[str], group: str | None, summary: bool = False) -> str:
        """A page of the records of `docs` that hold the words, listed, sorted, summed or grouped (core's
        listRecords): a collection's documents, or a conversation's own messages. `scope` says where (core's
        ListingScope): the page's `title`, the `key` its pages are kept under, the documents a find named
        (`wanted`), the collection's `documents` and the first one's characters (`first_chars`), how to ask for more
        (`again`), the sentence when nothing holds the words (`miss(where, words)`), the last line (`closing`) and,
        for a conversation, what a summary says it read (`coverage`). With `summary`, the count and sums of all of
        them and what was read, without the records."""
        lowered = [word.lower() for word in words]
        hits = [_holding(doc, [query.lower()]) for doc in docs]
        if not any(hits) and len(words) > 1:
            hits = [_holding(doc, lowered) for doc in docs]
        # All the words in one record, on different lines of it: a lot's depot, its type and its date.
        by_record = not any(hits) and len(words) > 1
        if by_record:
            # A table's row is one line, as is a conversation's record: a record of words on different lines is a
            # text's or a JSON object's.
            hits = [[index for index, found in enumerate(doc["objects"])
                     if all(word in found["search"] for word in lowered)] if "objects" in doc
                    else [hit for hit in holding_all(doc["lines"], lowered, doc["json"]) if hit not in doc["rows"]]
                    if doc["table"] is None and not doc.get("line_records") else [] for doc in docs]
        where = (f"the {len(docs)} documents {_quoted(scope['wanted'])} names" if len(docs) > LISTED_DOCUMENTS
                 else ", ".join(doc["path"] for doc in docs))
        key = "\n".join([scope["key"], query.lower(), *(f"{entry['field']}\n{entry['descending']}" for entry in sort)])
        total = sum(len(found) for found in hits)

        def coverage(count) -> list[str]:
            # The documents read, those without a match too: their citations are bounded apart from the calculation,
            # which covers them all.
            cited: list[str] = []
            chars = 0
            for doc in docs:
                # A conversation read in part was not read in full: a count over the first 2,000 events of 2,300
                # said so (core's listRecords).
                line = (f"- {doc['path']}:1-{len(doc['lines'])}: {count(doc)} matching records"
                        f"{', in what was read of it' if doc.get('partial') else '; read in full'}.")
                if chars + len(line) > knowledge.RECORD_CHARS:
                    break
                cited.append(line)
                chars += len(line) + 1
            return [scope.get("coverage") or f"Sources scanned: {len(docs)} selected documents in {scope['title']}. "
                    "This scope does not include other documents or collections.", *cited,
                    *([] if len(cited) == len(docs) else [
                        f"Source citations shown for {len(cited)} of {len(docs)} documents; the calculation includes "
                        f"all {len(docs)}. For the remaining citations, request a smaller document selection."])]

        if not total:
            page_next(key, None)
            if summary:
                return "\n".join([f"Complete scan for {_quoted(query)}. Matching records: 0.",
                                  *([f"No matching records; sums of {', '.join(map(_quoted, sums))} cannot be "
                                     "evaluated."] if sums else []), *coverage(lambda doc: 0)])
            return "\n".join([scope["miss"](where, query), NOTHING_FOUND])
        records = _records(docs, hits, query)

        def values_of(field: str) -> list[tuple]:
            # By record, not by place: sorting reorders the records.
            values = []
            for entry in records:
                typed = None if entry["typed"] is None else entry["typed"](field)
                values.append((entry["doc"], entry["table"], entry["read"](field) if entry["typed"] is None
                               else None if typed is None else typed[0], None if typed is None else typed[1]))
            return values

        def places_of(field: str) -> list:
            """Where each record's value of a field stands in its document, for a JSON value (core's valueAt): an
            entry and a part of it that the words found too read the same value."""
            places = []
            for entry in records:
                typed = None if entry["typed"] is None else entry["typed"](field)
                places.append(None if typed is None else f"{id(entry['doc'])} {typed[2]}")
            return places

        # A field no record has is said, with the fields a record has: sorted by a name it does not know, find kept
        # the document's order.
        missing = [field for field in dict.fromkeys([*fields, *(entry["field"] for entry in sort), *sums,
                                                     *([group] if group is not None else [])])
                   if all(value is None for _, _, value, _ in values_of(field))]
        head = records[0]
        span = (head["record"]["start"], head["record"]["end"]) if head["record"] else (head["hit"], head["hit"])
        named = head["names"] if "names" in head else [name for name, _ in pairs_of(*span, _text_of(head["doc"]),
                                                                                     head["table"])]
        minus = " (a-b is the days between two dates, the minutes between two times, or one number less another)"
        notes = ([f"No record found here has {' or '.join(map(_quoted, missing))}"
                  f"{minus if any('-' in field for field in missing) else ''}. The fields of the first one: "
                  f"{', '.join(named) if named else 'none find can name'}."] if missing else [])
        if sort:
            records = sorted_records(records, sort, values_of)
        totals = []
        # A summary says how many records each sum leaves out, and why.
        numeric = []
        for field in sums:
            read = values_of(field)
            once = added_once(places_of(field))
            values = [value for value, kept in zip(read, once) if kept]
            numbers = [number for number, kept in zip(numbers_of(read), once) if kept]
            # The records that read a value another of them gave already: it is in the sum once.
            again, base = len(read) - len(values), len(values)
            shared = ("" if not again else f"; {again} more {'record reads' if again == 1 else 'records read'} these "
                      "same values, in an object they hold or are a part of, added once")
            found = [number for number in numbers if number is not None]
            unread = [value for (_, _, value, _), number in zip(values, numbers) if value is not None and number is None]
            if summary and len(found) < base:
                absent = sum(1 for _, _, value, _ in values if value is None)
                nulls = sum(1 for _, _, _, kind in values if kind == "null")
                numeric.append(f"Numeric coverage of {field}: {len(found)} numeric; {absent} missing; {nulls} null; "
                               f"{base - len(found) - absent - nulls} nonnumeric or unreadable. "
                               f"{base - len(found)} of {base} records excluded from this sum; the full "
                               "total is unavailable.")
            if not found:
                totals.append(f"Sum of {field}: none of the {base} records has a number for it"
                              f"{f' ({_quoted(unread[0])} is not one find can read)' if unread else ''}.")
                continue
            over = (f"all {base} records" if len(found) == base
                    else f"the {len(found)} of {base} records that have a number for it")
            totals.append(f"Sum of {field}: {sum_text(found)}, over {over}"
                          + (f"; {len(unread)} could not be read as numbers, such as {_quoted(unread[0])}" if unread
                             else "") + f"{shared}.")
        # Objects found within one another can start on one line: then they are counted as records.
        shared = len({(entry["doc"]["path"], entry["hit"]) for entry in records}) < len(records)
        found = (f"{len(records)} records that hold every word of {_quoted(query)}, on different lines," if by_record
                 else f"{len(records)} records with {_quoted(query)}" if shared
                 else f"{total} lines with {_quoted(query)}" if total == len(records)
                 else f"{total} lines in {len(records)} records with {_quoted(query)}")
        each = (docs[0]["path"] if len(docs) == 1
                else f"{len(docs)} documents (records in {len({id(entry['doc']) for entry in records})} of them)"
                if len(docs) > LISTED_DOCUMENTS
                else "{} documents ({})".format(len(docs), ", ".join(
                    f"{sum(1 for entry in records if entry['doc'] is doc)} in {doc['path']}" for doc in docs)))

        def matching(doc: dict) -> int:
            return sum(1 for entry in records if entry["doc"] is doc)

        if summary and group is None:
            return "\n".join([f"Complete scan for {_quoted(query)}. Matching records: {len(records)}.", *notes,
                              *totals, *numeric, *coverage(matching),
                              "The count and sums above already cover every matching record in this scope. No record "
                              "pagination is needed to calculate them. Numeric omissions, if any, are stated above. "
                              "Individual records are omitted; find without summary returns them for inspection."])
        if group is not None:
            return groups_page(scope["title"], key, records, values_of, places_of, [*notes, *totals, *numeric,
                                                                         *(coverage(matching) if summary else [])],
                               f"{found} in {each}", more, sort, sums, group)
        first = min(page_at(key, more) or 0, len(records))
        if first == len(records):
            # Kept at the end: asked on again, find says so again instead of starting from the first record.
            page_next(key, first)
            return (f"All {len(records)} records with {_quoted(query)} in {where} were shown already. find without "
                    "more: true starts again from the first.")
        body: list[str] = []
        # The tables whose columns the page shows, once each, before their first row.
        shown_tables: set[int] = set()
        room = knowledge.RECORD_CHARS
        after = first
        while after < len(records):
            entry = records[after]
            doc, hit, around, table = entry["doc"], entry["hit"], entry["record"], entry["table"]
            text = _text_of(doc)
            # A JSON record that is part of its line is shown as itself, never as the line (core's listRecords): one
            # node of a network model written on one line of 148,362 characters was never shown (2026-11-14).
            found_object = entry.get("object")
            alone = (json.dumps(found_object["value"], ensure_ascii=False, separators=(",", ":"))
                     if found_object is not None and found_object["start"] == found_object["end"] else None)
            matched = alone if alone is not None and len(alone) < len(text(hit).strip()) else text(hit)
            rows = [field_line(doc["path"], hit, around, text, fields, table, entry["read"]) if fields
                    else f"- {doc['path']}:{hit}: {f'{matched[:LINE_CHARS]}…' if len(matched) > LINE_CHARS else matched}"]
            if table is not None and not fields and id(table) not in shown_tables:
                shown_tables.add(id(table))
                rows.insert(0, columns_line(table) if len(docs) == 1 else f"- {doc['path']}: {columns_line(table)}")
            # A record of one line is the line found: it is not shown twice. A long one is shown as far as a page
            # holds it, its long lines cut.
            if (around is not None and not fields and table is None
                    and not (around["start"] == hit and around["end"] == hit)):
                end, chars, shortened, excerpt = around["start"] - 1, 0, False, []
                for line in range(around["start"], min(around["end"], around["start"] + record.RECORD_LINES - 1) + 1):
                    raw = text(line)
                    row = f"{raw[:LINE_CHARS]}…" if len(raw) > LINE_CHARS else raw
                    if chars + len(row) + 20 > knowledge.RECORD_CHARS - LINE_CHARS - 500:
                        break
                    chars += len(row) + 20
                    shortened = shortened or raw != row
                    end = line
                    if row:
                        excerpt.append(f"  {line}: {row}")
                whole = around["whole"] and end == around["end"] and not shortened
                rows.append(f"  The whole record, lines {around['start']}-{end}:" if whole
                            else f"  Lines {around['start']}-{end} of its record (long lines may be shortened):")
                rows += excerpt
            size = sum(len(row) + 1 for row in rows)
            if after > first and (size > room or (sort and after - first >= SORTED_PAGE)):
                break
            room -= size
            body += rows
            after += 1
        page_next(key, after)
        dates = _dates_in(body[0] if body else "") if fields and not any("-" in field for field in fields) else []
        between = ([f'Days between two of these dates: add a field such as "{dates[1]}-{dates[0]}", and each record '
                    f'shows them as a number; sort: "{dates[1]}-{dates[0]} desc" puts the largest first.']
                   if len(dates) >= 2 else [])
        kin = ([entry for entry in scope["documents"] if entry["path"] != docs[0]["path"]
                and _alike(entry["path"]) == _alike(docs[0]["path"])]
               if len(docs) == 1 and first == 0 and re.search(r"[0-9]", docs[0]["path"]) else [])
        family = [docs[0]["path"], *(entry["path"] for entry in kin)]
        # Too many to name one by one, but few and small enough to list together: a pattern names them, if one names
        # exactly them.
        pattern = (_kin_pattern(scope["documents"], family) if LISTED_DOCUMENTS <= len(kin) < LISTED_SMALL
                   and sum((entry.get("chars") or 0 for entry in kin), scope["first_chars"]) <= LISTED_CHARS
                   else None)
        such_as = f"Documents of the same kind: {len(kin)} more, such as {', '.join(family[1:4])}."
        same_kind = ([] if not kin
                     else [f"{such_as} To find in all {len(kin) + 1} at once, sorted and summed as one list, name "
                           f"them by {'these patterns' if ', ' in pattern else 'a pattern'}: document: \"{pattern}\"."]
                     if pattern is not None
                     else [f"Documents of the same kind: {', '.join(family[1:])}. To find in all of them at once, "
                           f'sorted and summed as one list, name them together: document: "{", ".join(family)}".']
                     if len(kin) < LISTED_DOCUMENTS
                     else [such_as])
        columns = ": a table's columns" if any(entry["table"] is not None for entry in records) else ""
        sorted_as = f", sorted by {_sorted_by(sort)}{', all documents together' if len(docs) > 1 else ''}" if sort else ""
        return "\n".join([
            f"{scope['title']}: {found} in {each}, all of them. Records {first + 1}-{after} of {len(records)}"
            f"{', as ' + ', '.join(fields) if fields else ''}{sorted_as}:",
            *notes, *totals,
            *(["The sums already include every matching record, including records not displayed below. More only "
               "pages individual records; it does not add to these totals."] if sums else []),
            *body,
            (f"More: find again with {scope['again']} for records {after + 1}-{len(records)}."
             + (f" To go through many records at once, name the fields you need (fields{columns}), and each record "
                "takes one line." if not fields and len(records) - after > 40 else ""))
            if after < len(records) else "That is the last of them.",
            *between,
            *same_kind,
            scope["closing"],
        ])

    def groups_page(title: str, key: str, records: list[dict], values_of, places_of, notes: list[str], found: str,
                    more: bool, sort: list[dict], sums: list[str], group: str) -> str:
        """One line a value of the field instead of the records (core's listedIn): how many records have it, and
        their sums, the largest sum first, or most records first without sum. Asked for the three largest customers
        of a year's invoices, a model would otherwise add up 2,400 rows a customer at a time. `notes` go before the
        groups: fields no record has, the sums over all the records, and with summary what they leave out and the
        documents read."""
        # A JSON value is grouped with values of its kind: the text "1" and the number 1 are two groups, and a text
        # that reads like another group's value is shown in quotes. A record without the field is in the group "-".
        members: dict[str, dict] = {}
        for index, (_, _, value, kind) in enumerate(values_of(group)):
            at = "missing" if value is None else f"{kind or 'string'}:{value}"
            if at in members:
                members[at]["indexes"].append(index)
            else:
                members[at] = {"value": "-" if value is None else value, "kind": kind, "missing": value is None,
                               "indexes": [index]}
        labels: dict[str, int] = {}
        for member in members.values():
            labels[member["value"]] = labels.get(member["value"], 0) + 1
        read = [(numbers_of(values_of(field)), places_of(field)) for field in sums]
        by = sort[0]["field"].lower() if sort else None
        which = (0 if sums else -1) if by is None else next(
            (index for index, field in enumerate(sums) if field.lower() == by), -1)
        descending = sort[0]["descending"] if sort else True
        groups = []
        for member in members.values():
            value, indexes = member["value"], member["indexes"]
            if not member["missing"] and member["kind"] in ("string", None) and labels[value] > 1:
                value = json_records.stringify(value)
            # A group's sum adds a value once, as the sum over all the records does.
            summed = [[numbers[index] for index, kept in zip(indexes, added_once([places[index] for index in indexes]))
                       if kept and numbers[index] is not None] for numbers, places in read]
            # An entry and the parts of it the words found too are counted once (core's distinct): a group of a
            # technician's two entries said "4 records", the first one's details and part beside it (2026-11-16).
            count = len({records[index].get("outermost", index) for index in indexes})
            amount = float(count)
            if which >= 0:
                amount = 0.0
                for number in summed[which]:
                    amount += number["value"]
            groups.append({"value": value, "indexes": indexes, "count": count, "totals": summed, "amount": amount,
                           "order": _collation(value)})

        def compare(a: dict, b: dict) -> int:
            by_amount = b["amount"] - a["amount"] if descending else a["amount"] - b["amount"]
            # Two sums past the largest number tie, as in JavaScript (NaN), and go by their values.
            if by_amount and not math.isnan(by_amount):
                return 1 if by_amount > 0 else -1
            return (a["order"] > b["order"]) - (a["order"] < b["order"])

        groups.sort(key=functools.cmp_to_key(compare))
        group_key = f"{key}\ngroup\n{group.lower()}\n" + "\n".join(sums)
        first = min(page_at(group_key, more) or 0, len(groups))
        if first == len(groups):
            page_next(group_key, first)
            return (f"All {len(groups)} groups by {group} were shown already. find without more: true starts again "
                    "from the first.")
        page = groups[first:first + GROUPS_PAGE]
        page_next(group_key, first + len(page))
        order = "most records first" if which < 0 else f"largest sum of {sums[which]} first"
        if not descending:
            order = order.replace("most", "fewest", 1).replace("largest", "smallest", 1)

        def line(row: dict) -> str:
            count = row["count"]
            return (f"- {group}={row['value']}: {count} {'record' if count == 1 else 'records'}" + "".join(
                f"; {field} = {sum_text(row['totals'][index], False) if row['totals'][index] else '-'}"
                for index, field in enumerate(sums)))

        return "\n".join([
            f"{title}: {found}, all of them, in {len(groups)} groups by {group}, {order}. Groups "
            f"{first + 1}-{first + len(page)} of {len(groups)}:",
            *notes, *map(line, page),
            f"More: find again with the same words, document, group and more: true for groups "
            f"{first + len(page) + 1}-{len(groups)}." if first + len(page) < len(groups) else "That is the last of them.",
            "The counts and sums already include all matching records; more only pages the group list. For an "
            "optional inspection of one group's records, find with its value among the words, without group or "
            "summary.",
        ])

    def find_in_conversation(session: str, args: dict) -> str:
        """Every line the person wrote in this conversation that holds the words, listed, sorted, summed or grouped
        as find does it in documents (core's findInConversation, plugins #121). A conversation read in parts shows a
        reply what matches it, never all of it: asked for the fines over 60 operations notes, MiMo added up the
        latest one alone. A failed read is said as one, never as a conversation that holds nothing."""
        try:
            written = written_of(session)
        except Exception:  # noqa: BLE001 - whatever failed, the messages were not read
            _fail(CONVERSATION_UNREAD)
        if written is None:
            _fail(CONVERSATION_UNHELD)
        lines = written_lines(written)
        query = str(args.get("query") or "").strip()[:120]
        if not query:
            _fail(f"Say which words to find in {THIS_CONVERSATION}.")
        if not lines:
            _fail("This conversation has no messages to find in yet.")
        # A version of the lines themselves: a message edited in place changes it, which a count would miss.
        version = hashlib.sha256(json_records.stringify(lines).encode("utf-8", "surrogatepass")).hexdigest()
        rows = markdown_rows(lines)
        heads = {line for header in rows.values() for line in (header["line"], header["line"] + 1)}
        doc = {"path": THIS_CONVERSATION, "lines": lines, "lower": [line.lower() for line in lines], "table": None,
               "rows": rows, "heads": heads, "json": False, "line_records": True}
        return records_page({
            "title": "This conversation", "coverage": CONVERSATION_SCANNED,
            "key": "\n".join([session, "conversation", version, THIS_CONVERSATION]),
            "wanted": "", "documents": [], "first_chars": 0, "again": "the same words and more: true",
            "miss": lambda where, words: f"No line of {THIS_CONVERSATION} holds {_quoted(words)}.",
            "closing": CONVERSATION_FOUND}, [doc], query, query.split(), args.get("more") is True,
            fields_of(args.get("fields")), sort_of(args.get("sort")), names_in(args.get("sum")),
            group_of(args.get("group")), args.get("summary") is True)

    def conversation_doc(name: str, lines: list[str], partial: bool = False) -> dict:
        """A conversation's lines as find lists them: each line a record. `partial` when the lines are not all of it
        (a long one read in part, a message cut): a summary then never says it was read in full."""
        rows = markdown_rows(lines)
        heads = {line for header in rows.values() for line in (header["line"], header["line"] + 1)}
        return {"path": name, "lines": lines, "lower": [line.lower() for line in lines], "table": None, "rows": rows,
                "heads": heads, "json": False, "line_records": True, "partial": partial}

    def person_whole(conversation_id: str, read: dict, model: str | None) -> dict:
        """A conversation's timeline with each message of the person's that it cut read whole (core's personWhole).
        Cut at 4,000 characters, the part number further into a message of about 47 KB was never searched (e11, MiMo
        eval, 2026-11-15). A message that can't be read whole stays cut, and is said to be."""
        def whole(event: dict) -> dict:
            if event["role"] != "user" or not event["truncated"] or event["id"] is None:
                return event
            try:
                return {**event, "text": knowledge.message(conversation_id, event["id"], model), "truncated": False}
            except Exception:  # noqa: BLE001 - whatever failed, the message stays cut and is said to be
                return event

        if not any(event["role"] == "user" and event["truncated"] for event in read["events"]):
            return read
        with ThreadPoolExecutor(max_workers=READ_AT_ONCE) as pool:
            return {**read, "events": list(pool.map(whole, read["events"]))}

    def find_in_linked(session: str, which: str, args: dict) -> str:
        """find_in_conversation in a conversation added here, by its name, or in this one and every added one
        together ("all"), as one list (core's findInLinked): what the person wrote in an added one is read from
        AdaptiqAI's timeline of it, a message the timeline cut read whole. Only a conversation added here is read;
        what is still cut or left out is said under the page's first line, so a miss there never reads as not
        written."""
        conversation = conversation_of(session)
        view = knowledge.conversation_view(conversation, model_of(session))
        added = [item for item in view["conversations"] if item["conversation_id"] != conversation]
        # "all", unless a conversation added here is called so, which is then the one meant (core's meansAll).
        everything = same_name(which) == "all" and not any(same_name(item["name"]) == "all" for item in added)
        if everything:
            chosen = added
        else:
            if not added:
                _fail(f"No conversation is added here, so none is called {_quoted(which)}. find_in_conversation "
                      "without conversation finds in this one.")
            wanted = same_name(which)
            exact = [item for item in added if same_name(item["name"]) == wanted]
            partial = [item for item in added if wanted and wanted in same_name(item["name"])]
            if len(exact) > 1:
                _fail(f"{len(exact)} conversations added here are called {_quoted(which)}. Use \"all\" to find in "
                      "them together.")
            if not exact and len(partial) > 1:
                _fail(f"Several conversations added here match {_quoted(which)}: {_names(partial)}. Name one as status "
                      "shows it.")
            if not exact and not partial:
                _fail(f"No conversation added here is called {_quoted(which)}. Added here: {_names(added)}.")
            chosen = exact or partial
        docs, versions, notes = [], [], []
        if everything:
            try:
                written = written_of(session)
            except Exception:  # noqa: BLE001 - whatever failed, the messages were not read
                _fail(CONVERSATION_UNREAD)
            if written is None:
                _fail(CONVERSATION_UNHELD)
            own = written_lines(written)
            docs.append(conversation_doc(THIS_CONVERSATION, own))
            versions.append(hashlib.sha256(json_records.stringify(own).encode("utf-8", "surrogatepass")).hexdigest())
            if not added:
                notes.append("No other conversation is added here, so this is this conversation alone.")
        for item in chosen:
            name = _quoted(item["name"])
            try:
                read = person_whole(item["conversation_id"], knowledge.timeline(item["conversation_id"], model_of(session)),
                                    model_of(session))
            except knowledge.KnowledgeError as error:
                notes.append(f"{name} could not be read just now; this does not mean it holds nothing. Do not search "
                             "it again in this turn: tell the person it could not be read right now, and that they "
                             "can ask again shortly." if error.passing else
                             f"AdaptiqAI refused to read {name}{f' ({error.answer})' if error.answer else ''}. This "
                             "does not mean it holds nothing; tell the person once, with that answer.")
                continue
            said = [event for event in read["events"] if event["role"] == "user"]
            cut = [event for event in said if event["truncated"]]
            if cut:
                shown = max(len(event["text"]) for event in cut)
                longest = max(event["bytes"] or 0 for event in cut)
                notes.append(f"{name}: {len(cut)} of the person's {len(said)} messages there come cut by AdaptiqAI at "
                             f"{shown:,} characters{f' (the longest has {longest:,} bytes)' if longest > shown else ''}"
                             ", so what stands past that was not searched: a miss there does not mean it was not "
                             "written.")
            events = len(read["events"])
            of = f" of {read['count']}" if read["count"] is not None and read["count"] > events else ""
            if read["left"] == "after":
                notes.append(f"{name}: its first {events}{of} messages were read, so what was said after them was not "
                             "searched.")
            if read["left"] == "before":
                notes.append(f"{name}: only its latest {events}{of} messages could be read, so what was said before "
                             "them was not searched.")
            docs.append(conversation_doc(f"{name} (added)", written_lines([event["text"] for event in said]),
                                         bool(cut) or read["left"] is not None))
            last = read["latest"] if read["latest"] is not None else (read["events"][-1]["seq"] if read["events"] else -1)
            versions.append(f"{item['conversation_id']} {last} "
                            f"{read['count'] if read['count'] is not None else len(read['events'])}")
        if not everything and not docs:
            _fail("\n".join(notes))
        page = conversations_page(session, docs, "\n".join(versions), args)
        if not notes:
            return page
        first, *rest = page.split("\n")
        return "\n".join([first, *notes, *rest])

    def with_unsearched(session: str, page: str) -> str:
        """A find in this conversation alone, with the conversations added here that it did not search named under
        its first line (core's withUnsearched). Told "1 lines …, all of them", their own question, agents said the
        meeting was never shared here, or counted one mention (MiMo eval, 2026-11-15)."""
        try:
            conversation = conversation_of(session)
            view = knowledge.conversation_view(conversation, model_of(session))
        except knowledge.KnowledgeError:
            return page
        note = unsearched_note([item["name"] for item in view["conversations"]
                                if item["conversation_id"] != conversation])
        if not note:
            return page
        first, *rest = page.split("\n")
        return "\n".join([first, *note, *rest])

    def conversations_page(session: str, docs: list[dict], version: str, args: dict) -> str:
        """One page of every line of the conversations given that holds the words, as one list (core's
        listConversations)."""
        one = len(docs) == 1
        alone = one and docs[0]["path"] == THIS_CONVERSATION
        what = docs[0]["path"] if one else "these conversations"
        title = what[:1].upper() + what[1:]
        query = str(args.get("query") or "").strip()[:120]
        if not query:
            _fail(f"Say which words to find in {what}.")
        if all(not doc["lines"] for doc in docs):
            _fail(f"{title} {'has' if one else 'have'} no messages to find in yet.")
        return records_page({
            "title": title,
            # An added conversation's lines are AdaptiqAI's, not the harness's own: where from is not said of them.
            "coverage": CONVERSATION_SCANNED if alone else (f"Sources scanned: {what if one else f'{len(docs)} conversations'}"
                                                             ". Other conversations and collections are not included."),
            "key": "\n".join([session, "conversation", version, *(doc["path"] for doc in docs)]),
            "wanted": "", "documents": [], "first_chars": 0,
            "again": "the same words and more: true" if alone else "the same words, conversation and more: true",
            "miss": lambda where, words: f"No line of {what} holds {_quoted(words)}.",
            "closing": CONVERSATION_FOUND if alone else f"These are the exact lines of {what}."},
            docs, query, query.split(), args.get("more") is True, fields_of(args.get("fields")), sort_of(args.get("sort")),
            names_in(args.get("sum")), group_of(args.get("group")), args.get("summary") is True)

    def find(session: str, model: str, view: dict, args: dict) -> str:
        query = str(args.get("query") or "").strip()[:120]
        document = args.get("document") if isinstance(args.get("document"), str) else None
        named = document is not None and document.strip() != ""
        # Asked for every record of a document with no words, an agent got only "Say which words" and had to guess
        # one.
        if not query:
            _fail("Say which words to find. To go through every record of a document, find a word or a field name "
                  "each record holds (with fields, sort and sum as needed)." if named else "Say which words to find.")
        words = query.split()
        used = used_here(view, str(args.get("name") or ""))
        fields, sort, sums = fields_of(args.get("fields")), sort_of(args.get("sort")), names_in(args.get("sum"))
        group = group_of(args.get("group"))
        summary = args.get("summary") is True
        paths: list[str | None] = [document]
        if named:
            listed = listed_in(session, model, used, document.strip(), query, words, args.get("more") is True,
                               fields, sort, sums, group, summary)
            if isinstance(listed, str):
                return listed
            if listed is not None:
                paths = listed
        # The engine's search gives at most its first lines: never a complete count or sum.
        if summary:
            _fail("summary requires named documents that find can read in full. No complete count or sum was "
                  "computed. Use contents to select readable documents; an engine search excerpt cannot establish a "
                  "complete total.")
        lines = []
        # The engine's search gives lines: it knows no fields, order or sums.
        if fields or sort or sums or group is not None:
            lines.append(f"fields, sort, sum and group work on documents find reads whole; {document.strip()} is "
                         "searched by the engine instead, so these are its results, as lines." if named else
                         "fields, sort, sum and group work when document names the documents to look in; without it, "
                         "these are the engine's search results, as lines.")
        room = knowledge.RECORD_CHARS
        found_any = False
        # A miss that shows where an id stands alone ends on those lines, not on "nothing was found".
        hinted = False
        for item in used:
            def lookup(pattern: str, item: dict = item) -> dict:
                found = {"matches": [], "truncated": False}
                for at in paths:
                    try:
                        one = knowledge.search(model, item["memory_id"], pattern, at)
                    except knowledge.KnowledgeError as error:
                        raise _unreadable(item, "to search", error) from error
                    found["matches"] += one["matches"]
                    found["truncated"] = found["truncated"] or one["truncated"]
                return found

            pattern = f"(?i){_literal(query)}"
            # A phrase that is longer than the pattern AdaptiqAI's search takes, once its marks are escaped, is
            # refused (400, with 77 marks in a query of 120 characters): its words are searched instead.
            found = (lookup(pattern) if len(words) == 1 or len(pattern) <= PATTERN_CHARS
                     else {"matches": [], "truncated": False})
            # The words the lines found hold when AdaptiqAI's search takes only the longest of them at once, and
            # those it left out.
            searched, left = words, []
            if not found["matches"] and len(words) > 1:
                pattern, searched, left = _all_words(words)
                found = lookup(pattern)
            if not found["matches"]:
                # No line holds even the words searched, so none holds them all.
                lines.append(f"No line of {_quoted(item['name'])}{'' if document is None else f' in {document}'} holds "
                             f"{_quoted(query)}.")
                # Where an id among the words stands alone: searching an employee's id with "name employee
                # directory", an agent found nothing, never searched the id alone, and said the collection had no
                # names.
                if len(words) > 1 and not named:
                    for word in [word for word in words if re.search(r"[0-9]", word)][:2]:
                        holders = holders_of(model, item, f"(?i){_literal(word)}")
                        if holders is None or not holders["matches"]:
                            continue
                        hinted = True
                        many = ("one document; the first line with it:" if len(holders["matches"]) == 1
                                else f"{len(holders['matches'])} documents; the first line with it in each:")
                        lines += [f"{_quoted(word)} alone is in {many}", *_holder_lines(holders["matches"]),
                                  "Find it alone, or name one of them as document."]
                continue
            found_any = True
            lines.append(f"{_quoted(item['name'])}: {'the first ' if found['truncated'] else ''}"
                         f"{len(found['matches'])} lines with "
                         f"{_quoted(' '.join(searched) if left else query)}:")
            if left:
                lines.append(f"AdaptiqAI's search takes a pattern of {PATTERN_CHARS} characters, which holds these "
                             f"words of {_quoted(query)} and not {_quoted(' '.join(left))}: a line here may lack "
                             "those.")
            tables = tables_of(model, item, found["matches"])
            matches = [_as_written(match, tables.get(match["path"])) for match in found["matches"]]
            records = knowledge.records(model, item["memory_id"], matches)
            shown: list[tuple[str, int, int]] = []
            columns_shown: set[str] = set()
            # A year of monthly exports has one header: said for the first document, it holds for the next ones too.
            columns_above: str | None = None
            columns_reused = False
            for match in matches:
                table = (tables.get(match["path"]) or {}).get("table")
                if table is not None and match["path"] not in columns_shown:
                    columns_shown.add(match["path"])
                    columns = columns_line(table)
                    if columns == columns_above:
                        columns_reused = True
                    else:
                        columns_above = columns
                        lines.append(f"- {match['path']}: {columns}")
                if any(at == match["path"] and start <= match["line"] <= stop for at, start, stop in shown):
                    lines.append(f"- {match['path']}:{match['line']}: {match['text']} (in the record above)")
                    continue
                lines.append(f"- {match['path']}:{match['line']}: {match['text']}")
                whole = records.get(id(match))
                size = sum(len(text) for text in whole["lines"].values()) if whole is not None else 0
                if whole is not None and size <= room:
                    room -= size
                    shown.append((match["path"], whole["start"], whole["end"]))
                    lines.append(f"  The whole record, lines {whole['start']}-{whole['end']}:" if whole["whole"]
                                 else f"  Lines {whole['start']}-{whole['end']} of its record, not all of it:")
                    lines += [f"  {at}: {whole['lines'][at]}" for at in range(whole["start"], whole["end"] + 1)
                              if whole["lines"].get(at)]
                    continue
                # A record on one line has other records beside it: asked for one machine's service cases, an agent
                # got 8 of them and 16 of other machines. A line beside it that is no record of its own (the object or
                # the list it is in) is still shown.
                one_line = json_records.is_structured_json(match["path"]) and record.whole_object(match["text"])

                def beside(text: str) -> bool:
                    return text != "" and not (one_line and record.whole_object(text))

                before = match["before"]
                lines += [f"  {match['line'] - len(before) + index}: {text}" for index, text in enumerate(before)
                          if beside(text)]
                lines += [f"  {match['line'] + 1 + index}: {text}" for index, text in enumerate(match["after"])
                          if beside(text)]
            if columns_reused:
                lines.append(COLUMNS_ABOVE)
            if found["truncated"]:
                shown_in = list(dict.fromkeys(match["path"] for match in found["matches"]))
                holders = None if named else holders_of(model, item, pattern)
                others = ([] if holders is None else
                          [match for match in holders["matches"] if match["path"] not in shown_in])
                # The documents not seen yet come first: named fifth, a directory with an employee's name was never
                # looked in.
                some = ", ".join([*(match["path"] for match in others), *shown_in][:4])
                if others:
                    count = ("one other document; the first line with it there:" if len(others) == 1
                             else f"{len(others)} other documents; the first line with it in each:")
                    lines += [f"More lines hold it, here and in {count}", *_holder_lines(others)]
                where = ("here and perhaps in other documents" if holders is None or not holders["complete"]
                         else "in these documents only")
                lead = "To see every line" if others else f"More lines hold it, {where}. To see every one"
                lines.append(f"{lead}, find again "
                             f"with document naming the documents ({some}; several with commas between, or a part of "
                             "their names): that lists them all, page by page, and words that stand on different lines "
                             "of one record (an id, a type, a date) find the records that hold them all. Or add words.")
        # A miss must never read as a result: say plainly that nothing was found.
        if found_any:
            lines.append(FOUND)
        elif not hinted:
            lines.append(NOTHING_FOUND)
        if not found_any:
            # What find did not search: the conversations added here.
            try:
                here = conversation_of(session)
            except knowledge.KnowledgeError:
                here = None
            if here is not None:
                lines += not_searched_by_find([item["name"] for item in view.get("conversations") or []
                                               if item["conversation_id"] != here])
        return "\n".join(lines)

    def tables_of(model: str, item: dict, matches: list[dict]) -> dict[str, dict]:
        """The lines of the tables (CSV, TSV) that found lines are in, when their originals' rows are known (core's
        tablesOf): the engine's text cuts a semicolon row at its decimal commas, so a found row is shown as the
        original has it."""
        paths = [path for path in dict.fromkeys(match["path"] for match in matches)
                 if re.search(r"\.(?:csv|tsv)$", path, re.I)]
        if not paths:
            return {}
        try:
            documents = documents_of(model, item)
        except knowledge.KnowledgeError:
            documents = []
        found = {}
        for path in paths:
            document = next((entry for entry in documents if entry["path"] == path), None)
            try:
                read = None if document is None else lines_of(model, item, document)
            except knowledge.KnowledgeError:
                read = None
            if read is not None and read.get("table") is not None:
                found[path] = read
        return found

    def keep_many(key: str, read: dict, chars: int) -> None:
        """Keep a document of a find over many as the most recently used, leaving out the least recently used beyond
        a find's worth of characters (core's keepMany). Called holding the lock."""
        nonlocal many_chars
        held = known_many.pop(key, None)
        if held is not None:
            many_chars -= held[1]
        known_many[key] = (read, chars)
        many_chars += chars
        while many_chars > MANY_KEPT_CHARS and next(iter(known_many)) != key:
            _, (_, old) = known_many.popitem(last=False)
            many_chars -= old

    def key_of(item: dict, document: dict) -> str:
        """Where a document's lines are kept: its collection, the collection's version and the document."""
        return f"{item['memory_id']}\n{document.get('version') or item.get('changed', 0)}\n{document['path']}"

    def touch(item: dict, documents: list[dict], many: bool) -> None:
        """Move a find's kept documents to the back of their shelves before it reads the rest, so its new reads leave
        out others, never its own (core's touch)."""
        with lock:
            for document in documents:
                key = key_of(item, document)
                held = known_many.get(key)
                if held is not None:
                    if many:
                        keep_many(key, *held)
                    continue
                kept = known.get(key)
                if kept is not None:
                    _keep(known, key, kept, DOCUMENTS_KEPT)

    def shelve(key: str, document: dict, read: dict, many: bool) -> None:
        """Keep a document read whole: with a find's over many, by its characters, else among the sixteen kept by
        count. Called holding the lock."""
        if many:
            keep_many(key, read, document.get("chars") or 0)
            return
        _keep(known, key, read, DOCUMENTS_KEPT)
        known_chars[key] = sum(map(len, read["lines"]))
        # What open keeps is held to KNOWN_CHARS: the oldest go first, never the one just read.
        total = 0
        for held in list(known_chars):
            if held in known:
                total += known_chars[held]
            else:
                del known_chars[held]
        for held in list(known):
            if total <= KNOWN_CHARS:
                break
            if held != key:
                total -= known_chars.pop(held, 0)
                del known[held]

    def lines_of(model: str, item: dict, document: dict, sources=None, many: bool = False,
                 keep: bool = True) -> dict | None:
        """A document's lines, or None when it is part of a larger source (a repository) that open can't show
        apart. Only all of a document's lines are kept: a start may be all there is (a PDF) or a passing failure,
        so the next open asks again. `sources()` lists the collection's sources when one list serves several
        documents; `many` keeps a find's documents by their characters when it lists more than twelve; `keep`
        false leaves keeping a new read to the caller."""
        key = key_of(item, document)
        with lock:
            held = known_many.get(key)
            kept = known.get(key) or (held[0] if held is not None else None)
        if kept is not None:
            return kept
        try:
            listed = sources() if sources is not None else knowledge.source_list(model, item["memory_id"])
            source = next((entry for entry in listed if entry["name"] == document["path"]), None)
            if source is None:
                return None
            start = knowledge.source_text(model, item["memory_id"], source["id"])
        except knowledge.KnowledgeError as error:
            raise _unreadable(item, "to read the documents of", error,
                              ": find still searches all of its text") from error
        table = source["type"] in TABLE_TYPES
        original = None
        if ((table or (not start["whole"] and source["type"] in VERBATIM_TYPES)) and source["bytes"] is not None
                and source["bytes"] <= ORIGINAL_BYTES):
            try:
                # As the core's decoder reads it: a byte order mark first is no character of the text.
                original = knowledge.original(model, item["memory_id"], source["id"]).decode("utf-8-sig")
            except (knowledge.KnowledgeError, ValueError):
                original = None
        if table:
            read = table_lines(start, original, document.get("lines"))
        else:
            read = document_lines(start, source, original)
            read = {**read, "headings": headings_in(document["path"], source["type"], read["lines"])}
        if read["whole"] and keep:
            with lock:
                shelve(key, document, read, many)
        return read

    def show(session: str, model: str, item: dict, document: dict, line: int | None, more: bool) -> str:
        """Lines of a document, from a line on, as open shows them."""
        path = document["path"]
        named = f"{path} ({document['lines']} lines, in {_quoted(item['name'])})"
        find_it = f"find gives its exact lines: find words or ids it holds, with document: {path}"
        read = lines_of(model, item, document)
        if read is None:
            return f"{named} is part of a larger source, so open can't show its lines; {find_it}."
        lines = read["lines"]
        # Kept by the document's version: a changed document is read again from its first line.
        key = f"{session}\nopen\n{key_of(item, document)}"
        # A line is read as the section it stands in, never past it: ten lines above a clause are the end of
        # another agreement, and the hundred after it are other clauses.
        # Reading on, an agent often names the line again (core's show): the line a page was asked at, or the line it
        # stopped inside, with more: true, is more of it, not the page again.
        asked = line
        held = page_at(key, more) if line is not None else None
        if held is not None and (held.get("asked") == line or (held["at"] > 0 and held["line"] == line)):
            asked, line = held.get("asked"), None
        section = section_at(read, line) if line is not None and line <= len(lines) else None
        # Where more reads on: a line, and a character of it.
        start = (page_at(key, more) if line is None else None) or {"line": 1, "at": 0}
        first = max(section["first"] if section else 1, line - OPEN_BEFORE) if line is not None else start["line"]
        # A line past what is known is never answered with a page without it (core's show).
        if first > len(lines) or (line is not None and line > len(lines)):
            return (f"{named} ends at line {len(lines)}." if read["whole"]
                    else f"open shows lines 1-{len(lines)} of {named}, not line {line or first}; {find_it}.")

        def text_of(number: int) -> str:
            row = lines[number - 1]
            return row[:-1] if row.endswith("\r") else row

        def cost(number: int, most: int, at: int = 0) -> int:
            """What a line takes of a page, from a character on: all of it up to `most`."""
            return min(_units(text_of(number)) - at, most)

        def page(begin_at: int, to: int, asked: int | None, at: int, slide: bool) -> dict:
            """A page, {first, shown, last, stop, parts}: lines of up to 2,000 characters whole, as many as 16,000
            characters and 120 lines hold, to `to` at most. The line asked for, or the first when none is, is shown
            whole up to a page, and past that the page ends inside it (`stop`), so more reads on in it. Another line
            longer than 2,000 characters is shown in part, and says so (`parts`). With `slide`, the page starts later
            where the lines above the one asked for would leave it no room (core's page in show)."""
            own = asked if asked is not None and begin_at <= asked <= to else None
            head = begin_at
            if slide and own is not None:
                head = max(head, own - OPEN_LINES + 1)
                needed = cost(own, OPEN_CHARS, at if own == head else 0)
                for number in range(head, own):
                    needed += cost(number, LINE_CHARS, at if number == begin_at else 0)
                while head < own and needed > OPEN_CHARS:
                    needed -= cost(head, LINE_CHARS, at if head == begin_at else 0)
                    head += 1
            shown: list[str] = []
            size = 0
            last = head - 1
            stop = None
            parts = False
            while last + 1 <= to and last + 1 < head + OPEN_LINES:
                number = last + 1
                text = text_of(number)
                wide = _wide(text)
                length = len(text) if wide is None else len(wide) // 2
                begin = at if number == begin_at else 0
                whole = number == own or (own is None and number == head)
                end = begin + min(length - begin, max(1, OPEN_CHARS - size) if whole else LINE_CHARS)
                if shown and size + end - begin > OPEN_CHARS:
                    break
                # Never between the two halves of a character outside the basic plane.
                if end < length and end > begin + 1 and _high(text, wide, end - 1):
                    end -= 1
                last = number
                piece = text[begin:end] if wide is None else wide[2 * begin:2 * end].decode("utf-16-le", "surrogatepass")
                shown.append((f"{number}: {piece}" if piece else f"{number}:") if begin == 0 and end == length
                             else f"{number} (characters {begin + 1:,}-{end:,} of {length:,}): {piece}")
                size += end - begin
                if end < length:
                    if whole:
                        stop = end
                        break
                    parts = True
            return {"first": head, "shown": shown, "last": last, "stop": stop, "parts": parts}

        # The section whole when a page holds it. One too long for a page is read around the line, and never past
        # its end. Where the line stands above the section's own sections, its own lines and their names instead:
        # the first 120 lines of a chapter stop inside one of its clauses and name none of the rest.
        kind = "page"
        shown_page = page(first, len(lines), line, start["at"] if line is None else 0, True)
        if section is not None:
            whole_section = page(section["first"], section["last"], line, 0, False)
            own = None if section["leaf"] else page(section["first"], section["own"], line, 0, False)
            if whole_section["last"] == section["last"] and whole_section["stop"] is None:
                shown_page = whole_section
                kind = "section"
            elif own is not None and own["last"] == section["own"] and own["stop"] is None:
                shown_page = own
                kind = "own"
            else:
                shown_page = page(first, section["last"], line, 0, True)
        # Asked for a line of JSON or of a table, the record or the row it stands in: the lines around it are other
        # records. A record too long for a page is read around the line, never past its end. Where it stands, and
        # its place among the others, are said.
        place: list[str] = []
        if line is not None and kind == "page" and line <= len(lines):
            in_json = json_records.is_structured_json(path) and read.get("table") is None
            if in_json and "units" not in read:
                read["units"] = json_units(lines, path)
            # Named apart from the record module this file reads.
            found_record = record_at(read["units"] or [], line, lines) if in_json else None
            rows = None if in_json or read.get("table") is not None else table_rows_of(read)
            header = None if in_json else read.get("table") or rows.get(line)
            if found_record is not None:
                unit = page(found_record["first"], found_record["last"], line, 0, False)
                shown_page = (unit if unit["last"] == found_record["last"] and unit["stop"] is None
                              and unit["first"] == found_record["first"]
                              else page(max(found_record["first"], line - OPEN_BEFORE), found_record["last"], line, 0,
                                        True))
                if shown_page is unit:
                    kind = "object" if found_record["list"] is None else "record"

                def labelled(held: dict) -> int:
                    # A record's "{" alone on its line says nothing: its first field does.
                    return (held["line"] + 1 if held["key"] == "" and lines[held["line"] - 1].strip(_JS_TRIMMED) == "{"
                            else held["line"])

                # A line on the page is not said again.
                place = [f"Under (line {labelled(held)}): "
                         f"{cut_text(lines[labelled(held) - 1].strip(_JS_TRIMMED), HEADING_CHARS)}"
                         for held in found_record["above"] if labelled(held) < shown_page["first"]]
                listed = found_record["list"]
                if listed is not None:
                    # How many there are only where all of the document is known: of its start, a count would be
                    # wrong.
                    where = "in the list" if listed["key"] == "" else f'in "{listed["key"]}"'
                    count = f" of {listed['count']:,}" if read["whole"] else ""
                    place.append(f"Record {listed['index']:,}{count} {where} at line {listed['line']}.")
            elif header is not None and line > header["line"]:
                top = bottom = line
                if rows is not None:
                    while rows.get(top - 1) is header:
                        top -= 1
                    while rows.get(bottom + 1) is header:
                        bottom += 1
                else:
                    top, bottom = header["line"] + 1, len(lines)
                shown_page = page(line, line, line, 0, True)
                kind = "row"
                count = f" of {bottom - top + 1:,}" if read["whole"] else ""
                place = [f"Row {line - top + 1:,}{count} of the table at line {header['line']}."]
        first, last, stop = shown_page["first"], shown_page["last"], shown_page["stop"]
        ends = last == len(lines) and stop is None
        # The end is kept too: another "more" says the document ends there, never shows its first page again.
        page_next(key, {"line": last + 1, "at": 0, "asked": asked} if stop is None else {"line": last, "at": stop, "asked": asked})
        said = {"page": "", "section": "That is all of that section. ",
                "own": "Those are its own lines, above its sections. ", "record": "That is the whole record. ",
                "object": "That is the whole object. ", "row": "That is the whole row. "}[kind]
        after = (f"Line {last} goes on past character {stop:,}: open it again with more: true for the rest of it."
                 if stop is not None else f"{said}More: open it again with more: true." if not ends
                 else "That is the end of it." if read["whole"] else f"open shows no line past {last}; {find_it}.")
        # A page that starts inside a table (a CSV's rows, a workbook's sheet) shows its columns, as its header row
        # is above it.
        header = read.get("table") or (None if record.is_json_document(path) else table_rows_of(read).get(first))
        columns = ([columns_line(header)] if header is not None and (first > header["line"] or last < header["line"])
                   else [])
        # The headings the page stands under, as their lines are above it: whose clause, which chapter's part.
        headings = read.get("headings") or []
        under = [*(f"Under (line {heading['line']}): {title_of(heading)}" for heading in headings_above(headings, first)),
                 *(row for row in place if row.startswith("Under "))]
        beside = [row for row in place if not row.startswith("Under ")]
        if section is not None and kind in ("section", "own"):
            beside += (sections_line(headings, section["at"]) if kind == "own"
                       else beside_line(headings, section["at"]) if section["leaf"] else [])
        whole = ["A line shown in part is shown whole by open at its line."] if shown_page["parts"] else []
        return "\n".join([f"{named}, lines {first}-{last}:", *under, *columns, *shown_page["shown"], *beside, *whole,
                          f"These are its exact lines: answer from them and cite {path}:line. {after}"])

    def open_document(session: str, model: str, view: dict, args: dict) -> str:
        wanted = str(args.get("document") or "").strip()
        if not wanted:
            _fail("Say which document to open, as contents or find shows it.")
        line = args.get("line")
        line = (int(line) if isinstance(line, (int, float)) and not isinstance(line, bool) and math.isfinite(line)
                and line > 0 else None)
        # A document by its whole path, else the one path of all those used here that ends with the name, as core's
        # open: never a shorter path the name ends with (a Rust book listing it lacks opened its top-level
        # Cargo.toml), nor the first of many (src/main.rs ends hundreds of its paths).
        ending: list[tuple[dict, dict]] = []
        close: list[str] = []
        for item in used_here(view, str(args.get("name") or "")):
            documents = documents_of(model, item)
            match = next((entry for entry in documents if entry["path"] == wanted), None)
            if match is not None:
                return show(session, model, item, match, line, args.get("more") is True)
            ending += [(item, entry) for entry in documents if entry["path"].endswith("/" + wanted)]
            close += [entry["path"] for entry in documents if wanted.lower() in entry["path"].lower()]
        if len(ending) == 1:
            return show(session, model, ending[0][0], ending[0][1], line, args.get("more") is True)
        # Those that end with the name come first, to pick one from.
        named = list(dict.fromkeys([entry["path"] for _, entry in ending] + close))
        rest = f", and {len(named) - 8:,} more" if len(named) > 8 else ""
        return _fail(f"No document is called {_quoted(wanted)}. Close: {', '.join(named[:8])}{rest}." if named
                     else f"No document is called {_quoted(wanted)}. Take contents, or find words it holds.")

    @_guarded("adaptiq_knowledge")
    def knowledge_tool(args: dict, session_id: str = "", **_) -> str:
        def work() -> str:
            _connected()
            session = _session(session_id)
            if elsewhere(session):
                _fail(ELSEWHERE)
            action = args.get("action")
            if action not in ("status", "search", "use", "stop", "contents", "find", "open", "find_in_conversation"):
                _fail("Say what to do: status, search, use, stop, contents, find, open or find_in_conversation.")
            if action == "find_in_conversation":
                which = str(args.get("conversation") or "").strip()
                if which:
                    # One added here is read from AdaptiqAI's timeline of it, beside Hermes' own store for this one.
                    return told(session, action, args, find_in_linked(session, which, args), which)
                # What the person wrote here, from Hermes' own store, with the ones added here it did not search.
                return told(session, action, args, with_unsearched(session, find_in_conversation(session, args)),
                            THIS_CONVERSATION)
            conversation = conversation_of(session)
            view = knowledge.conversation_view(conversation, model_of(session))
            if action == "status":
                return status(session, view)
            model = model_here(session, view)
            selected = frozenset(item["memory_id"] for item in view["knowledge"])
            if action == "search":
                return search(session, conversation, model, view, args, selected)
            # Each read says when it was asked again and nothing is new.
            if action in ("contents", "find", "open"):
                if action != "find" or not str(args.get("conversation") or "").strip():
                    return told(session, action, args, read(session, conversation, model, view, selected, args))
                # What conversation is for comes first, whatever find answers (core's tools.ts).
                try:
                    page = read(session, conversation, model, view, selected, args)
                except knowledge.KnowledgeError as error:
                    error.args = (f"{CONVERSATION_NOT_FOR_FIND}\n{error}", *error.args[1:])
                    raise
                return told(session, action, args, f"{CONVERSATION_NOT_FOR_FIND}\n{page}")
            use = action == "use"
            item = named(model, view, str(args.get("name") or ""), not use, selected)
            if (item["memory_id"] in selected) == use:
                return (f"{_quoted(item['name'])} is already used in this conversation." if use
                        else f"{_quoted(item['name'])} is not used in this conversation.")
            if use:
                return start(conversation, model, item)
            knowledge.use(conversation, model, item["memory_id"], False)
            return f"{_quoted(item['name'])} is no longer used in this conversation. Nothing was deleted."

        return _run(work)

    def read(session: str, conversation: str, model: str, view: dict, selected: frozenset, args: dict) -> str:
        """A read (contents, find, open) of a collection the conversation does not use yet uses it first, and a read
        with no collection to read names the ones there are (core's read). Told to search and use one first, agents
        spent two to four steps on it, in 173 of 325 captured conversations (MiMo eval, 2026-11-14)."""
        reader = {"contents": contents, "find": find, "open": open_document}[args.get("action")]
        try:
            return reader(session, model, view, args)
        except _NotUsedHere as error:
            def refused(said: str):
                there = usable_here(model, selected) if error.none else []
                if not there:
                    raise error
                _fail("\n".join([said, "The collections there are, newest first:", *there]))

            if error.asked is None:
                refused("This conversation uses no ready collection. Name the one to read: contents, find and open "
                        "use it here.")
            try:
                item = named(model, view, error.asked, False, selected)
            except knowledge.KnowledgeError as missing:
                if not str(missing).startswith("No saved knowledge is called"):
                    raise
                if not error.none:
                    raise error from None
                refused(f"No saved collection is called {_quoted(error.asked)}, and this conversation uses none.")
            if item["memory_id"] in selected:
                if item["readiness"] != "ready":
                    _fail(f"{_quoted(item['name'])} is {_state(item)}; it can be read once it is ready.")
                raise error
            started = start(conversation, model, item)
            view = knowledge.conversation_view(conversation, model_of(session))
            return f"{started}\n{reader(session, model, view, args)}"

    @_guarded("adaptiq_memory")
    def memory_tool(args: dict, session_id: str = "", **_) -> str:
        def work() -> str:
            _connected()
            session = _session(session_id)
            if elsewhere(session):
                _fail(ELSEWHERE)
            action = args.get("action")
            if action == "continue_without_saving":
                set_remember(session, False)
                return "From now on this conversation is not saved. Everything saved before stays."
            conversation = conversation_of(session)
            if action == "review":
                url = knowledge.console_link(conversation)
                try:
                    opened = bool(open_url(url))
                except Exception:
                    opened = False
                return ("Knowledge is open in the browser, so the person can tidy up there. Nothing was deleted."
                        if opened else f"The browser did not open. The person can tidy up in Knowledge: {url}. Nothing "
                                       "was deleted.")
            if action not in ("status", "inventory"):
                _fail("Say what to do: status, inventory, review or continue_without_saving.")
            view = knowledge.conversation_view(conversation, model_of(session))
            if action == "status":
                held = _memory_view((core.account_summary() or {}).get("memory"))
                lines = []
                if held is None or not held["available"]:
                    lines.append("The workspace memory size is not known right now.")
                elif held["history"] is not None:
                    lines.append(f"Knowledge library: {format_tokens(held['used'])} of {format_tokens(held['limit'])} "
                                 f"tokens{' — full: new knowledge needs room first' if held['full'] else ''}.")
                    lines.append(f"Conversation history: {format_tokens(held['history']['used'])} of "
                                 f"{format_tokens(held['history']['limit'])} tokens. Idle conversations rest by "
                                 "themselves to make room; nothing is deleted and they wake when used.")
                else:
                    lines.append(f"Memory for all conversations and knowledge together: {format_tokens(held['used'])} "
                                 f"of {format_tokens(held['limit'])} tokens{' — full' if held['full'] else ''}.")
                if view["memory_tokens"] is not None:
                    lines.append(f"This conversation holds {format_tokens(view['memory_tokens'])} tokens"
                                 f"{'' if remember_of(session) else ' and is private: new messages are not saved'}.")
                lines.append("A new conversation, compacting or removing knowledge from a conversation does not free "
                             "memory.")
                return "\n".join(lines)
            model = model_here(session, view)
            key = f"{session}\ninventory"
            page = library(model, None, page_at(key, args.get("more")),
                           frozenset(item["memory_id"] for item in view["knowledge"]))
            page_next(key, page["next_cursor"] if page["next_cursor"] and page["has_more"] else None)
            items = [item for item in page["items"] if item["kind"] == "collection"]
            if not items:
                return "No knowledge is saved yet."

            def line(item: dict) -> str:
                tokens = _count(item.get("logical_tokens"))
                return ", ".join(part for part in (
                    f"- {_quoted(item['name'])}",
                    None if tokens is None else f"{format_tokens(tokens)} tokens",
                    f"changed {item['updated']}" if item.get("updated") else None,
                    "used in this conversation" if item["selected"] else None,
                ) if part)

            return "\n".join([("Saved knowledge (sizes are not guaranteed savings; tidy up in Knowledge before "
                               "deleting anything):"), *map(line, items),
                              *(["More: inventory again with more: true."] if page["has_more"] else [])])

        return _run(work)

    return knowledge_tool, memory_tool


class _ChangeRefused(knowledge.KnowledgeError):
    """Correcting or taking back a collection is not possible here: `code` says why (`connection` or `late`)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _now() -> float:
    return time.time()


# Where the plugin state keeps the collections the agent saved lately, so a correction after a restart of Hermes'
# gateway corrects the same collection instead of making a second one.
MADE_STATE = "made_knowledge"


def create_handler(written: Written, model_of, follow, unfollow=lambda memory_id: None,
                   conversation_of=lambda session: session, state=None):
    """The create tool. `follow(session, memory_id, model, name)` uses a collection in its conversation once it is
    ready, and `unfollow(memory_id)` stops that; `conversation_of(session)` is the AdaptiqAI conversation a Hermes
    session belongs to; `state` (Hermes' plugin state) keeps what the agent saved in the last 15 minutes across
    restarts."""
    # Collections created here, by conversation and operation id.
    operations: dict[str, dict] = {}
    # Collections the agent saved, by id: its conversation, name, store, when, and the sources it put in.
    made: OrderedDict[str, dict] = OrderedDict()
    lock = threading.Lock()

    def recent(entry) -> bool:
        return isinstance(entry, dict) and isinstance(entry.get("at"), (int, float)) and _now() - entry["at"] <= UNDO_WINDOW

    def load() -> None:
        """What the plugin state holds that this process does not: saved here before a restart, or by another
        Hermes process."""
        if state is None:
            return
        try:
            saved = state.get(MADE_STATE, {}) or {}
        except Exception:  # an unreadable state file: this process's own record still counts
            return
        with lock:
            for memory_id, entry in (saved.get("made") or {}).items():
                if memory_id not in made and recent(entry) and isinstance(entry.get("session"), str):
                    made[memory_id] = entry
            for key, entry in (saved.get("operations") or {}).items():
                if key not in operations and isinstance(entry, dict) and entry.get("id") in made:
                    operations[key] = entry

    def save() -> None:
        """Keep what was saved in the last 15 minutes; older entries can no longer be corrected or taken back."""
        if state is None:
            return
        with lock:
            kept = {memory_id: entry for memory_id, entry in made.items() if recent(entry)}
            ops = {key: entry for key, entry in operations.items() if entry.get("id") in kept}
        try:
            state.set(MADE_STATE, {"made": kept, "operations": ops})
        except Exception:  # the record stays in this process
            pass

    def recently_made(session: str, name: str) -> str | None:
        """The collection the agent saved in this conversation in the last 15 minutes under this name, newest."""
        wanted = same_name(name)
        found = [(memory_id, entry) for memory_id, entry in made.items()
                 if entry["session"] == session and _now() - entry["at"] <= UNDO_WINDOW
                 and same_name(entry["name"]) == wanted]
        return max(found, key=lambda pair: pair[1]["at"])[0] if found else None

    def made_within(session: str, memory_id: str) -> dict:
        entry = made.get(memory_id)
        if entry is None or entry["session"] != session:
            _fail("Only knowledge saved in this conversation can be changed or taken back here.")
        if _now() - entry["at"] > UNDO_WINDOW:
            raise _ChangeRefused("late", "It was saved more than 15 minutes ago, so it can no longer be changed or "
                                         "taken back from the conversation.")
        return entry

    def new(session: str, name: str, description: str, body: list[dict], key: str, use_here: bool) -> dict:
        created = knowledge.create(name, description, body, model=model_of(session), idempotency_key=key)
        _keep(made, created["memory_id"], {"session": session, "name": created["name"], "model": created["model"],
                                           "at": _now(), "sources": created["sources"], "attempt": None}, MADE_KEPT)
        if use_here:
            follow(session, created["memory_id"], created["model"], created["name"])
        return created

    def replace(session: str, memory_id: str, name: str, body: list[dict], key: str) -> str:
        """Replace the content of a collection the agent saved here a moment ago: the same collection gets a new
        version, no second one is made. Only the agent's own content goes out, after the new content is in: never a
        source anyone added since, nor what this correction (or its first try) put in."""
        entry = made_within(session, memory_id)
        model = entry["model"]

        def refused(error: knowledge.KnowledgeError):
            if error.status in (401, 403) or error.missing_route:
                raise _ChangeRefused("connection", "This connection can't change knowledge.") from error
            raise error

        def listed() -> list[str]:
            try:
                return knowledge.source_ids(memory_id, model)
            except knowledge.KnowledgeError as error:
                refused(error)

        attempt = entry.get("attempt")
        retry = attempt if attempt and attempt["key"] == key else None
        before = listed()
        old = retry["old"] if retry else entry["sources"]
        if old is None:
            _fail(f"{entry['name']} could not be told apart from what was added to it since, so it was not changed. "
                  "Take it back and save it again, or save the correction under another name.")
        try:
            knowledge.add_sources(memory_id, model, body, key)
        except knowledge.KnowledgeError as error:
            refused(error)
        after = listed()
        added = list(dict.fromkeys([*(retry["added"] if retry else []), *(source for source in after if source not in before)]))
        # A correction whose new content cannot be seen takes nothing out.
        if not added:
            _fail(f"AdaptiqAI did not confirm the corrected content of {entry['name']}, so nothing was taken out. "
                  "Check Knowledge before trying again.")
        entry["attempt"] = {"key": key, "added": added, "old": old}
        for source in old:
            if source in before and source not in added:
                knowledge.remove_source(memory_id, model, source)
        entry["sources"], entry["attempt"] = added, None
        if name[:160] != entry["name"]:
            try:
                knowledge.rename(memory_id, model, name)
            except knowledge.KnowledgeError:
                pass
            entry["name"] = name[:160]
        return entry["name"]

    def used_here(session: str, memory_id: str, model: str | None) -> bool:
        try:
            return any(item["memory_id"] == memory_id
                       for item in knowledge.conversation_view(conversation_of(session), model)["knowledge"])
        except knowledge.KnowledgeError:
            return False

    def undo(session: str, memory_id: str) -> tuple[str, str]:
        """Take back a collection the agent saved in this conversation in the last 15 minutes: it stops being used
        here and is removed (`removed`), or only stops being used where this connection may not remove it (`kept`)."""
        entry = made_within(session, memory_id)
        unfollow(memory_id)
        if used_here(session, memory_id, entry["model"]):
            try:
                knowledge.use(conversation_of(session), entry["model"], memory_id, False)
            except knowledge.KnowledgeError:
                pass
        if not knowledge.remove(memory_id, entry["model"]):
            return entry["name"], "kept"
        made.pop(memory_id, None)
        return entry["name"], "removed"

    @_guarded("adaptiq_create_knowledge")
    def create_tool(args: dict, session_id: str = "", task_id: str = "default", **_) -> str:
        def work() -> str:
            _connected()
            session = _session(session_id)
            load()
            operation = str(args.get("operation_id") or "")
            if not OPERATION_ID.fullmatch(operation):
                _fail("Use a short operation_id of letters, digits and . _ : -")
            opkey = f"{session}\n{operation}"
            known = operations.get(opkey)
            given = args.get("name") if isinstance(args.get("name"), str) else None
            if args.get("action") == "undo":
                memory_id = (known["id"] if known else None) or (recently_made(session, given) if given else None)
                if memory_id is None:
                    _fail("Only knowledge you saved in this conversation in the last 15 minutes can be taken back. "
                          "Nothing was removed.")
                name, outcome = undo(session, memory_id)
                operations.pop(opkey, None)
                return (f"Taken back: {_quoted(name)} is removed." if outcome == "removed"
                        else f"{_quoted(name)} is no longer used in this conversation, but this connection can't "
                             "remove it, so it stays in the Knowledge list.")
            if args.get("action") != "create":
                _fail("Say what to do: create or undo.")
            name = (given or "").strip()
            if not name:
                _fail("Give the collection a name.")
            correcting = known["id"] if known else recently_made(session, name)
            if correcting is None and not written.asked_for_knowledge(session, str(args.get("user_request") or "")):
                _fail(ASK_FIRST)
            items = args.get("sources") if isinstance(args.get("sources"), list) else []
            files = any(isinstance(item, dict) and item.get("type") == "file" for item in items)
            if files and _container(task_id):
                _fail("Files here are inside the terminal's container, not on this computer. Add their text, or a link.")
            body = knowledge.sources(items, _cwd(task_id), name,
                                     resolve=_resolver(task_id) if files else None, blocked=_blocked())
            description = str(args.get("description") or "")
            digest = hashlib.sha256(json.dumps([name, description, body], sort_keys=True).encode()).hexdigest()
            if known is not None and known["hash"] == digest:
                return f"{_quoted(known['name'])} is already saved as knowledge; nothing changed."
            use_here = args.get("use_here") is not False

            def key(kind: str) -> str:
                return (f"{KEY_PREFIX}-{kind}-"
                        + hashlib.sha256(f"{session}\n{operation}\n{digest}".encode()).hexdigest()[:48])

            if correcting is not None:
                try:
                    changed = replace(session, correcting, name, body, key("fix"))
                except _ChangeRefused as error:
                    if error.code == "late":
                        _fail(f"{error} Ask the person whether to save the new content as a separate collection with "
                              "another name.")
                    # This connection may not change knowledge: save the corrected one, then take the first back.
                    created = new(session, name, description, body, key("tool"), use_here)
                    try:
                        _, outcome = undo(session, correcting)
                    except knowledge.KnowledgeError:
                        outcome = "kept"
                    operations[opkey] = {"id": created["memory_id"], "name": created["name"], "hash": digest}
                    return (f"Corrected: {_quoted(created['name'])} is saved with the new content and the first "
                            "version is removed." if outcome == "removed"
                            else f"Saved the corrected version as {_quoted(created['name'])}. This connection can't "
                                 "remove the first version; it is no longer used in this conversation.")
                operations[opkey] = {"id": correcting, "name": changed, "hash": digest}
                model = made[correcting]["model"]
                if use_here and not used_here(session, correcting, model):  # used as soon as it is ready
                    follow(session, correcting, model, changed)
                return (f"Corrected: {_quoted(changed)} now holds the new content; no second collection was made."
                        + (" It is used in this conversation as soon as it is ready." if use_here else ""))
            created = new(session, name, description, body, key("tool"), use_here)
            operations[opkey] = {"id": created["memory_id"], "name": created["name"], "hash": digest}
            if not use_here:
                return f"Saved as knowledge: {_quoted(created['name'])}."
            if created["readiness"] == "ready":
                return f"Saved as knowledge: {_quoted(created['name'])} — ready and used in this conversation."
            return f"Saved as knowledge: {_quoted(created['name'])} — it is used in this conversation as soon as it is ready."

        try:
            return _run(work)
        finally:
            save()

    return create_tool
