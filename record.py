"""The JSON record a line find found is in (core's record.ts).

The engine gives a found line with at most five lines on each side and cannot read from a line number, so a
record of ten lines took the same id found again and again: on prod, five questions over JSON records took 27 to
66 lookups. The rest of a record is read by finding a line at the edge of what is known, which brings the five
lines beyond it, until its brackets close.
"""

from __future__ import annotations

import re
from typing import Callable

from . import json_records

# The most lines a record is shown with; a longer one is shown in part.
RECORD_LINES = 40
# The lines the engine gives on each side of a found line, at most.
RECORD_SIDE = 5
# The searches one record may take: a step often brings only three new lines, as the line at the very edge
# ("breve": -4) is in too many records to find.
READS = 12
# A line with this many digits and capitals is likely to be found once in its document.
DISTINCT = 4


def is_json_document(path: str) -> bool:
    """A JSON document's lines are shown with their record; a JSON Lines line is its own record already."""
    return path.lower().endswith(".json")


# What JavaScript's trim takes off a text's ends, as core trims a line: a byte order mark too, never a file separator
# as Python's strip does.
JS_TRIMMED = ("\t\n\v\f\r \u00a0\u1680" + "".join(map(chr, range(0x2000, 0x200B)))
              + "\u2028\u2029\u202f\u205f\u3000\ufeff")


def whole_object(text: str) -> bool:
    """Whether a line is a whole JSON object by itself, its closing comma aside (core's wholeObject): a register
    written one record a line (JSON Lines, or a list with an object a line)."""
    line = text.strip(JS_TRIMMED)
    line = line[:-1] if line.endswith(",") else line
    if not line.startswith("{") or not line.endswith("}"):
        return False
    try:
        json_records._parse(line)
    except (ValueError, RecursionError):
        return False
    return True


def literal(text: str) -> str:
    """A text as a regular expression that matches it literally."""
    return re.sub(r"[.*+?^${}()|\[\]\\]", r"\\\g<0>", text)


def _brackets(text: str) -> list[str]:
    """The brackets of a JSON line outside its strings, in order."""
    found, quoted, at = [], False, 0
    while at < len(text):
        char = text[at]
        if quoted:
            if char == "\\":
                at += 1
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{}[]":
            found.append(char)
        at += 1
    return found


def _settle(lines: dict[int, str], top: int, bottom: int, hit: int, first: bool, last: bool) -> dict:
    """The object a found line is in, when its lines are known, else which way to read on. A line that opens an
    object ("MED-73": {) is inside it; an array the line is in (a list of names) is passed on the way out."""
    own = _brackets(lines.get(hit, ""))
    inside = own.index("{") + 1 if "{" in own else 0
    openers: list[tuple[int, str]] = []
    closers: list[int] = []
    shut = 0
    for line, marks in [(hit, own[:inside])] + [(line, _brackets(lines.get(line, ""))) for line in range(hit - 1, top - 1, -1)]:
        for char in reversed(marks):
            if char in "}]":
                shut += 1
            elif shut:
                shut -= 1
            else:
                openers.append((line, char))
    depth = 0
    for line, marks in [(hit, own[inside:])] + [(line, _brackets(lines.get(line, ""))) for line in range(hit + 1, bottom + 1)]:
        for char in marks:
            if char in "{[":
                depth += 1
            elif depth:
                depth -= 1
            else:
                closers.append(line)
    innermost = next((line for line, char in openers if char == "{"), None)
    level = 0
    while True:
        opener = openers[level] if level < len(openers) else None
        closer = closers[level] if level < len(closers) else None
        if opener is None or closer is None:
            # Its ends are not read yet, and what is read of it is already too long.
            done = (opener is None and first) or (closer is None and last) or \
                (bottom if closer is None else closer) - (top if opener is None else opener[0]) + 1 >= RECORD_LINES
            return {"innermost": innermost, "up": not done and opener is None, "down": not done and closer is None}
        if closer - opener[0] + 1 > RECORD_LINES:
            return {"innermost": innermost, "up": False, "down": False}
        if opener[1] == "{":
            return {"innermost": innermost, "record": (opener[0], closer), "up": False, "down": False}
        level += 1


def record_in(lines: list[str], hit: int, json: bool) -> dict | None:
    """The record a found line is in, when all of its document's lines are known (core's recordIn): in a JSON
    document the object around it, in a text document the paragraph (the lines between two empty ones). A record
    longer than RECORD_LINES is shown in part: a JSON record from its start, a paragraph as the line and one line on
    each side. `lines` are the document's lines, the first one numbered 1. None for a JSON object on one line,
    which is the line itself."""
    top = max(1, hit - RECORD_LINES)
    bottom = min(len(lines), hit + RECORD_LINES)
    known = {line: lines[line - 1].strip() for line in range(top, bottom + 1)}
    if json:
        state = _settle(known, top, bottom, hit, top <= 1, bottom >= len(lines))
        if "record" in state:
            start, end = state["record"]
            return {"start": start, "end": end, "whole": True, "lines": known} if end > start else None
        start = max(top if state["innermost"] is None else state["innermost"], hit - RECORD_LINES + 1)
        return {"start": start, "end": min(bottom, start + RECORD_LINES - 1), "whole": False, "lines": known}
    start = end = hit
    while start > top and known.get(start - 1) != "":
        start -= 1
    while end < bottom and known.get(end + 1) != "":
        end += 1
    closed = (start == 1 or known.get(start - 1) == "") and (end == len(lines) or known.get(end + 1) == "")
    if closed and end - start + 1 <= RECORD_LINES:
        return {"start": start, "end": end, "whole": True, "lines": known}
    return {"start": max(1, hit - 1), "end": min(len(lines), hit + 1), "whole": False, "lines": known}


def json_record(hit: dict, side: int, read: Callable[[str], list[dict]]) -> dict | None:
    """The record a line of a JSON document is in: the object around it, or its first RECORD_LINES lines when it
    is longer. `hit` is the found line as find gave it (line, text, before, after), with `side` lines on each side;
    `read` searches its document for a pattern, RECORD_SIDE lines around each match. None when no more of the
    record than the hit could be read."""
    lines: dict[int, str] = {}
    end = float("inf")

    def take(match: dict, asked: int) -> None:
        nonlocal end
        before, after = match.get("before") or [], match.get("after") or []
        for index, text in enumerate(before):
            lines[match["line"] - len(before) + index] = text
        lines[match["line"]] = match["text"]
        for index, text in enumerate(after):
            lines[match["line"] + 1 + index] = text
        if len(after) < asked:
            end = min(end, match["line"] + len(after))

    def span() -> tuple[int, int]:
        top = bottom = hit["line"]
        while top - 1 in lines:
            top -= 1
        while bottom + 1 in lines:
            bottom += 1
        return top, bottom

    def check() -> dict:
        top, bottom = span()
        return _settle(lines, top, bottom, hit["line"], top <= 1, bottom >= end)

    tried: set[int] = set()

    def score(text: str) -> int:
        return len(re.findall(r"[0-9A-Z]", text))

    def anchor(start: int, stop: int, edge: int) -> int | None:
        """The line to find next near an edge: a distinctive one nearest the edge, else the most distinctive."""
        candidates = sorted((line for line in range(start, stop + 1)
                             if line in lines and line not in tried and len(lines[line]) <= 200 and score(lines[line]) > 0),
                            key=lambda line: abs(line - edge))
        return next((line for line in candidates if score(lines[line]) >= DISTINCT), None) or \
            next(iter(sorted(candidates, key=lambda line: -score(lines[line]))), None)

    take(hit, side)
    state, reads = check(), 0
    while reads < READS and (state["up"] or state["down"]):
        top, bottom = span()
        asks = []
        for wanted, start, stop, edge in ((state["up"], top, top + RECORD_SIDE - 1, top),
                                          (state["down"], bottom - RECORD_SIDE + 1, bottom, bottom)):
            line = anchor(start, stop, edge) if wanted else None
            if line is not None and line not in asks:
                asks.append(line)
        if not asks:
            break
        reads += len(asks)
        tried.update(asks)
        for line in asks:
            match = next((entry for entry in read(literal(lines[line])) if entry["line"] == line), None)
            if match is not None:
                take(match, RECORD_SIDE)
        state = check()

    # An object on one line is the line find gave.
    if "record" in state:
        start, stop = state["record"]
        return {"start": start, "end": stop, "whole": True, "lines": lines} if stop > start else None
    top, bottom = span()
    start = max(top if state["innermost"] is None else state["innermost"], hit["line"] - RECORD_LINES + 1)
    stop = min(bottom, start + RECORD_LINES - 1)
    return {"start": start, "end": stop, "whole": False, "lines": lines} if stop - start + 1 > 3 else None
