"""A JSON document's objects as find's records (core's json-records.ts): the whole document is checked as JSON first,
then each object is a record with its lines and its values as JSON types them, however its lines are laid out.

Written as core's JavaScript reads them: JSON.parse's checks, an object's keys in JavaScript's order (whole-number
keys first, smallest first) and JSON.stringify's text, so a record holds the same words and values in Hermes."""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal

_STRUCTURED = re.compile(r"\.(?:json|jsonl|ndjson)\Z", re.I)
_LINES = re.compile(r"\.(?:jsonl|ndjson)\Z", re.I)
# A string, a number, a mark, a word (true, false, null) or a line break of a JSON text: what core's scan walks over a
# character at a time, a token at a time. A valid JSON string has no line break in it.
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|[{}\[\],:\n]|[a-z]+', re.S)
_INDEX = re.compile(r"0|[1-9][0-9]*")
_SURROGATE = re.compile("[\ud800-\udfff]")
SAFE = 2 ** 53 - 1
DEEPEST = 100


class JsonRecordError(ValueError):
    """A JSON document find cannot list: its sentence is said as find's refusal."""


def is_structured_json(path: str) -> bool:
    """Whether find lists a document's JSON objects as its records: JSON, JSON Lines and NDJSON."""
    return _STRUCTURED.search(path) is not None


def number_text(value: float) -> str:
    """A number as JavaScript writes it (String(number)): its shortest digits, as a point number from 10^-6 to 10^21
    ("1152921504606847000" for 2^60), else with an exponent ("1e-7", "1e+21")."""
    if not math.isfinite(value):
        return "NaN" if value != value else "Infinity" if value > 0 else "-Infinity"
    if value == int(value) and abs(value) <= SAFE:
        return str(int(value))
    text = repr(float(value))
    if 1e-6 <= abs(value) < 1e21:
        fixed = format(Decimal(text), "f")
        return fixed[:-2] if fixed.endswith(".0") else fixed
    return re.sub(r"e([+-])0*(?=[0-9])", r"e\1", text)


def _decimal_key(token: str) -> str:
    """A decimal number's digits and exponent, without rounding (core's decimalKey): "1.50" and "15e-1" are the same."""
    mantissa, _, exponent = token.lower().partition("e")
    whole, _, fraction = mantissa.removeprefix("-").partition(".")
    digits = f"{whole}{fraction}".lstrip("0")
    if not digits:
        return "0"
    significant = digits.rstrip("0")
    sign = "-" if mantissa.startswith("-") else ""
    return f"{sign}{significant}e{int(exponent or '0') - len(fraction) + len(digits) - len(significant)}"


def _unsafe(value: float) -> bool:
    return not math.isfinite(value) or (value == int(value) and abs(value) > SAFE)


def _refuse_constant(name: str):
    raise ValueError(f"JSON has no {name}")


def _parse(text: str):
    """JSON.parse: NaN and Infinity are no JSON, and every number is a double, however many digits it has."""
    return json.loads(text, parse_constant=_refuse_constant, parse_int=float)


def entries(value: dict) -> list[tuple[str, object]]:
    """An object's keys and values in JavaScript's order (Object.entries): keys that are array indexes (whole numbers
    from 0 to 2^32 - 2, as written by String) first, smallest first, then the others as they stand."""
    index = [key for key in value if _INDEX.fullmatch(key) and int(key) <= 2 ** 32 - 2]
    if not index:
        return list(value.items())
    taken = set(index)
    return [(key, value[key]) for key in sorted(index, key=int)] + [
        (key, item) for key, item in value.items() if key not in taken]


def _string(text: str) -> str:
    """A string as JSON.stringify writes it: a lone surrogate as an escape."""
    return _SURROGATE.sub(lambda found: f"\\u{ord(found.group()):04x}", json.dumps(text, ensure_ascii=False))


def stringify(value, indent: str | None = None, depth: int = 0) -> str:
    """JSON.stringify(value, null, indent) of a parsed JSON value: numbers as JavaScript writes them, keys in its
    order."""
    if value is None:
        return "null"
    if value is True or value is False:
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return number_text(value) if math.isfinite(value) else "null"
    if isinstance(value, str):
        return _string(value)
    inner = None if indent is None else "\n" + indent * (depth + 1)
    close = "" if indent is None else "\n" + indent * depth
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [stringify(item, indent, depth + 1) for item in value]
        return "[" + ",".join(items) + "]" if inner is None else "[" + ",".join(inner + item for item in items) + close + "]"
    pairs = entries(value)
    if not pairs:
        return "{}"
    if inner is None:
        return "{" + ",".join(f"{_string(key)}:{stringify(item)}" for key, item in pairs) + "}"
    return "{" + ",".join(f"{inner}{_string(key)}: {stringify(item, indent, depth + 1)}" for key, item in pairs) + close + "}"


def field(value) -> tuple[str, str]:
    """A JSON value as a field's text and kind (core's jsonField): a string as it is, a number as JavaScript writes
    it, anything else as JSON; kind is string, number, boolean, null or object (an array too)."""
    if isinstance(value, bool):
        return ("true" if value else "false"), "boolean"
    if isinstance(value, (int, float)):
        if _unsafe(value):
            raise JsonRecordError("A JSON number is outside the supported finite, safe numeric range; find cannot "
                                  "report it exactly. Use open to inspect the source.")
        return number_text(value), "number"
    if isinstance(value, str):
        return value, "string"
    if value is None:
        return "null", "null"
    return stringify(value), "object"


def _scalar(value) -> bool:
    return (value is None or not isinstance(value, (dict, list))
            or (isinstance(value, list) and all(item is None or not isinstance(item, (dict, list)) for item in value)))


def _validate(source: str, path: str) -> None:
    """JSON.parse over the whole document, or each line of JSON Lines (core's jsonRecords). A .json of several lines
    that each hold one object is JSON Lines too; a part of a document that is not JSON is never listed."""
    if _LINES.search(path):
        for line in source.split("\n"):
            if line.strip():
                _parse(line)
        return
    try:
        _parse(source)
    except ValueError:
        # Some exports name JSONL .json. Accept only complete object rows, never salvage a valid-looking part of a
        # malformed JSON document.
        rows = [line for line in source.split("\n") if line.strip()]
        if len(rows) < 2:
            raise
        for row in rows:
            if not isinstance(_parse(row), dict):
                raise ValueError("Invalid JSON Lines record") from None


def whole_records(records: list[dict]) -> list[int]:
    """The records a find with no words goes through, by their places (core's wholeRecords): the outermost ones.
    Where a document holds several objects at the top (JSON Lines, or a list), those; where one object stands around
    its lists, the items of the lists in it that stand in no other item. Never an object under a key of another (a
    log entry's details are part of the entry), nor an item of a list a record holds (a station's pumps are part of
    the station)."""
    tops = [index for index, entry in enumerate(records) if entry["parent"] is None]
    if len(tops) != 1:
        return tops
    by_offset = {entry["offset"]: entry for entry in records}

    def item(entry: dict) -> bool:
        # A list's item: it stands under no key, in another object.
        return entry["key"] == "" and entry["parent"] is not None

    def in_item(entry: dict) -> bool:
        at = by_offset.get(entry["parent"])
        while at is not None:
            if item(at):
                return True
            at = by_offset.get(at["parent"])
        return False

    items = [index for index, entry in enumerate(records) if item(entry) and not in_item(entry)]
    # One object with no list of objects in it is the record itself.
    return items or tops


def json_records(lines: list[str], path: str) -> list[dict]:
    """A JSON document's objects, each a record: its first and last line, where it starts, its value, and the text a
    find searches (core's jsonRecords): the key it stands under, its own fields that are no object (a parent never
    joins its children's words), and its JSON when all its fields are such. Records are in the order they start.
    Raises JsonRecordError with find's sentence for a document that is not valid JSON, a number find can't report
    exactly, or objects nested too deeply."""
    heading = 4 if (len(lines) >= 4 and lines[0].startswith("# ") and lines[1].strip() == ""
                    and lines[2].startswith("Source type:") and lines[3].strip() == "") else 0
    source = "\n".join(lines[heading:])
    source = source[1:] if source.startswith("\ufeff") else source
    deep = JsonRecordError(f"{path} has JSON nested too deeply to list safely. Use open to inspect the source.")
    try:
        _validate(source, path)
    except RecursionError:
        # Python's parser stops at about a thousand levels; core's scan refuses past a hundred objects.
        raise deep from None
    except ValueError:
        raise JsonRecordError(f"{path} is not valid JSON{' Lines' if _LINES.search(path) else ''}; find cannot safely "
                              "list, sort, sum or group its records. Use open to inspect the source.") from None
    records: list[dict] = []
    stack: list[tuple[int, int, str]] = []
    line = heading + 1
    last_string = previous = ""
    for found in _TOKEN.finditer(source):
        token = found.group()
        mark = token[0]
        if mark == "\n":
            line += 1
            continue
        if mark == '"':
            last_string, previous = token, '"'
            continue
        if mark == "-" or "0" <= mark <= "9":
            value = float(token)
            if _unsafe(value) or _decimal_key(token) != _decimal_key(repr(value)):
                raise JsonRecordError(f"{path} contains a JSON number outside the supported finite, safe numeric "
                                      "range; find cannot report it exactly. Use open to inspect the source.")
            previous = "number"
            continue
        if mark == "{":
            if len(stack) >= DEEPEST:
                raise deep
            stack.append((found.start(), line, last_string if previous == ":" else ""))
        elif mark == "}":
            offset, start, key = stack.pop()
            raw = source[offset:found.end()]
            try:
                value = _parse(raw)
            except RecursionError:
                raise deep from None
            pairs = entries(value)
            own = [(name, item) for name, item in pairs if _scalar(item)]
            search = "\n".join([key, stringify(dict(own), "  "), *([raw] if len(own) == len(pairs) else [])]).lower()
            # Where the object it stands in starts, and the key it stands under there, or "" for a list's item or
            # the top (core's parent and key).
            records.append({"start": start, "end": line, "offset": offset, "value": value, "search": search,
                            "parent": stack[-1][0] if stack else None, "key": _parse(key) if key else ""})
        previous = mark
    records.sort(key=lambda entry: entry["offset"])
    return records
