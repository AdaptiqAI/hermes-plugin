"""find's where (core's reading.ts: whereOf, conditionOf, comparison, keepWhere): what find keeps of the records it
found, those that meet every condition, compared here and not by a model's eye. Asked how many of 360 measurements
stayed above their limit after an intervention, a model read all 360, three pages in one turn, and the turn was
refused as more than its live window holds.

A condition is a dict: its `field`, its `op` (<, <=, >, >=, =, !=, in, not in), the `other` side as written and
whether that was `quoted` (a text, never a field's name); for in and not in the `values` listed, without their
quotes, or the `select` they are read from ({field, document, where}) and, once it is read, what it `found`
({texts, numbers, records})."""

from __future__ import annotations

import re

from . import expression, knowledge, record

# The most conditions one where takes; more are refused, never left out.
WHERE_CONDITIONS = 12
_S = f"[{expression.SPACE}]"
# What JavaScript's "." takes: any character but a line's end.
_DOT = "[^\n\r" + chr(0x2028) + chr(0x2029) + "]"
# A number or a date as written: the other side of a condition, never a field's name.
_LITERAL = re.compile(r"-?[0-9][0-9.,]*\Z|[0-9]{4}-[0-9]{2}-[0-9]{2}")
# A date as ISO writes it, at the start of a value, with its time if it has one: compared as dates are.
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:[T ][0-9]{2}:[0-9]{2}(?::[0-9]{2})?)?")
_SELECT = re.compile(rf"({_DOT}+?){_S}+(not{_S}+in|in){_S}*\({_S}*select{_S}+({_DOT}+?){_S}+from{_S}+({_DOT}+?)"
                     rf"(?:{_S}+where{_S}+([\s\S]+?))?{_S}*\)\Z", re.IGNORECASE)
_LISTED = re.compile(rf"({_DOT}+?){_S}+(not{_S}+in|in){_S}*\(([\s\S]*)\)\Z", re.IGNORECASE)
_COMPARED = re.compile(rf"({_DOT}+?){_S}*(<=|>=|!=|<>|==|=|<|>){_S}*({_DOT}+)\Z")
_QUOTED = re.compile(rf"([\"'])({_DOT}*)\1\Z")
_AND = re.compile(rf"{_S}+and{_S}+", re.IGNORECASE)
_PLAIN_NUMBER = re.compile(r"-?[0-9]+(?:[.,][0-9]+)?\Z")
_TRIMMED = record.JS_TRIMMED


def _fail(message: str):
    raise knowledge.KnowledgeError(message)


def _quoted(name: str) -> str:
    return f'"{name}"'


def _bare(text: str) -> str:
    """A name without the quotes around it."""
    text = text.strip(_TRIMMED)
    found = _QUOTED.match(text)
    return text if found is None else found.group(2)


def listed_values(written: str) -> list[str]:
    """The values of an in list, a comma between them, each with or without quotes: 'A-1', B-2, "C 3"."""
    values: list[str] = []
    value = ""
    quote = None
    for char in written:
        if quote is not None:
            if char == quote:
                quote = None
            else:
                value += char
        elif char in "\"'":
            quote = char
        elif char == ",":
            values.append(value.strip(_TRIMMED))
            value = ""
        else:
            value += char
    values.append(value.strip(_TRIMMED))
    return [entry for entry in values if entry != ""]


def where_of(value) -> list[dict]:
    """What find keeps of the records it found: those that meet every condition, up to WHERE_CONDITIONS, each a
    field, a comparison and a number, a date, a text in quotes or another field ("after > upper_limit",
    "beløb >= 10000", "status = 'open'"), either side arithmetic of fields and numbers (expression.py), or a field in
    or not in a list of values, as a list or one string with semicolons between. Asked for one of several customers,
    an agent wrote one condition for each, and every record failed one of them."""
    parts = [part.strip(_TRIMMED) for written in (value if isinstance(value, list) else [value])
             if isinstance(written, str) for part in written.split(";")]
    parts = [part for part in parts if part != ""]
    # Five conditions kept as four answered a question the agent never asked.
    if len(parts) > WHERE_CONDITIONS:
        _fail(f"where takes up to {WHERE_CONDITIONS} conditions, every one of which a record meets; {len(parts)} were "
              "given. Ask with fewer, or a field in a list of values (\"customer in ('A', 'B')\") where several are "
              "any of one field's values.")
    return [condition_of(part, True) for part in parts]


def and_parts(text: str) -> list[str]:
    """The parts of a select's where, cut at "and" outside quotes."""
    parts: list[str] = []
    part = ""
    quote = None
    at = 0
    while at < len(text):
        char = text[at]
        if quote is not None:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        else:
            joined = _AND.match(text, at)
            if joined is not None and part.strip(_TRIMMED) != "":
                parts.append(part.strip(_TRIMMED))
                part = ""
                at = joined.end()
                continue
        part += char
        at += 1
    parts.append(part.strip(_TRIMMED))
    return [entry for entry in parts if entry != ""]


def condition_of(part: str, selects: bool) -> dict:
    """One condition of where. A field in or not in a select takes the values of another document's records:
    "kunde_id in (select kunde_id from kunder.csv where tarif = 'Nat')"; its own conditions have "and" between, and
    hold no select."""
    select = _SELECT.match(part)
    if select is not None:
        if not selects:
            _fail(f"A select in where holds no select of its own: {_quoted(part)}.")
        own = [] if select.group(5) is None else [condition_of(inner, False) for inner in and_parts(select.group(5))]
        if len(own) > WHERE_CONDITIONS:
            _fail(f"A select in where takes up to {WHERE_CONDITIONS} conditions; {len(own)} were given.")
        return {"field": _bare(select.group(1)), "op": "not in" if select.group(2)[:3].lower() == "not" else "in",
                "other": "", "quoted": False,
                "select": {"field": _bare(select.group(3)), "document": _bare(select.group(4)), "where": own}}
    listed = _LISTED.match(part)
    if listed is not None:
        values = listed_values(listed.group(3))
        if not values:
            _fail(f"where's in takes the values between brackets, a comma between them: {_quoted(part)} lists none.")
        return {"field": _bare(listed.group(1)), "op": "not in" if listed.group(2)[:3].lower() == "not" else "in",
                "other": "", "quoted": False, "values": values}
    found = _COMPARED.match(part)
    if found is None:
        _fail("where takes a field, a comparison (<, <=, >, >=, =, !=) and a number, a date, a text in quotes or "
              "another field, such as \"after > upper_limit\" or \"status = 'open'\"; "
              f"{_quoted(part)} is none of these.")
    other = found.group(3).strip(_TRIMMED)
    text = _QUOTED.match(other)
    op = {"==": "=", "<>": "!="}.get(found.group(2), found.group(2))
    return {"field": _bare(found.group(1)), "op": op, "other": other if text is None else text.group(2),
            "quoted": text is not None}


def conditions_text(conditions: list[dict]) -> str:
    """Conditions as a find's first line names them."""
    said = []
    for condition in conditions:
        select = condition.get("select")
        if select is not None:
            own = f" where {conditions_text(select['where'])}" if select["where"] else ""
            found = condition.get("found")
            read = "" if found is None else f": {len(found['texts'])} values from {found['records']} records"
            said.append(f"{condition['field']} {condition['op']} (select {select['field']} from "
                        f"{select['document']}{own}{read})")
        elif "values" in condition:
            values = condition["values"]
            first = ", ".join(f"'{value}'" for value in values[:3])
            said.append(f"{condition['field']} {condition['op']} ({first}"
                        f"{f', … {len(values)} values' if len(values) > 3 else ''})")
        else:
            other = f"'{condition['other']}'" if condition["quoted"] else condition["other"]
            said.append(f"{condition['field']} {condition['op']} {other}")
    return " and ".join(said)


def _listed_equals(value: str, number: dict | None, listed: str) -> bool:
    """One value of an in list against a record's: as numbers when both are, else as text in any case."""
    plain = float(listed.replace(",", ".")) if _PLAIN_NUMBER.match(listed) else None
    if number is not None and plain is not None:
        return number["value"] == plain
    return value.strip(_TRIMMED).lower() == listed.strip(_TRIMMED).lower()


def _in_found(found: dict, value: str, number: dict | None) -> bool:
    """Whether a record's value is one a select read: as numbers when both are, else as text in any case."""
    if number is not None and number["value"] in found["numbers"]:
        return True
    return value.strip(_TRIMMED).lower() in found["texts"]


def comparison(left: str, left_number: dict | None, right: str, right_number: dict | None, op: str) -> bool | None:
    """One comparison of where: numbers as numbers, dates as dates, else equal or not equal as text in any case.
    None when the two can't be compared: a text against a number with < or >."""
    left_date, right_date = _ISO_DATE.match(left.strip(_TRIMMED)), _ISO_DATE.match(right.strip(_TRIMMED))
    if left_number is not None and right_number is not None:
        by = (left_number["value"] > right_number["value"]) - (left_number["value"] < right_number["value"])
    elif left_date is not None and right_date is not None:
        by = (left_date.group(0) > right_date.group(0)) - (left_date.group(0) < right_date.group(0))
    elif op in ("=", "!="):
        by = 0 if left.strip(_TRIMMED).lower() == right.strip(_TRIMMED).lower() else 1
    else:
        return None
    return (by < 0 if op == "<" else by <= 0 if op == "<=" else by > 0 if op == ">" else by >= 0 if op == ">="
            else by == 0 if op == "=" else by != 0)


def found_of(values: list[tuple]) -> dict:
    """What a select read: its field's values over the records kept, each `(document, table, value, kind)`, as
    texts in lower case and as numbers, and how many records gave them."""
    texts: set[str] = set()
    numbers: set[float] = set()
    for _, _, value, kind in values:
        if value is None or kind == "null":
            continue
        texts.add(value.strip(_TRIMMED).lower())
        if _PLAIN_NUMBER.match(value.strip(_TRIMMED)):
            numbers.add(float(value.strip(_TRIMMED).replace(",", ".")))
    return {"texts": texts, "numbers": numbers, "records": len(values)}


def keep_where(records: list[dict], values_of, names_of, numbers_of, conditions: list[dict]) -> int:
    """Keeps the records that meet every condition of where, in place (core's keepWhere). `values_of(field)` is the
    field's `(document, table, value, kind)` in each record, `names_of(record)` the fields a record names, as it
    names them, and `numbers_of(values)` their numbers. Returns how many records could not be compared."""
    meets = [True] * len(records)
    compared = [True] * len(records)
    # The other side is a field only when a record names one exactly so, in its own case: "status != BETALT"
    # compares with the text BETALT, though "betalt" is a field of the same records (the day it was paid).
    names = {name for entry in records[:200] for name in names_of(entry)}

    def is_name(name: str) -> bool:
        return name in names or ("-" in name and all(part.strip(_TRIMMED) in names for part in name.split("-")[:2]))

    known = {name.lower() for name in names}
    for condition in conditions:
        left = values_of(condition["field"])
        left_numbers = numbers_of(left)
        if "select" in condition and "found" not in condition:
            _fail("A select in where reads the documents of the collections used here; it works in find with "
                  "document.")
        if "values" in condition or "found" in condition:
            for index, (_, _, value, kind) in enumerate(left):
                if value is None or kind == "null":
                    verdict = None
                else:
                    held = (_in_found(condition["found"], value, left_numbers[index]) if "found" in condition
                            else any(_listed_equals(value, left_numbers[index], entry)
                                     for entry in condition["values"]))
                    verdict = held == (condition["op"] == "in")
                if verdict is None:
                    compared[index] = False
                if verdict is not True:
                    meets[index] = False
            continue
        other = condition["other"]
        is_field = (not condition["quoted"] and _LITERAL.match(other) is None
                    and (is_name(other) or expression.arithmetic_of(other) is not None))
        right = values_of(other) if is_field else [(doc, table, other, None) for doc, table, _, _ in left]
        right_numbers = numbers_of(right)
        # A field the records have, left empty, is an empty text to = '' and != '': "note != ''" finds the records
        # with a note.
        empty = (condition["quoted"] and other == "" and condition["op"] in ("=", "!=")
                 and condition["field"].lower() in known)
        for index, (_, _, own, kind) in enumerate(left):
            _, _, against, against_kind = right[index]
            value = "" if own is None and empty else own
            # A JSON null is no value to compare.
            verdict = (None if value is None or kind == "null" or against is None or against_kind == "null"
                       else comparison(value, left_numbers[index], against, right_numbers[index], condition["op"]))
            if verdict is None:
                compared[index] = False
            if verdict is not True:
                meets[index] = False
    uncompared = sum(1 for done in compared if not done)
    records[:] = [entry for index, entry in enumerate(records) if meets[index]]
    return uncompared
