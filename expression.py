"""Arithmetic over a record's fields, for find's where, fields, sum, sort and group (core's expression.ts): field
names and numbers with + - * /, parentheses, round(x, n) and abs(x) ("total != round(quantity * price, 2)",
"amount / hours > 950").

Asked which of thousands of readings had been charged at the wrong price, two agents wrote "beloeb_kr / kwh = 2.44"
and "beloeb_kr = 2.44 * kwh" as where, were told no record could be compared, tried "beloeb_kr - kwh - kwh" in place
of a product, and then multiplied page after page by hand until they ran out of time (2026-11-14). Nothing here
knows a field: names are read from each record by the caller.

len, sum, min and max of one field that holds a list (a JSON array) count or add up its items, a list of objects by
one key of each ("sum(pumper.kw)"); min and max of several values take the least or the greatest. Asked which
stations had more than three pumps, agents read every station's list and counted it themselves (2026-11-14).

item of a list and a place gives the one item there, counted from 1 ("item(maalinger, 600)"). Asked for the 600th of
820 measurements in one log entry, every agent that answered wrote the list to a file and ran a script over it, and
one whose script call ended its turn gave no answer (2026-11-15).

An expression is a tuple: ("number", value), ("field", name), ("negate", of), ("operation", op, left, right) or
("call", name, args)."""

from __future__ import annotations

import math
import re

from . import json_records

# What JavaScript's \s takes for a space, as core splits and matches by it (tools.py's _JS_SPACE), a character each.
SPACE = "".join(map(chr, [9, 10, 11, 12, 13, 32, 0xa0, 0x1680, *range(0x2000, 0x200b), 0x2028, 0x2029, 0x202f, 0x205f,
                          0x3000, 0xfeff]))
_SPACES = re.compile(f"[{SPACE}]+")
# A number as written in an expression: digits, with one decimal point or comma between digits.
_NUMBER = re.compile(r"[0-9]+(?:[.,][0-9]+)?")
# A field's name: a letter or underscore first, then letters, digits, underscores, points and spaces between words.
_NAME = re.compile(r"[^\W\d][\w.]*(?: +\w[\w.]*)*")
# Whether a text can be arithmetic at all: an operator or a bracket, or a minus with a space on each side.
_ARITHMETIC = re.compile(rf"[*/+()]|[{SPACE}]-[{SPACE}]")
# The functions an expression may call.
FUNCTIONS = frozenset({"round", "abs", "len", "sum", "min", "max", "item"})
# The functions that read a field's list when they are given one field.
OF_LIST = frozenset({"len", "sum", "min", "max"})


def _tokens(text: str) -> list[tuple] | None:
    out: list[tuple] = []
    rest = text.strip(SPACE)
    while rest:
        space = _SPACES.match(rest)
        if space is not None:
            rest = rest[space.end():]
            continue
        # A comma between digits is a decimal comma (2,44); round(total, 2) and round(total,2) are two arguments.
        number = _NUMBER.match(rest)
        if number is not None:
            out.append(("number", float(number.group(0).replace(",", "."))))
            rest = rest[number.end():]
            continue
        name = _NAME.match(rest)
        if name is not None:
            out.append(("name", name.group(0).strip()))
            rest = rest[name.end():]
            continue
        symbol = rest[0]
        if symbol not in "+-*/(),":
            return None
        out.append(("symbol", symbol))
        rest = rest[1:]
    return out


def arithmetic_of(text: str) -> tuple | None:
    """The expression a text writes, or None when it is no arithmetic: a plain field's name, a number alone, or
    anything that does not parse. A field with a minus in its name is the caller's to find first."""
    if _ARITHMETIC.search(text) is None:
        return None
    tokens = _tokens(text)
    if not tokens:
        return None
    at = 0

    def symbol(wanted: str) -> bool:
        nonlocal at
        if at < len(tokens) and tokens[at] == ("symbol", wanted):
            at += 1
            return True
        return False

    def chain(operand, operators: str) -> tuple | None:
        nonlocal at
        left = operand()
        while left is not None:
            token = tokens[at] if at < len(tokens) else None
            if token is None or token[0] != "symbol" or token[1] not in operators:
                break
            at += 1
            right = operand()
            if right is None:
                return None
            left = ("operation", token[1], left, right)
        return left

    def total() -> tuple | None:
        return chain(product, "+-")

    def product() -> tuple | None:
        return chain(unary, "*/")

    def unary() -> tuple | None:
        if symbol("-"):
            of = unary()
            return None if of is None else ("negate", of)
        return atom()

    def atom() -> tuple | None:
        nonlocal at
        token = tokens[at] if at < len(tokens) else None
        if token is None:
            return None
        if token[0] == "number":
            at += 1
            return token
        if token[0] == "name":
            at += 1
            called = token[1].lower()
            if called in FUNCTIONS and symbol("("):
                args: list[tuple] = []
                while True:
                    arg = total()
                    if arg is None:
                        return None
                    args.append(arg)
                    if not symbol(","):
                        break
                if not symbol(")"):
                    return None
                fits = (len(args) <= 2 if called == "round" else len(args) == 1 if called in ("abs", "len", "sum")
                        else len(args) == 2 and args[0][0] == "field" if called == "item" else len(args) <= 12)
                return ("call", called, args) if fits else None
            return ("field", token[1])
        if symbol("("):
            inner = total()
            return inner if inner is not None and symbol(")") else None
        return None

    expression = total()
    if expression is None or at != len(tokens):
        return None
    # A name alone is a field, and a number alone a number: no arithmetic to do.
    return None if expression[0] in ("field", "number") else expression


def fields_of_expression(expression: tuple) -> list[str]:
    """The fields an expression reads."""
    kind = expression[0]
    if kind == "number":
        return []
    if kind == "field":
        return [expression[1]]
    if kind == "negate":
        return fields_of_expression(expression[1])
    if kind == "operation":
        return [*fields_of_expression(expression[2]), *fields_of_expression(expression[3])]
    return [field for arg in expression[2] for field in fields_of_expression(arg)]


def _precise(value: float) -> float:
    """Binary noise off, as decimals would write it (JavaScript's Number(x.toPrecision(15))): 0.1 + 0.2 is 0.3."""
    return float(f"{value:.15g}")


def round_half_up(value: float, places: int) -> float:
    """Rounded half away from zero, as amounts are: round(2.345, 2) is 2.35, round(-2.345, 2) is -2.35."""
    factor = 10 ** places
    # The product as decimals write it, before rounding: 1.005 * 100 is 100.49999999999999 in binary.
    scaled = _precise(abs(value) * factor)
    whole = math.floor(scaled)
    rounded = whole + 1 if scaled - whole >= 0.5 else whole
    return math.copysign(rounded / factor, value) if value != 0 else 0.0


def evaluate(expression: tuple, value, items=lambda name, count: None) -> float | None:
    """The expression's value for one record, or None when a field it reads has no number there, or it divides by
    zero. `value(name)` is a field's number in this record, or None. `items(name, count)` is the numbers a field's
    list holds in this record (each item, or each item's key after a point: "pumper.kw"), its length for len (when
    `count`), or None when the field holds no list."""
    kind = expression[0]
    if kind == "number":
        return expression[1]
    if kind == "field":
        return value(expression[1])
    if kind == "negate":
        of = evaluate(expression[1], value, items)
        return None if of is None else -of
    if kind == "operation":
        _, op, first, second = expression
        left, right = evaluate(first, value, items), evaluate(second, value, items)
        if left is None or right is None:
            return None
        if op == "/" and right == 0:
            return None
        return _precise(left + right if op == "+" else left - right if op == "-" else left * right if op == "*"
                        else left / right)
    _, name, args = expression
    only = args[0] if len(args) == 1 else None
    if name in OF_LIST and only is not None and only[0] == "field":
        held = items(only[1], name == "len")
        if name == "len":
            return held if isinstance(held, (int, float)) and not isinstance(held, bool) else None
        if isinstance(held, list):
            if not held:
                return 0 if name == "sum" else None
            return _precise(sum(held) if name == "sum" else min(held) if name == "min" else max(held))
    if name == "len":
        return None
    if name == "item":
        of, place = args
        held = items(of[1], False) if of[0] == "field" else None
        at = evaluate(place, value, items)
        # Counted from 1, as people number them; no item there is no value.
        return (held[int(at) - 1] if isinstance(held, list) and at is not None and float(at).is_integer()
                and 1 <= at <= len(held) else None)
    values = [evaluate(arg, value, items) for arg in args]
    if any(entry is None for entry in values):
        return None
    if name in ("abs", "sum"):
        return abs(values[0]) if name == "abs" else values[0]
    if name == "min":
        return min(values)
    if name == "max":
        return max(values)
    places = values[1] if len(values) > 1 else 0
    return round_half_up(values[0], int(places)) if float(places).is_integer() and 0 <= places <= 10 else None


def number_text(value: float) -> str:
    """A computed number as find shows it: a decimal point, no binary noise, no trailing zeros."""
    return json_records.number_text(_precise(value))
