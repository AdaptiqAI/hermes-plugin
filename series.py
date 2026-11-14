"""Series of documents, for contents (core's series.ts).

Documents whose names differ only in a number, a date or a month are one line that names the first and the last, how
many there are and every one that is missing. A year of monthly exports was 24 lines that said one thing. Each line
is written only when it alone gives back exactly its documents, so every name can still be told from it. The rules
read file names only.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from typing import Callable

# How many documents a series has before contents names it on one line, and how many missing ones a line names.
SERIES_MIN = 4
SERIES_GAPS = 6

# Month names as file names write them, in Danish and English, whole and short.
MONTHS = [words.split(" ") for words in (
    "januar februar marts april maj juni juli august september oktober november december",
    "january february march april may june july august september october november december",
    "jan feb mar apr maj jun jul aug sep okt nov dec",
    "jan feb mar apr may jun jul aug sep oct nov dec",
)]
# Longest first, as core's alternation tries them.
_MONTH_WORDS = [re.compile(re.escape(word), re.IGNORECASE)
                for word in sorted(dict.fromkeys(word for names in MONTHS for word in names), key=len, reverse=True)]
_NUMBER = re.compile(r"[0-9]+")
_DIGITS = "0123456789"
# A line of text as JavaScript's . reads it: no line terminator.
_DOT = r"[^\n\r  ]"
_SERIES = re.compile(
    rf"- ({_DOT}+?) to ({_DOT}+?): ([0-9]+) documents, (one a month|one a day|numbered in a row|every pair of "
    rf"([0-9]+)-([0-9]+) and ([0-9]+)-([0-9]+))( \(day-month-year\))?, [0-9-]+ lines each\.({_DOT}*)\Z")
_PLAIN = re.compile(rf"- ({_DOT}+) \([0-9]+ lines\)\Z")
_NONE = re.compile(rf" None for ({_DOT}+?)\.(?: In find, {_DOT}*)?\Z")
_EPOCH = datetime.date(1970, 1, 1)


def _letter_or_digit(char: str) -> bool:
    """core's [\\p{L}\\d]: a letter of any script, or 0-9."""
    return char.isalpha() or char in _DIGITS


def _token_end(path: str, at: int) -> int | None:
    """Where a number or a month name standing alone starts at `at` ends (core's TOKEN), else None."""
    number = _NUMBER.match(path, at)
    if number is not None:
        return number.end()
    if at > 0 and path[at - 1].isalpha():
        return None
    for word in _MONTH_WORDS:
        found = word.match(path, at)
        if found is not None and not (found.end() < len(path) and path[found.end()].isalpha()):
            return found.end()
    return None


def parts_of(path: str) -> list[str]:
    """A name cut into its text and, at the odd places, its numbers and month names."""
    parts, start, at = [], 0, 0
    while at < len(path):
        end = _token_end(path, at)
        if end is None:
            at += 1
            continue
        parts += [path[start:at], path[at:end]]
        start = at = end
    return [*parts, path[start:]]


def _tokens_of(parts: list[str]) -> list[int]:
    return list(range(1, len(parts), 2))


def _is_number(token: str) -> bool:
    return token[:1] != "" and token[0] in _DIGITS


def _units(text: str) -> int:
    """A text's length as JavaScript counts it, in UTF-16 code units."""
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _js_sorted(names) -> list[str]:
    """Names in the order JavaScript's sort gives them: by UTF-16 code units."""
    return sorted(names, key=lambda name: name.encode("utf-16-be", "surrogatepass"))


def _case_of(word: str) -> str | None:
    if word == word.lower():
        return "lower"
    if word == word.upper():
        return "upper"
    return "title" if word == word[0].upper() + word[1:].lower() else None


def _cased(word: str, how: str | None) -> str:
    return word.upper() if how == "upper" else word[0].upper() + word[1:] if how == "title" else word


def _day_of(y: int, m: int, d: int) -> int | None:
    # JavaScript's Date reads a year below 100 as 19xx, so core finds no such day.
    if y < 100:
        return None
    try:
        return (datetime.date(y, m, d) - _EPOCH).days
    except ValueError:
        return None


def _date_of(day: int) -> tuple[int, int, int]:
    date = _EPOCH + datetime.timedelta(days=day)
    return date.year, date.month, date.day


def _year(value: int) -> str:
    return str(value).rjust(4, "0")


@dataclass(frozen=True)
class Calendar:
    """The date in a series' names: which tokens it spans, and how a day or a month is read and written."""
    kind: str
    # Said after the kind when a name alone can't tell the order.
    order: str
    first: int
    last: int
    value: Callable[[list[str]], int]
    # The date's tokens for a value, or None when its name is not certain (maj or may).
    render: Callable[[int], dict[int, str] | None]


def _calendar_of(rows: list[list[str]], varying: list[int], told: dict | None = None) -> Calendar | None:
    """The date in names that change only in it, if they are dated (core's calendarOf). Year-first (2043-03-15) and
    day-first (15-03-2043, as Danish writes it) are told apart by where the four-digit year stands. Day-first is told
    from month-first (03-15-2043) by a day above 12, and without one the order is not guessed. `told`, unfolding a
    line: its kind and order, so its first and last name suffice."""
    row0 = rows[0]

    def n(row: list[str], token: int) -> int:
        return int(row[token])

    def digits(token: int, *widths: int) -> bool:
        return 0 < token < len(row0) and all(_is_number(row[token]) and len(row[token]) in widths for row in rows)

    def mark_after(token: int) -> str:
        return row0[token + 1] if token + 1 < len(row0) else ""

    def one_mark(token: int) -> bool:
        mark = mark_after(token)
        return _units(mark) == 1 and not _letter_or_digit(mark)

    def covers(span: list[int]) -> bool:
        return all(token in span for token in varying)

    def changes(*tokens: int) -> bool:
        return told is not None or all(token in varying for token in tokens)

    several = told is not None or len(varying) >= 2
    days = told is None or told["kind"] != "one a month"
    months = told is None or told["kind"] != "one a day"

    def pad_of(token: int) -> Callable[[int], str] | None:
        """A day or a month as the names write it: two digits, or no leading zero; both seen, neither."""
        texts = [row[token] for row in rows]
        short = any(len(text) == 1 for text in texts)
        if short and any(text.startswith("0") for text in texts):
            return None
        return str if short else (lambda value: str(value).rjust(2, "0"))

    def monthly(token: int) -> bool:
        return all(1 <= n(row, token) <= 12 for row in rows)

    def dated(y: int, m: int, d: int) -> bool:
        return all(_day_of(n(row, y), n(row, m), n(row, d)) is not None for row in rows)

    def a_day(y: int, m: int, d: int, pad_y, pad_m, pad_d, order: str, first: int, last: int) -> Calendar:
        def render(value: int) -> dict[int, str]:
            yy, mm, dd = _date_of(value)
            return {y: pad_y(yy), m: pad_m(mm), d: pad_d(dd)}
        return Calendar("one a day", order, first, last, lambda row: _day_of(n(row, y), n(row, m), n(row, d)), render)

    def a_month(m: int, y: int, pad_m, first: int, last: int) -> Calendar:
        return Calendar("one a month", "", first, last, lambda row: n(row, y) * 12 + n(row, m) - 1,
                        lambda value: {y: _year(value // 12), m: pad_m(value % 12 + 1)})

    for t in _tokens_of(row0):
        # Year first: 2043-03-15, a day; 2043-03, a month.
        if digits(t, 4) and digits(t + 2, 1, 2) and one_mark(t):
            y, m, d = t, t + 2, t + 4
            pm, pd = pad_of(m), pad_of(d) if digits(d, 1, 2) else None
            if (days and pm is not None and pd is not None and mark_after(m) == mark_after(y) and covers([y, m, d])
                    and changes(d) and several and dated(y, m, d)):
                return a_day(y, m, d, _year, pm, pd, "", y, d)
            if months and pm is not None and covers([y, m]) and changes(y, m) and monthly(m):
                return a_month(m, y, pm, y, m)
        # Day first: 15-03-2043, 15.03.2043.
        if (days and digits(t, 1, 2) and digits(t + 2, 1, 2) and digits(t + 4, 4) and one_mark(t)
                and mark_after(t + 2) == mark_after(t)):
            d, m, y = t, t + 2, t + 4
            pd, pm = pad_of(d), pad_of(m)
            day_first = any(n(row, d) > 12 for row in rows) if told is None else told["day_first"]
            if (pd is not None and pm is not None and day_first and covers([d, m, y]) and changes(d) and several
                    and dated(y, m, d)):
                return a_day(y, m, d, _year, pm, pd, " (day-month-year)", d, y)
        # A month and a year: 03-2043.
        if months and digits(t, 1, 2) and digits(t + 2, 4) and one_mark(t):
            m, y = t, t + 2
            pm = pad_of(m)
            if pm is not None and covers([m, y]) and changes(m) and monthly(m):
                return a_month(m, y, pm, m, y)
        # A month's name, with a year before or after it, or none: marts 2043, 2043 marts, Mar-2043, januar.
        if months and not _is_number(row0[t]):
            words = [row[t].lower() for row in rows]
            languages = [names for names in MONTHS if all(word in names for word in words)]
            how = _case_of(row0[t])
            if not languages or how is None or any(_case_of(row[t]) != how for row in rows):
                continue
            year = next((at for at in (t + 2, t - 2) if digits(at, 4) and 1 <= len(row0[min(t, at) + 1]) <= 3
                         and not any(map(_letter_or_digit, row0[min(t, at) + 1]))), None)
            span = [t] if year is None else [t, year]
            if not covers(span) or not changes(t):
                continue
            return _named_months(t, year, languages, how, span, n)
    return None


def _named_months(t: int, y: int | None, languages: list[list[str]], how: str, span: list[int], n) -> Calendar:
    def month(row: list[str]) -> int:
        return languages[0].index(row[t].lower())

    def render(value: int) -> dict[int, str] | None:
        # Written only when every language the names may be in writes it alike.
        written = {language[value % 12] for language in languages}
        if len(written) != 1:
            return None
        tokens = {t: _cased(next(iter(written)), how)}
        if y is not None:
            tokens[y] = _year(value // 12)
        return tokens

    return Calendar("one a month", "", min(span), max(span),
                    lambda row: (0 if y is None else n(row, y) * 12) + month(row), render)


def _span(least: int, most: int) -> str:
    return str(least) if least == most else f"{least}-{most}"


def contents_lines(files: list[dict], pattern_of: Callable[[list[str]], str | None] = lambda paths: None) -> list[str]:
    """A collection's documents as contents lists them, series folded (core's contentsLines).

    Args:
        files: the documents of one page (`path`, `lines`), in the order listed.
        pattern_of: a pattern find takes for exactly these documents of the page, or None.

    Returns:
        The lines.
    """
    parts = [parts_of(file["path"]) for file in files]
    # Alike but for their numbers and month names. A four-digit number, a year, stays its own: 25.01.2043 is not
    # 2043.01.25.
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(parts):
        key = "\u0001".join(part if at % 2 == 0 else "\u0000m" if not _is_number(part)
                            else "\u0000y" if len(part) == 4 else "\u0000n" for at, part in enumerate(row))
        groups.setdefault(key, []).append(index)
    folded: dict[int, str] = {}
    hidden: set[int] = set()

    def fold(members: list[int], head: str, missing: list[str]) -> None:
        pattern = pattern_of([files[index]["path"] for index in members])
        line = " ".join([head, *([f"None for {', '.join(missing)}."] if missing else []),
                         *([f'In find, document: "{pattern}" names all of them.'] if pattern is not None else [])])
        try:
            names = series_names(line)
        except Exception:  # a line that does not unfold is not written
            names = []
        if names != _js_sorted(files[index]["path"] for index in members):
            return
        first = min(members)
        folded[first] = line
        hidden.update(index for index in members if index != first)

    def line_counts(members: list[int]) -> str:
        lengths = [files[index]["lines"] for index in members]
        return _span(min(lengths), max(lengths))

    def grid(members: list[int], rows: list[list[str]], a: int, b: int) -> None:
        """Two numbers that change, when the names are every pair of two ranges: servicelog-2042-K1 to
        servicelog-2043-K4."""
        pairs = [(int(row[a]), int(row[b])) for row in rows]
        firsts = sorted({pair[0] for pair in pairs})
        seconds = sorted({pair[1] for pair in pairs})

        def whole(values: list[int]) -> bool:
            return values[-1] - values[0] + 1 == len(values)

        if (len(firsts) < 2 or len(seconds) < 2 or not whole(firsts) or not whole(seconds)
                or len(firsts) * len(seconds) != len(members) or len(set(pairs)) != len(members)):
            return

        def width(token: int) -> int:
            lengths = {len(row[token]) for row in rows}
            return next(iter(lengths)) if len(lengths) == 1 else 0

        def ranged(values: list[int], token: int) -> str:
            return f"{str(values[0]).rjust(width(token), '0')}-{str(values[-1]).rjust(width(token), '0')}"

        order = sorted(zip(members, pairs), key=lambda entry: entry[1])
        fold(members, f"- {files[order[0][0]]['path']} to {files[order[-1][0]]['path']}: {len(members)} documents, "
                      f"every pair of {ranged(firsts, a)} and {ranged(seconds, b)}, {line_counts(members)} lines each.",
             [])

    for members in groups.values():
        if len(members) < SERIES_MIN:
            continue
        rows = [parts[index] for index in members]
        row0 = rows[0]
        varying = [token for token in _tokens_of(row0) if any(row[token] != row0[token] for row in rows)]
        if not varying or any(_is_number(row[token]) and len(row[token]) > 9 for row in rows for token in varying):
            continue
        calendar = _calendar_of(rows, varying)
        if calendar is not None:
            value_of = calendar.value

            def written(value: int, calendar: Calendar = calendar, row0: list[str] = row0) -> str | None:
                tokens = calendar.render(value)
                return None if tokens is None else "".join(
                    tokens.get(calendar.first + at, part) for at, part in enumerate(row0[calendar.first:calendar.last + 1]))
            kind = f"{calendar.kind}{calendar.order}"
        elif len(varying) == 1 and _is_number(row0[varying[0]]):
            # One number: written to one width, or with no leading zero (Rapport_1 to Rapport_120).
            token = varying[0]
            widths = {len(row[token]) for row in rows}
            width = (next(iter(widths)) if len(widths) == 1
                     else 0 if all(re.match(r"0[0-9]", row[token]) is None for row in rows) else -1)
            if width == -1:
                continue

            def value_of(row: list[str], token: int = token) -> int:
                return int(row[token])

            def written(value: int, width: int = width) -> str | None:
                return str(value).rjust(width, "0")
            kind = "numbered in a row"
        else:
            if len(varying) == 2 and all(_is_number(row0[token]) for token in varying):
                grid(members, rows, varying[0], varying[1])
            continue
        order = sorted(({"index": index, "value": value_of(rows[at])} for at, index in enumerate(members)),
                       key=lambda entry: entry["value"])
        if len({entry["value"] for entry in order}) != len(order):
            continue
        # Cut where a gap would take a line past SERIES_GAPS missing: invoices with a few voided, two binders
        # numbered apart.
        segments: list[list[dict]] = []
        missing_so_far = 0
        for entry in order:
            if segments and missing_so_far + entry["value"] - segments[-1][-1]["value"] - 1 <= SERIES_GAPS:
                missing_so_far += entry["value"] - segments[-1][-1]["value"] - 1
                segments[-1].append(entry)
            else:
                segments.append([entry])
                missing_so_far = 0
        for segment in segments:
            if len(segment) < SERIES_MIN:
                continue
            present = {entry["value"] for entry in segment}
            missing: list[str] = []
            certain = written(segment[0]["value"]) is not None and written(segment[-1]["value"]) is not None
            for value in range(segment[0]["value"], segment[-1]["value"] + 1):
                if not certain:
                    break
                if value in present:
                    continue
                text = written(value)
                if text is None:
                    certain = False
                else:
                    missing.append(text)
            if not certain:
                continue
            indices = [entry["index"] for entry in segment]
            fold(indices, f"- {files[segment[0]['index']]['path']} to {files[segment[-1]['index']]['path']}: "
                          f"{len(segment)} documents, {kind}, {line_counts(indices)} lines each.", missing)
    return [folded.get(index, f"- {file['path']} ({file['lines']} lines)")
            for index, file in enumerate(files) if index not in hidden]


def series_names(line: str) -> list[str]:
    """The names a line of contents stands for, read from the line alone, as a reader unfolds it (core's
    seriesNames): a document's own line gives its name, a series line every name from its first to its last but the
    missing ones.

    Args:
        line: one line of contents.

    Returns:
        The names, sorted.

    Raises:
        ValueError: the line is no line of contents, or does not give back the names it counts.
    """
    series = _SERIES.match(line)
    if series is None:
        plain = _PLAIN.match(line)
        if plain is None:
            raise ValueError(f"Not a line of contents: {line}")
        return [plain.group(1)]
    first, last, count, kind, a0, a1, b0, b1, day_first, rest = series.groups()
    none = _NONE.match(rest)
    missing = set() if none is None else set(none.group(1).split(", "))
    f, l = parts_of(first), parts_of(last)
    varying = [token for token in _tokens_of(f) if token >= len(l) or f[token] != l[token]]
    names: list[str] = []
    if kind.startswith("every pair"):
        if len(varying) < 2:
            raise ValueError(f"No two numbers change in: {line}")
        ta, tb = varying[0], varying[1]
        wa = len(a0) if len(a0) == len(a1) else 0
        wb = len(b0) if len(b0) == len(b1) else 0
        for x in range(int(a0), int(a1) + 1):
            for y in range(int(b0), int(b1) + 1):
                names.append("".join(str(x).rjust(wa, "0") if at == ta else str(y).rjust(wb, "0") if at == tb
                                     else part for at, part in enumerate(f)))
    elif kind == "numbered in a row":
        if len(varying) != 1:
            raise ValueError(f"Not one number changes in: {line}")
        token = varying[0]
        width = len(f[token]) if len(f[token]) == len(l[token]) else 0
        for value in range(int(f[token]), int(l[token]) + 1):
            text = str(value).rjust(width, "0")
            if text not in missing:
                names.append("".join(text if at == token else part for at, part in enumerate(f)))
    else:
        calendar = _calendar_of([f, l], varying, {"kind": kind, "day_first": day_first is not None})
        if calendar is None:
            raise ValueError(f"No date in: {line}")
        for value in range(calendar.value(f), calendar.value(l) + 1):
            tokens = calendar.render(value)
            if tokens is None:
                raise ValueError(f"A name can't be told from: {line}")
            text = "".join(tokens.get(calendar.first + at, part)
                           for at, part in enumerate(f[calendar.first:calendar.last + 1]))
            if text not in missing:
                names.append("".join(tokens.get(at, part) for at, part in enumerate(f)))
    if len(names) != int(count):
        raise ValueError(f"{len(names)} names, not {count}, in: {line}")
    return _js_sorted(names)
