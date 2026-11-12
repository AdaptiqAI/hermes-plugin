"""Tokens processed per conversation, apart from its memory size (core's processed.ts), for Hermes Desktop's memory
pill: every request of a turn reads the whole memory again, so a turn of many steps processes many times the memory.
Counted from Hermes' own usage of each request to AdaptiqAI (`post_api_request`), in its latest turn and in all."""
from __future__ import annotations

COUNTS = ("requests", "new_input", "from_memory", "written", "output", "reasoning")
NOTHING = {**dict.fromkeys(COUNTS, 0), "cost_usd": 0.0}


def step(usage: dict, cost) -> dict:
    """One request, from Hermes' canonical usage (new input apart from what was read from and saved to memory) and
    what it cost (None without prices)."""
    def count(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0

    return {"requests": 1, "new_input": count("input_tokens"), "from_memory": count("cache_read_tokens"),
            "written": count("cache_write_tokens"), "output": count("output_tokens"),
            "reasoning": count("reasoning_tokens"), "cost_usd": None if cost is None else float(cost)}


def _add(view: dict, one: dict) -> dict:
    added = {key: int(view.get(key) or 0) + one[key] for key in COUNTS}
    added["cost_usd"] = (None if view.get("cost_usd") is None or one["cost_usd"] is None
                         else float(view["cost_usd"]) + one["cost_usd"])
    return added


def count(record, one: dict, turn_id: str, now: float) -> dict:
    """A conversation's count with one more request: into the conversation, and into its latest turn, which starts
    afresh with another turn id."""
    record = record if isinstance(record, dict) else {}
    same = bool(turn_id) and record.get("turn_id") == turn_id and isinstance(record.get("turn"), dict)
    return {"turn_id": turn_id, "at": now, "since": record.get("since") or now,
            "turn": _add(record["turn"] if same else NOTHING, one),
            "conversation": _add(record.get("conversation") if isinstance(record.get("conversation"), dict)
                                 else NOTHING, one)}
