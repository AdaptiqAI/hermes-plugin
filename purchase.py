"""Getting lanes from inside Hermes (core's purchase.ts and offers.ts).

A signed-in Hermes cannot buy: billing needs a person in the browser. The plugin asks the gateway for
a purchase request, the workspace owner confirms it (and pays, by card or from the balance) on the
page it opens, and the plugin follows the request until the lanes appear in the workspace. Where the
gateway takes no purchase requests from this connection, the console's purchase page does the same.
"""

from __future__ import annotations

import hashlib
import time
import urllib.parse
import uuid

from . import adaptiq_core as core
from .errors import CONSOLE

SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£"}
# How long a purchase is followed: an operator may take a while to activate one.
FOLLOW_FOR = 3 * 3600
TERMINAL = {"denied", "canceled", "cancelled", "expired", "stopped", "failed", "provisioning_failed"}


class PurchaseError(Exception):
    """One sentence for the person."""


def _number(value) -> float | None:
    return value if isinstance(value, (int, float)) and value >= 0 else None


def offers() -> tuple[list[dict], str | None]:
    """The models on offer and how purchases are completed (`stripe` or `manual_activation`)."""
    try:
        status, body = core._call("GET", "/public/inference-offers", timeout=8)
    except (OSError, ValueError):
        status, body = 0, {}
    if status != 200:
        raise PurchaseError("The models on offer can't be read right now. Try again in a moment.")
    items = []
    for raw in body.get("items") or []:
        if not isinstance(raw, dict) or not raw.get("id") or raw.get("status", "available") != "available":
            continue
        speed = raw.get("tokens_per_second") if isinstance(raw.get("tokens_per_second"), dict) else {}
        minimum = max(1, round(_number(raw.get("minimum_lanes")) or 1))
        available = _number(raw.get("available_lanes"))
        items.append({
            "id": str(raw["id"]),
            "name": str(raw.get("name") or raw["id"]),
            "description": raw.get("tagline") or raw.get("description"),
            "price": (_number(raw.get("unit_amount")), str(raw.get("currency") or "").upper(),
                      str(raw.get("billing_period") or "hour")) if raw.get("currency") else None,
            "min_lanes": minimum,
            "max_lanes": max(minimum, round(_number(raw.get("maximum_lanes")) or 64)),
            "available_lanes": None if available is None else round(available),
            "speed": _number(speed.get("typical")) or _number(speed.get("full_load")),
            "region": raw.get("public_region") or raw.get("region"),
            "billing_note": raw.get("billing_note") or raw.get("payment_note"),
        })
    mode = body.get("checkout_mode")
    return items, mode if mode in ("stripe", "manual_activation") else None


def lane_limit(offer: dict) -> int:
    """The most lanes one purchase can add: the offer's maximum, capped by what is free; 0 when full."""
    free = offer["available_lanes"] if offer["available_lanes"] is not None else offer["max_lanes"]
    return 0 if free < offer["min_lanes"] else min(offer["max_lanes"], free)


def money(amount: float, currency: str) -> str:
    """A price as people read it: €0.50, €12, 3.75 DKK."""
    value = f"{amount / 100:.0f}" if amount % 100 == 0 else f"{amount / 100:.2f}"
    return f"{SYMBOLS[currency]}{value}" if currency in SYMBOLS else f"{value} {currency}"


def period_word(period: str) -> str:
    return "month" if period == "month" else "purchase" if period == "one_time" else "hour"


def billing_note(offer: dict, mode: str | None) -> str | None:
    """How the offer is paid, in the server's words when it sends them (core's billingNote)."""
    if offer["billing_note"]:
        return str(offer["billing_note"])
    if mode == "manual_activation":
        return "Our team activates new lanes after you confirm."
    if mode is None or offer["price"] is None:
        return None
    if period_word(offer["price"][2]) == "hour":
        return "Runs from your balance and stops when you stop it."
    return "You pay by card in your browser."


def offers_text(items: list[dict], mode: str | None, owned: dict[str, int]) -> str:
    """What `/adaptiq models` says: one line per model on offer, numbered for /adaptiq get."""
    if not items:
        return "No models are on offer right now. Check back soon."
    lines = ["Models you can get:"]
    for number, offer in enumerate(items, 1):
        parts = [offer["name"]]
        if offer["price"] and offer["price"][0] is not None:
            amount, currency, period = offer["price"]
            parts.append(f"{money(amount, currency)} per lane per {period_word(period)}")
        if offer["speed"]:
            parts.append(f"{round(offer['speed'])} tokens/s")
        if offer["region"]:
            parts.append(str(offer["region"]))
        lanes = owned.get(offer["name"].lower(), 0)
        if lanes:
            parts.append(f"in your workspace ({lanes} {'lane' if lanes == 1 else 'lanes'})")
        if lane_limit(offer) == 0:
            parts.append("full right now")
        line = f"{number}. " + " · ".join(parts)
        if offer["description"]:
            line += f" — {offer['description']}"
        lines.append(line)
    note = billing_note(items[0], mode) if len({billing_note(offer, mode) for offer in items}) == 1 else None
    if note:
        lines.append(note)
    lines.append("Get one: /adaptiq get <number> [lanes]. Each lane runs one conversation at a time.")
    return "\n".join(lines)


def failure(status: int, reply: dict, name: str) -> str:
    """One plain sentence for a purchase the gateway refused (core's purchaseFailure)."""
    detail = reply.get("detail") if isinstance(reply.get("detail"), dict) else {}
    code = detail.get("code")
    if code == "email_unconfirmed":
        return "Confirm your email address first (check your inbox), then try again."
    if code == "waiting_line":
        return f"{name} is full right now. Try again later, or pick another model."
    if code == "reservation_limit":
        return "You already have purchases waiting. Finish or cancel one first."
    if status == 429:
        return "Too many purchase requests just now. Try again in a few minutes."
    if status == 503:
        return "Purchases are paused for a moment. Try again shortly."
    if status == 404:
        return f"{name} is no longer offered."
    return str(detail.get("message") or f"AdaptiqAI answered {status}.")


def request_key(offer_id: str, lanes: int, key: str, minute: int | None = None) -> str:
    """One idempotency key per offer, lane count and connection within a minute: the same command
    typed twice asks for the same purchase, and the gateway answers with the same request."""
    minute = int(time.time() // 60) if minute is None else minute
    digest = hashlib.sha256(f"{offer_id}\n{lanes}\n{key[-8:]}\n{minute}".encode()).hexdigest()[:40]
    return f"hermes-purchase-{digest}"


def start(offer: dict, lanes: int, key: str, purchase_url: str | None) -> dict:
    """Ask for lanes on an offer. The answer's `url` is the page where the owner confirms.

    A gateway without purchase requests, or a connection that may not make them (an API key),
    gets the console's purchase page instead (`via: console`).
    """
    limit = lane_limit(offer)
    if limit == 0:
        raise PurchaseError(f"{offer['name']} has no free lanes right now. Try again later, or pick another model.")
    if lanes < offer["min_lanes"] or lanes > limit:
        raise PurchaseError(f"Choose between {offer['min_lanes']} and {limit} lanes.")
    try:
        status, reply = core._call("POST", "/cli-auth/purchase/requests", body={"offer_id": offer["id"], "lanes": lanes},
                                   key=key, headers={"idempotency-key": request_key(offer["id"], lanes, key)})
    except (OSError, ValueError) as error:
        raise PurchaseError("AdaptiqAI can't be reached right now. Try again in a moment.") from error
    detail = reply.get("detail")
    missing_route = status == 405 or (status == 404 and isinstance(detail, str)
                                      and detail.strip().lower() in ("not found", "route not found"))
    if status in (200, 201) and (reply.get("request_id") or reply.get("id")):
        return {"via": "request", "request_id": str(reply.get("request_id") or reply.get("id")),
                "url": reply.get("verification_uri_complete") or reply.get("approve_url")}
    if not (missing_route or status == 403):
        raise PurchaseError(failure(status, reply, offer["name"]))
    page = urllib.parse.urlsplit(purchase_url or f"{CONSOLE}?purchase={urllib.parse.quote(offer['id'])}#/models")
    query = urllib.parse.urlencode([*urllib.parse.parse_qsl(page.query), ("lanes", str(lanes))])
    return {"via": "console", "request_id": None, "url": urllib.parse.urlunsplit(page._replace(query=query))}


def request_status(request_id: str, key: str) -> str | None:
    """The purchase request's status as the gateway says it, or None when it is not known."""
    try:
        status, reply = core._call("GET", "/cli-auth/purchase/requests/" + urllib.parse.quote(request_id, safe=""),
                                   key=key)
    except (OSError, ValueError):
        return None
    if status == 404:
        return "expired"
    if status != 200:
        return None
    return str(reply.get("status") or "") or None


def model_for(offer: dict, summary: dict) -> tuple[str | None, str | None]:
    """The model an offer sells and the console page that sells it, from the account summary."""
    for row in summary.get("available_models") or []:
        if isinstance(row, dict) and row.get("offer_id") == offer["id"]:
            return row.get("id"), row.get("purchase_url")
    for row in summary.get("models") or []:
        if isinstance(row, dict) and str(row.get("display_name") or "").lower() == offer["name"].lower():
            return row.get("id"), None
    return None, None


def lanes_in(rows: list[dict], model_id: str | None, name: str) -> int:
    """The lanes the workspace has on a model, from `/v1/models` rows (by id, else by name)."""
    for row in rows:
        if (model_id and row.get("id") == model_id) or (
                not model_id and str(row.get("display_name") or "").lower() == name.lower()):
            return int((row.get("adaptiq") or {}).get("lanes") or 0)
    return 0


def follow(purchase: dict, key: str, lanes_on, on_ready, sleep=time.sleep, now=time.monotonic) -> str:
    """Wait for the lanes of a started purchase: they are the proof, whatever the request says.

    `lanes_on()` reads the lanes the workspace has on the offer's model now; more than `baseline`
    means they arrived, and `on_ready()` runs. Returns how the purchase ended.
    """
    started = now()
    while now() - started < FOLLOW_FOR:
        age = now() - started
        sleep(4 if age < 120 else 10 if age < 900 else 30)
        if purchase.get("request_id"):
            status = request_status(purchase["request_id"], key)
            if status in TERMINAL:
                return status
        try:
            arrived = lanes_on() > purchase["baseline"]
        except Exception:  # read again on the next round
            arrived = False
        if arrived:
            on_ready()
            return "ready"
    return "still waiting"
