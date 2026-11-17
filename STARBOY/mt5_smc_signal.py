"""XAU/USD SMC signal engine for MetaTrader 5 (read-only; never sends orders).

Run: python mt5_smc_signal_full.py --watch
Set SYMBOL to your broker's exact gold symbol first.
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Literal

import MetaTrader5 as mt5

# ===== Settings =====
SYMBOL = "XAUUSD"              # e.g. XAUUSD.a or GOLD at some brokers
PIVOT_BARS = 3
OB_SEARCH_BARS = 12
MAX_ORDER_BLOCKS = 5
CONFIRMATION_BODY_FRACTION = 0.55
STOP_BUFFER = 0.30
MAX_RR = 4.0
POLL_SECONDS = 5
PIP_SIZE = 0.01
SL_PIPS = 10.0
TP_PIPS = 30.0
MAX_ENTRY_DRIFT_PIPS = 8.0

# Set True only after your MT5 broker confirms that Market Depth works for gold.
# A CFD/spot-gold DOM is usually broker-local, not a global market heat map.
REQUIRE_LIQUIDITY_HEATMAP = False
HEATMAP_SNAPSHOTS = 60
HEATMAP_STRENGTH_MULTIPLIER = 2.0
HEATMAP_ZONE_BUFFER = 0.50


@dataclass
class OrderBlock:
    direction: Literal["bullish", "bearish"]
    low: float
    high: float
    break_time: int


@dataclass
class Signal:
    status: Literal["BUY", "SELL", "WAIT", "BLOCKED"]
    reason: str
    symbol: str
    entry: float | None = None
    stop: float | None = None
    take_profit: float | None = None
    rr: float | None = None
    refined_order_block: tuple[float, float] | None = None
    bullish_order_blocks: list[OrderBlock] = field(default_factory=list)
    bearish_order_blocks: list[OrderBlock] = field(default_factory=list)
    liquidity_heatmap: dict = field(default_factory=dict)


def candle_body(bar: dict) -> tuple[float, float]:
    """Order-block zone = open to close only. Wicks are excluded."""
    return min(bar["open"], bar["close"]), max(bar["open"], bar["close"])


def closed_bars(timeframe: int, count: int) -> list[dict]:
    """Fetch completed MT5 candles only; ignore the still-forming last candle."""
    raw = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, count + 1)
    if raw is None or len(raw) < count:
        raise RuntimeError(f"Could not get {count} candles: {mt5.last_error()}")
    return [{name: row[name].item() for name in raw.dtype.names} for row in raw[:-1]]


def pivots(bars: list[dict], width: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs, lows = [], []
    for i in range(width, len(bars) - width):
        window = bars[i - width:i + width + 1]
        if bars[i]["high"] == max(bar["high"] for bar in window):
            highs.append((i, bars[i]["high"]))
        if bars[i]["low"] == min(bar["low"] for bar in window):
            lows.append((i, bars[i]["low"]))
    return highs, lows


def four_hour_bias(bars: list[dict]) -> Literal["bullish", "bearish", "neutral"]:
    highs, lows = pivots(bars, PIVOT_BARS)
    if len(highs) < 2 or len(lows) < 2:
        return "neutral"
    if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
        return "bullish"
    if highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
        return "bearish"
    return "neutral"


def active_order_blocks(bars: list[dict], direction: Literal["bullish", "bearish"]) -> list[OrderBlock]:
    """Find uninvalidated OB bodies after confirmed 1H BOS; newest first."""
    pivot_highs, pivot_lows = pivots(bars, PIVOT_BARS)
    reference = pivot_highs if direction == "bullish" else pivot_lows
    found: list[tuple[int, OrderBlock]] = []
    used_breaks: set[int] = set()

    for pivot_index, level in reference:
        for break_index in range(pivot_index + 1, len(bars)):
            previous_close = bars[break_index - 1]["close"]
            broke = (
                bars[break_index]["close"] > level and previous_close <= level
                if direction == "bullish"
                else bars[break_index]["close"] < level and previous_close >= level
            )
            if not broke:
                continue
            if break_index in used_breaks:
                break
            used_breaks.add(break_index)

            # Last opposing candle before BOS. Body only; wicks are never OB bounds.
            for j in range(break_index - 1, max(-1, break_index - OB_SEARCH_BARS), -1):
                opposing = bars[j]["close"] < bars[j]["open"] if direction == "bullish" else bars[j]["close"] > bars[j]["open"]
                if opposing:
                    low, high = candle_body(bars[j])
                    invalidated = any(
                        bar["close"] < low if direction == "bullish" else bar["close"] > high
                        for bar in bars[break_index + 1:]
                    )
                    if not invalidated:
                        found.append((break_index, OrderBlock(direction, low, high, bars[break_index]["time"])))
                    break
            break

    found.sort(key=lambda item: item[0], reverse=True)
    return [block for _, block in found[:MAX_ORDER_BLOCKS]]


def overlap(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float] | None:
    low, high = max(a[0], b[0]), min(a[1], b[1])
    return (low, high) if low < high else None


def refine_zone(bars: list[dict], zone: tuple[float, float], direction: Literal["bullish", "bearish"]) -> tuple[float, float] | None:
    """Use only the latest overlapping opposing-candle body to tighten the zone."""
    for bar in reversed(bars[-50:]):
        opposing = bar["close"] < bar["open"] if direction == "bullish" else bar["close"] > bar["open"]
        if opposing:
            refined = overlap(zone, candle_body(bar))
            if refined:
                return refined
    return None


def one_minute_mss(bars: list[dict], direction: Literal["bullish", "bearish"]) -> bool:
    """Closed 1M displacement candle breaking a recent confirmed pivot."""
    if len(bars) < 12:
        return False

    current = bars[-1]
    candle_range = current["high"] - current["low"]
    if candle_range <= 0:
        return False

    body_fraction = abs(current["close"] - current["open"]) / candle_range
    if body_fraction < CONFIRMATION_BODY_FRACTION:
        return False

    history = bars[:-1]
    pivot_highs, pivot_lows = pivots(history, PIVOT_BARS)

    recent_highs = [p for p in pivot_highs if p[0] >= max(0, len(history) - 20)]
    recent_lows = [p for p in pivot_lows if p[0] >= max(0, len(history) - 20)]

    if direction == "bullish":
        return (
            current["close"] > current["open"]
            and bool(recent_highs)
            and current["close"] > recent_highs[-1][1]
        )

    return (
        current["close"] < current["open"]
        and bool(recent_lows)
        and current["close"] < recent_lows[-1][1]
    )


class LiquidityHeatmap:
    """Keeps broker DOM snapshots to find persistent bid/ask liquidity clusters."""
    def __init__(self) -> None:
        self.snapshots: deque[dict[str, dict[float, float]]] = deque(maxlen=HEATMAP_SNAPSHOTS)

    def update(self, book: tuple | None, point: float) -> None:
        if not book:
            return
        bids: dict[float, float] = {}
        asks: dict[float, float] = {}
        buy_type = getattr(mt5, "BOOK_TYPE_BUY", 2)
        sell_type = getattr(mt5, "BOOK_TYPE_SELL", 1)
        for item in book:
            level = item._asdict()
            price = round(level["price"] / point) * point
            volume = float(level.get("volume_dbl", level.get("volume_real", level["volume"])))
            if level["type"] == buy_type:
                bids[price] = bids.get(price, 0.0) + volume
            elif level["type"] == sell_type:
                asks[price] = asks.get(price, 0.0) + volume
        if bids or asks:
            self.snapshots.append({"bids": bids, "asks": asks})

    def summary(self) -> dict:
        if not self.snapshots:
            return {"available": False, "snapshots": 0}
        return {
            "available": True,
            "snapshots": len(self.snapshots),
            "latest_bid_levels": len(self.snapshots[-1]["bids"]),
            "latest_ask_levels": len(self.snapshots[-1]["asks"]),
        }

    def confirms(self, zone: tuple[float, float], direction: Literal["bullish", "bearish"]) -> bool | None:
        """Buy requires a strong bid cluster; sell requires a strong ask cluster."""
        if not self.snapshots:
            return None
        side = "bids" if direction == "bullish" else "asks"
        all_volumes = [volume for snap in self.snapshots for volume in snap[side].values()]
        if not all_volumes:
            return None
        low, high = zone[0] - HEATMAP_ZONE_BUFFER, zone[1] + HEATMAP_ZONE_BUFFER
        nearby = [volume for snap in self.snapshots for price, volume in snap[side].items() if low <= price <= high]
        return bool(nearby) and max(nearby) >= mean(all_volumes) * HEATMAP_STRENGTH_MULTIPLIER


def market_book_snapshot() -> tuple | None:
    """Read the broker's currently available DOM. Returns None when unsupported."""
    if not mt5.market_book_add(SYMBOL):
        return None
    try:
        time.sleep(0.2)
        return mt5.market_book_get(SYMBOL)
    finally:
        mt5.market_book_release(SYMBOL)


def evaluate(heatmap: LiquidityHeatmap) -> Signal:
    if not mt5.initialize():
        return Signal("BLOCKED", f"MT5 connection failed: {mt5.last_error()}", SYMBOL)
    try:
        info = mt5.symbol_info(SYMBOL)
        if info is None:
            return Signal("BLOCKED", "Symbol not found. Set SYMBOL to your broker's gold name.", SYMBOL)
        if not info.visible and not mt5.symbol_select(SYMBOL, True):
            return Signal("BLOCKED", "Could not select symbol in Market Watch.", SYMBOL)

        heatmap.update(market_book_snapshot(), info.point)
        h1 = closed_bars(mt5.TIMEFRAME_H1, 250)
        bullish_blocks = active_order_blocks(h1, "bullish")
        bearish_blocks = active_order_blocks(h1, "bearish")
        heatmap_data = heatmap.summary()

        def result(status: Literal["BUY", "SELL", "WAIT", "BLOCKED"], reason: str, **kwargs) -> Signal:
            return Signal(status, reason, SYMBOL, bullish_order_blocks=bullish_blocks, bearish_order_blocks=bearish_blocks, liquidity_heatmap=heatmap_data, **kwargs)

        bias = four_hour_bias(closed_bars(mt5.TIMEFRAME_H4, 250))
        if bias == "neutral":
            return result("WAIT", "4H market structure is neutral; no directional bias.")

        blocks = bullish_blocks if bias == "bullish" else bearish_blocks
        if not blocks:
            return result("WAIT", f"4H is {bias}, but no active 1H {bias} body-only OB exists.")
        one_hour_zone = (blocks[0].low, blocks[0].high)
        m15_zone = refine_zone(closed_bars(mt5.TIMEFRAME_M15, 120), one_hour_zone, bias)
        m3_zone = refine_zone(closed_bars(mt5.TIMEFRAME_M3, 180), m15_zone, bias) if m15_zone else None
        if m3_zone is None:
            return result("WAIT", "No 15M/3M body-only refinement overlaps the 1H OB.", refined_order_block=one_hour_zone)

        sweep = closed_bars(mt5.TIMEFRAME_M5, 20)[-1]
        low, high = m3_zone
        swept = sweep["low"] < low and sweep["close"] > low if bias == "bullish" else sweep["high"] > high and sweep["close"] < high
        if not swept:
            return result("WAIT", "Waiting for 5M wick sweep beyond the refined body zone and close back inside.", refined_order_block=m3_zone)

        liquidity_ok = heatmap.confirms(m3_zone, bias)
        if REQUIRE_LIQUIDITY_HEATMAP and liquidity_ok is None:
            return result("WAIT", "5M sweep found, but MT5 has no usable DOM/heat-map data.", refined_order_block=m3_zone)
        if REQUIRE_LIQUIDITY_HEATMAP and not liquidity_ok:
            return result("WAIT", "5M sweep found, but no strong matching liquidity cluster near the zone.", refined_order_block=m3_zone)

        m1 = closed_bars(mt5.TIMEFRAME_M1, 120)
        if not one_minute_mss(m1, bias):
            return result("WAIT", "5M sweep complete; waiting for closed 1M confirmation candle + CHoCH/MSS.", refined_order_block=m3_zone)

        entry = m1[-1]["close"]
        if bias == "bullish":
            stop, risk, status = sweep["low"] - STOP_BUFFER, entry - (sweep["low"] - STOP_BUFFER), "BUY"
            target = entry + risk * MAX_RR
        else:
            stop, risk, status = sweep["high"] + STOP_BUFFER, (sweep["high"] + STOP_BUFFER) - entry, "SELL"
            target = entry - risk * MAX_RR
        if risk <= info.point:
            return result("WAIT", "Invalid stop distance; no signal.", refined_order_block=m3_zone)
        return result(status, f"{bias} setup confirmed: 5M sweep + closed 1M CHoCH/MSS.", entry=entry, stop=stop, take_profit=target, rr=MAX_RR, refined_order_block=m3_zone)
    finally:
        mt5.shutdown()


def print_result(signal: Signal) -> None:
    print(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(), **asdict(signal)}, indent=2), flush=True)




def watch() -> None:
    heatmap = LiquidityHeatmap()
    last_hour = None

    print("=" * 72)
    print("XAUUSD SMC FIRST STRATEGY - LIVE READ-ONLY ENGINE")
    print(f"Symbol: {SYMBOL} | Refresh: {POLL_SECONDS}s | Spread limit: {MAX_SPREAD_PIPS} pips")
    print(f"SL: {SL_PIPS:.0f} pips | TP: {TP_PIPS:.0f} pips | RR: 1:3")
    print("Strategy: 4H Bias -> 1H OB -> 15M -> 3M -> 5M Sweep -> 1M MSS")
    print("No orders are placed.")
    print("=" * 72, flush=True)

    while True:
        try:
            signal = evaluate(heatmap)
            now = datetime.now(timezone.utc)
            bucket = now.strftime("%Y-%m-%d %H")

            entry = f"{signal.entry:.2f}" if signal.entry is not None else "-"
            print(
                f"[{now.strftime('%H:%M:%S')} UTC] "
                f"{signal.status:<8} {entry:<10} {signal.reason}",
                flush=True,
            )

            if bucket != last_hour:
                print()
                print("#" * 72)
                print(f"HOURLY SMC DECISION - {now.strftime('%Y-%m-%d %H:00 UTC')}")
                print(f"STATUS : {signal.status}")
                print(f"REASON : {signal.reason}")
                if signal.entry is not None:
                    print(f"ENTRY  : {signal.entry:.2f}")
                    print(f"SL     : {signal.stop:.2f}")
                    print(f"TP     : {signal.take_profit:.2f}")
                    print("RR     : 1:3")
                print("#" * 72, flush=True)
                last_hour = bucket

        except Exception as exc:
            print(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] "
                f"ENGINE ERROR: {type(exc).__name__}: {exc}",
                flush=True,
            )
            print("Retrying in 5 seconds...", flush=True)

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    if "--watch" in sys.argv:
        watch()
    else:
        print_result(evaluate(LiquidityHeatmap()))