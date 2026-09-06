"""
Alert System for EGX Signals
Telegram notifications for signal changes
"""
import os
import logging
import requests
from typing import Dict, List, Optional
from datetime import date
from src.store.signal_store import load_latest_signals, load_store

log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ─────────────────────────────────────────────────────────────
# Telegram Helpers
# ─────────────────────────────────────────────────────────────
def _send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.debug("Telegram not configured, skipping alert")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        else:
            log.warning("Telegram send failed: %s", resp.text)
            return False
    except Exception as e:
        log.warning("Telegram error: %s", e)
        return False

# ─────────────────────────────────────────────────────────────
# Alert Logic
# ─────────────────────────────────────────────────────────────
def check_signal_changes(current_signals: List[Dict]) -> List[str]:
    """Compare current signals with previous day and return change messages."""
    if not current_signals:
        return []

    today = date.today()
    prev_date = today - pd.Timedelta(days=1)

    # Load previous day's signals
    store = load_store()
    if store.empty:
        return ["📊 First run - no previous signals to compare"]

    prev_signals = store[store["run_date"] == prev_date]
    if prev_signals.empty:
        return [f"📅 No signals for {prev_date} (weekend/holiday?)"]

    prev_dict = {row["ticker"]: row.to_dict() for _, row in prev_signals.iterrows()}
    curr_dict = {s.get("Selected Stock", s.get("ticker", "?")): s for s in current_signals}

    messages = []

    # Check for new Buy signals
    for ticker, curr in curr_dict.items():
        prev = prev_dict.get(ticker)
        if not prev:
            continue

        curr_rec = curr.get("recommendation", "N/A")
        prev_rec = prev.get("recommendation", "N/A")

        if curr_rec != prev_rec:
            emoji = {"Buy": "🟢", "Watch": "🟡", "Avoid": "🔴"}.get(curr_rec, "⚪")
            msg = f"{emoji} <b>{ticker}</b>: {prev_rec} → {curr_rec}"
            if curr.get("entry_price"):
                msg += f"\n   Entry: {curr['entry_price']:.3f} | SL: {curr.get('stop_loss', 'N/A'):.3f} | TP1: {curr.get('tp1', 'N/A'):.3f}"
            messages.append(msg)

    # New tickers
    new_tickers = set(curr_dict.keys()) - set(prev_dict.keys())
    if new_tickers:
        messages.append(f"➕ <b>New tickers added:</b> {', '.join(sorted(new_tickers))}")

    # Removed tickers
    removed = set(prev_dict.keys()) - set(curr_dict.keys())
    if removed:
        messages.append(f"➖ <b>Tickers removed:</b> {', '.join(sorted(removed))}")

    return messages

def check_high_priority_signals(signals: List[Dict]) -> List[str]:
    """Identify high-priority signals for immediate attention."""
    messages = []

    for s in signals:
        if s.get("recommendation") != "Buy":
            continue

        score = s.get("score", 0)
        tp1_rr = s.get("tp1_rr", 0)
        ml_conf = s.get("ml_confidence", 0)

        # High conviction Buy
        if score >= 80 and tp1_rr >= 3.0 and ml_conf >= 0.7:
            msg = (f"🚀 <b>HIGH CONVICTION BUY: {s['ticker']}</b>\n"
                   f"   Score: {score:.1f}/100 | TP1 R/R: {tp1_rr:.1f}x | ML Conf: {ml_conf:.0%}\n"
                   f"   Entry: {s.get('entry_price', 'N/A'):.3f} | SL: {s.get('stop_loss', 'N/A'):.3f} | TP1: {s.get('tp1', 'N/A'):.3f}")
            messages.append(msg)

        # Oversold bounce candidates
        rsi = s.get("rsi", 50)
        if rsi <= 30 and s.get("is_near_support", False):
            msg = (f"🔨 <b>OVERSOLD BOUNCE: {s['ticker']}</b>\n"
                   f"   RSI: {rsi:.1f} | Near Support: {s.get('support', 'N/A'):.3f}\n"
                   f"   Score: {score:.1f} | Entry: {s.get('entry_price', 'N/A'):.3f}")
            messages.append(msg)

    return messages

def send_daily_alert(signals: List[Dict]) -> None:
    """Send daily summary alert."""
    if not signals:
        _send_telegram("📊 EGX Signals: No valid signals generated today")
        return

    # Summary stats
    total = len(signals)
    buys = sum(1 for s in signals if s.get("recommendation") == "Buy")
    watches = sum(1 for s in signals if s.get("recommendation") == "Watch")
    avoids = sum(1 for s in signals if s.get("recommendation") == "Avoid")

    # Top 3 Buys by score
    top_buys = sorted([s for s in signals if s.get("recommendation") == "Buy"],
                      key=lambda x: x.get("score", 0), reverse=True)[:3]

    msg = f"📊 <b>EGX Daily Signals - {date.today().strftime('%d %b %Y')}</b>\n\n"
    msg += f"📈 <b>Summary:</b> {total} analyzed | 🟢 {buys} Buy | 🟡 {watches} Watch | 🔴 {avoids} Avoid\n\n"

    if top_buys:
        msg += "🏆 <b>Top Buys:</b>\n"
        for i, s in enumerate(top_buys, 1):
            msg += (f"  {i}. {s['ticker']} — Score: {s.get('score', 0):.1f} | "
                    f"R/R: {s.get('tp1_rr', 0):.1f}x | "
                    f"Entry: {s.get('entry_price', 'N/A'):.3f}\n")

    changes = check_signal_changes(signals)
    if changes:
        msg += "\n🔄 <b>Changes vs Yesterday:</b>\n"
        for c in changes[:10]:  # Limit to 10
            msg += f"  • {c}\n"
        if len(changes) > 10:
            msg += f"  ... and {len(changes) - 10} more\n"

    high_priority = check_high_priority_signals(signals)
    if high_priority:
        msg += "\n⚡ <b>High Priority:</b>\n"
        for h in high_priority[:3]:
            msg += f"  {h}\n"

    _send_telegram(msg)

def send_error_alert(error: str, context: str = "") -> None:
    """Send error alert."""
    msg = f"❌ <b>EGX Signal Error</b>\n"
    if context:
        msg += f"Context: {context}\n"
    msg += f"Error: {error}"
    _send_telegram(msg)

# Need pandas for Timedelta
import pandas as pd