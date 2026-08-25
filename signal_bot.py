#!/usr/bin/env python3
"""
XAUUSD signal bot  -  EMA 9/21 cross potvrdeny zonou.

Rovnaka logika ako Pine indikator:
  KROK 1  9 EMA krizi 21 EMA na UZAVRETEJ sviecke
  KROK 2  cross musi byt do X ATR od platnej urovne
          (support / resistance / flip zo swingov, PDH/PDL, D open,
           azijska session H/L, okruhle cisla)
  KROK 3  uroven musi byt na spravnej strane

Rozhoduje sa vyhradne z uzavretych sviecok. Posledna, este beziaca
sviecka sa ignoruje - inak by chodili signaly, ktore o par minut zmiznu.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------------------------------------- nastavenia
TICKERS       = ["GC=F", "XAUUSD=X"]   # skusa sa v tomto poradi
INTERVAL      = "5m"
PERIOD        = "5d"

FAST_EMA      = 9
SLOW_EMA      = 21
ATR_PERIOD    = 14

PIV_LEFT      = 6
PIV_RIGHT     = 6
SR_TOL_ATR    = 0.40
SR_MIN_TOUCH  = 3
FLIP_MIN_EACH = 2

CONF_MAX_ATR  = 1.00    # ako blizko musi byt cross k urovni
REQUIRE_SIDE  = True    # BUY nad supportom, SELL pod resistance

USE_PDHL      = True
USE_DOPEN     = True
USE_ASIA      = True
ASIA_START_H  = 0       # UTC
ASIA_END_H    = 8
USE_ROUND     = True
ROUND_STEP    = 10.0

# --- SL / TP -----------------------------------------------------------
SL_SWING_BARS = 10      # SL za extrem poslednych N sviecok
SL_BUFFER_ATR = 0.25    # buffer za extrem
SL_MIN_ATR    = 0.50    # SL nikdy blizsie nez tolkoto ATR
MIN_RR        = 1.5     # pod tymto sa obchod NEZADA
MAX_RR        = 4.0     # nad tymto je TP nerealne daleko -> orezeme
RISK_USD      = 125.0   # kolko dolarov riskujeme na obchod
CONTRACT_SIZE = 100.0   # XAUUSD: 1 lot = 100 uncii -> 1 USD pohyb = 100 USD
SKIP_LOW_RR   = True    # ak ziadna uroven neda MIN_RR, signal sa zahodi

STATE_FILE    = "state.json"

TG_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT  = os.environ.get("TELEGRAM_CHAT_ID", "")
DRY_RUN  = os.environ.get("DRY_RUN", "") == "1"
TEST_MSG = os.environ.get("TEST_MSG", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------- data
def fetch():
    for t in TICKERS:
        try:
            df = yf.download(t, interval=INTERVAL, period=PERIOD,
                             progress=False, auto_adjust=False)
        except Exception as e:
            print(f"[warn] {t}: {e}")
            continue
        if df is None or len(df) < 300:
            print(f"[warn] {t}: malo dat ({0 if df is None else len(df)})")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna().copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        print(f"[ok] {t}: {len(df)} sviecok, posledna {df.index[-1]}")
        return t, df
    return None, None


def add_indicators(df):
    df["ema_f"] = df["Close"].ewm(span=FAST_EMA, adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=SLOW_EMA, adjust=False).mean()

    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df


def last_closed_index(df):
    """Posledna sviecka, ktora uz naozaj skoncila."""
    now = datetime.now(timezone.utc)
    bar_len = timedelta(minutes=5)
    i = len(df) - 1
    while i >= 0 and df.index[i] + bar_len > now:
        i -= 1
    return i


# -------------------------------------------------------------- urovne
def swings(df, upto):
    """Swingy potvrdene PIV_RIGHT svieckami. Nic, co by v case signalu
    este nebolo zname."""
    highs, lows = [], []
    hi = df["High"].values
    lo = df["Low"].values
    for i in range(PIV_LEFT, upto - PIV_RIGHT + 1):
        if i + PIV_RIGHT > upto:
            break
        w_hi = hi[i - PIV_LEFT: i + PIV_RIGHT + 1]
        w_lo = lo[i - PIV_LEFT: i + PIV_RIGHT + 1]
        if hi[i] == w_hi.max() and (w_hi == hi[i]).sum() == 1:
            highs.append((float(hi[i]), i))
        if lo[i] == w_lo.min() and (w_lo == lo[i]).sum() == 1:
            lows.append((float(lo[i]), i))
    return highs[-40:], lows[-40:]


def cluster_zones(highs, lows, tol):
    """Zhluky swingov -> zony. Vracia (cena, typ, pocet dotykov)."""
    pts = [(p, i, True) for p, i in highs] + [(p, i, False) for p, i in lows]
    pts.sort(key=lambda x: x[0])
    used = [False] * len(pts)
    zones = []

    for a in range(len(pts)):
        if used[a]:
            continue
        base = pts[a][0]
        grp = []
        for b in range(len(pts)):
            if not used[b] and abs(pts[b][0] - base) <= tol:
                grp.append(pts[b])
                used[b] = True
        if len(grp) < SR_MIN_TOUCH:
            continue
        n_hi = sum(1 for g in grp if g[2])
        n_lo = len(grp) - n_hi
        grp_sorted = sorted(grp, key=lambda x: x[1])
        first_is_high = grp_sorted[0][2]
        last_is_high = grp_sorted[-1][2]
        is_flip = (n_hi >= FLIP_MIN_EACH and n_lo >= FLIP_MIN_EACH
                   and first_is_high != last_is_high)
        avg = sum(g[0] for g in grp) / len(grp)
        kind = "flip" if is_flip else ("resistance" if n_hi >= n_lo else "support")
        zones.append((avg, kind, len(grp)))
    return zones


def extra_levels(df, upto):
    """PDH/PDL, dnesny open, azijska session, okruhle cisla."""
    out = []
    sub = df.iloc[: upto + 1]
    days = sub.groupby(sub.index.date)

    keys = list(days.groups.keys())
    if USE_PDHL and len(keys) >= 2:
        prev = days.get_group(keys[-2])
        out.append((float(prev["High"].max()), "PDH"))
        out.append((float(prev["Low"].min()), "PDL"))
    if USE_DOPEN and len(keys) >= 1:
        today = days.get_group(keys[-1])
        out.append((float(today["Open"].iloc[0]), "D open"))

    if USE_ASIA and len(keys) >= 1:
        today = days.get_group(keys[-1])
        mask = (today.index.hour >= ASIA_START_H) & (today.index.hour < ASIA_END_H)
        asia = today[mask]
        if len(asia) > 0:
            out.append((float(asia["High"].max()), "Asia H"))
            out.append((float(asia["Low"].min()), "Asia L"))

    if USE_ROUND and ROUND_STEP > 0:
        c = float(sub["Close"].iloc[-1])
        base = round(c / ROUND_STEP) * ROUND_STEP
        for lv in (base - ROUND_STEP, base, base + ROUND_STEP):
            out.append((float(lv), "round"))
    return out


# ------------------------------------------------------------- logika
def find_signal(df):
    i = last_closed_index(df)
    if i < PIV_LEFT + PIV_RIGHT + SLOW_EMA + 5:
        return None

    f_now, f_prev = df["ema_f"].iloc[i], df["ema_f"].iloc[i - 1]
    s_now, s_prev = df["ema_s"].iloc[i], df["ema_s"].iloc[i - 1]

    cross_up = f_prev <= s_prev and f_now > s_now
    cross_dn = f_prev >= s_prev and f_now < s_now
    if not cross_up and not cross_dn:
        return None

    close = float(df["Close"].iloc[i])
    atr = float(df["atr"].iloc[i])
    if not np.isfinite(atr) or atr <= 0:
        return None

    tol = atr * SR_TOL_ATR
    max_d = atr * CONF_MAX_ATR
    is_long = bool(cross_up)

    highs, lows = swings(df, i)
    cands = [(p, k) for p, k, _ in cluster_zones(highs, lows, tol)]
    cands += extra_levels(df, i)

    best = None
    for price, kind in cands:
        d = abs(close - price)
        if d > max_d:
            continue
        if REQUIRE_SIDE:
            if is_long and price > close + tol:
                continue
            if not is_long and price < close - tol:
                continue
        if best is None or d < best[0]:
            best = (d, price, kind)

    if best is None:
        print(f"[info] cross {'BUY' if is_long else 'SELL'} bez zony - preskocene")
        return None

    # ---------------- SL / TP -------------------------------------
    lo_win = float(df["Low"].iloc[max(0, i - SL_SWING_BARS + 1): i + 1].min())
    hi_win = float(df["High"].iloc[max(0, i - SL_SWING_BARS + 1): i + 1].max())
    buf = atr * SL_BUFFER_ATR

    if is_long:
        sl = lo_win - buf
        if close - sl < atr * SL_MIN_ATR:
            sl = close - atr * SL_MIN_ATR
    else:
        sl = hi_win + buf
        if sl - close < atr * SL_MIN_ATR:
            sl = close + atr * SL_MIN_ATR

    risk_dist = abs(close - sl)
    if risk_dist <= 0:
        return None

    # ---------------- TP na skutocnu uroven, nie na fixny nasobok ----
    # Kandidati: vsetky zony a urovne v smere obchodu + surove swingy.
    tp_cands = list(cands)
    tp_cands += [(p, "swing high") for p, _ in highs]
    tp_cands += [(p, "swing low") for p, _ in lows]

    eps = atr * 0.10
    ahead = []
    for price, kind in tp_cands:
        if is_long and price > close + eps:
            ahead.append((price, kind))
        elif not is_long and price < close - eps:
            ahead.append((price, kind))

    ahead.sort(key=lambda x: abs(x[0] - close))

    tp = None
    tp_kind = ""
    tp_rr = 0.0
    for price, kind in ahead:
        rr = abs(price - close) / risk_dist
        if rr >= MIN_RR:
            if rr > MAX_RR:
                break                     # najblizsia vhodna je uz prilis daleko
            tp, tp_kind, tp_rr = price, kind, rr
            break

    capped = False
    if tp is None:
        if SKIP_LOW_RR and not ahead:
            print("[info] ziadna uroven pred cenou - preskocene")
            return None
        # ziadna uroven nesedi -> orezany fixny TP, ak to dovolime
        if SKIP_LOW_RR:
            near = abs(ahead[0][0] - close) / risk_dist if ahead else 0.0
            print(f"[info] najblizsia uroven da len {near:.2f}R "
                  f"(min {MIN_RR}) - obchod sa nezadava")
            return None
        tp = close + MAX_RR * risk_dist if is_long else close - MAX_RR * risk_dist
        tp_kind, tp_rr, capped = "fixny", MAX_RR, True

    lots = RISK_USD / (risk_dist * CONTRACT_SIZE)
    lots = max(0.01, round(lots, 2))
    reward_usd = RISK_USD * tp_rr

    return {
        "bar": df.index[i].isoformat(),
        "side": "BUY" if is_long else "SELL",
        "price": close,
        "zone": best[2],
        "zone_price": best[1],
        "atr": atr,
        "sl": sl,
        "tp": tp,
        "tp_kind": tp_kind,
        "rr": tp_rr,
        "risk_dist": risk_dist,
        "lots": lots,
        "reward": reward_usd,
        "capped": capped,
    }


# ------------------------------------------------------------ telegram
def send(text):
    print("SPRAVA:\n" + text)
    if DRY_RUN:
        return True
    if not TG_TOKEN or not TG_CHAT:
        print("[chyba] chyba TELEGRAM_TOKEN alebo TELEGRAM_CHAT_ID")
        return False
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT, "text": text},
        timeout=20,
    )
    if r.status_code != 200:
        print(f"[chyba] telegram {r.status_code}: {r.text}")
        return False
    return True


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)


# ---------------------------------------------------------------- main
def main():
    if TEST_MSG:
        ok = send(
            "TEST\n"
            "------------------------\n"
            "Spojenie GitHub -> Telegram funguje.\n"
            "Toto nie je obchodny signal.\n"
            "Od teraz ti pridu len skutocne signaly:\n"
            "EMA 9/21 cross pri zone, RR aspon 1:1.5."
        )
        print("test odoslany" if ok else "test ZLYHAL")
        return 0 if ok else 1

    ticker, df = fetch()
    if df is None:
        print("[chyba] ziadne data")
        return 0

    df = add_indicators(df)
    sig = find_signal(df)
    if sig is None:
        print("[info] ziadny signal")
        return 0

    st = load_state()
    if st.get("last_bar") == sig["bar"]:
        print("[info] tento signal uz bol odoslany")
        return 0

    when = datetime.fromisoformat(sig["bar"]).strftime("%d.%m. %H:%M UTC")
    note = ""
    if sig["capped"]:
        note = "\nPOZOR: TP nie je na urovni, je orezany na maximum."

    text = (
        f"{sig['side']}  XAUUSD  M5\n"
        f"------------------------\n"
        f"vstup: {sig['price']:.2f}\n"
        f"SL:    {sig['sl']:.2f}   ({sig['risk_dist']:.2f} USD)\n"
        f"TP:    {sig['tp']:.2f}   ({sig['tp_kind']})\n"
        f"RR:    1:{sig['rr']:.2f}\n"
        f"lot:   {sig['lots']:.2f}\n"
        f"riziko {RISK_USD:.0f} USD  /  zisk {sig['reward']:.0f} USD\n"
        f"------------------------\n"
        f"zona vstupu: {sig['zone']} @ {sig['zone_price']:.2f}\n"
        f"ATR: {sig['atr']:.2f}\n"
        f"sviecka: {when}  (zdroj {ticker})"
        f"{note}"
    )

    if send(text):
        st["last_bar"] = sig["bar"]
        st["last_side"] = sig["side"]
        save_state(st)
    return 0


if __name__ == "__main__":
    sys.exit(main())
