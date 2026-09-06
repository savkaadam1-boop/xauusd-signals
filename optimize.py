#!/usr/bin/env python3
"""
Optimalizator s poctivym delenim dat.

Vyskusa 288 kombinacii parametrov na kazdom trhu. Prvych 60 % obdobia
sluzi na LADENIE, poslednych 40 % na OVERENIE. Vypise najlepsie
kombinacie z ladenia a k nim ICH VYSLEDOK NA OVERENI.

Ak kombinacia zaria na ladeni a prepadne na overeni, bola to nahoda.
To je jediny sposob, ako to rozlisit.
"""
import itertools, os, sys
import numpy as np, pandas as pd, requests, yfinance as yf

TICKERS  = {"NQ=F": "Nasdaq", "ES=F": "S&P 500", "GC=F": "Zlato",
            "YM=F": "Dow", "CL=F": "Ropa"}
INTERVAL = os.environ.get("INTERVAL", "1h")
PERIOD   = os.environ.get("PERIOD", "730d")
TRAIN_FRAC = 0.60

PIV, SR_TOL, SR_MIN_TOUCH = 6, 0.40, 3
CONF_MAX_ATR, SWING_WINDOW = 1.00, 300
SL_MIN_ATR, MAX_HOLD = 0.50, 120
SPREAD_R = 0.05          # spread ako podiel R (odhad)

GRID = {
    "rr":     [1.5, 2.0, 2.5, 3.0],
    "slbars": [10, 15, 20],
    "slbuf":  [0.25, 0.60, 1.00],
    "trend":  [False, True],
    "kz":     [False, True],
    "chop":   [False, True],
}

TG_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT  = os.environ.get("TELEGRAM_CHAT_ID", "")
SEND_TG  = os.environ.get("SEND_TG", "").lower() in ("1", "true", "yes")


def fetch(t):
    df = yf.download(t, interval=INTERVAL, period=PERIOD, progress=False, auto_adjust=False)
    if df is None or len(df) < 2000:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().copy()
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df


def prepare(df):
    """Vsetko, co nezavisi od ladenych parametrov, spocitame raz."""
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    o = df["Open"].values.astype(float)

    ef = pd.Series(c).ewm(span=9,  adjust=False).mean().values
    es = pd.Series(c).ewm(span=21, adjust=False).mean().values
    e200 = pd.Series(c).ewm(span=200, adjust=False).mean().values

    prev = np.roll(c, 1); prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = pd.Series(tr).rolling(14).mean().values

    up = (ef[:-1] <= es[:-1]) & (ef[1:] > es[1:])
    dn = (ef[:-1] >= es[:-1]) & (ef[1:] < es[1:])
    cross_up = np.concatenate([[False], up])
    cross_dn = np.concatenate([[False], dn])

    hours = df.index.hour.values
    kz = ((hours >= 7) & (hours < 10)) | ((hours >= 12) & (hours < 15))

    slope = np.abs(pd.Series(es).diff(10).values)

    # potvrdene swingy
    n = len(c)
    sw_hi = np.zeros(n, dtype=bool)
    sw_lo = np.zeros(n, dtype=bool)
    for i in range(PIV, n - PIV):
        wh, wl = h[i-PIV:i+PIV+1], l[i-PIV:i+PIV+1]
        if h[i] == wh.max() and (wh == h[i]).sum() == 1: sw_hi[i] = True
        if l[i] == wl.min() and (wl == l[i]).sum() == 1: sw_lo[i] = True

    return dict(c=c, h=h, l=l, o=o, ef=ef, es=es, e200=e200, atr=atr,
                cu=cross_up, cd=cross_dn, kz=kz, slope=slope,
                sw_hi=sw_hi, sw_lo=sw_lo, n=n)


def zone_ok(P, i, is_long):
    """Je cross do CONF_MAX_ATR od urovne s aspon SR_MIN_TOUCH dotykmi?"""
    a = P["atr"][i]
    if not np.isfinite(a) or a <= 0: return False
    tol, maxd = a * SR_TOL, a * CONF_MAX_ATR
    s = max(PIV, i - SWING_WINDOW)
    e = i - PIV
    if e <= s: return False
    src = P["sw_lo"] if is_long else P["sw_hi"]
    arr = (P["l"] if is_long else P["h"])[s:e][src[s:e]]
    if arr.size == 0: return False
    close = P["c"][i]
    near = arr[np.abs(arr - close) <= maxd]
    if near.size == 0: return False
    if is_long:  near = near[near <= close + tol]
    else:        near = near[near >= close - tol]
    if near.size == 0: return False
    for lv in near:
        if (np.abs(arr - lv) <= tol).sum() >= SR_MIN_TOUCH:
            return True
    return False


def run(P, cfg, i0, i1):
    """Simulacia na useku [i0, i1). Vracia (pocet, PF, expectancy v R)."""
    res = []
    i = max(i0, SWING_WINDOW + 10)
    while i < min(i1, P["n"] - 1):
        long_, short_ = P["cu"][i], P["cd"][i]
        if not long_ and not short_:
            i += 1; continue
        a = P["atr"][i]
        if not np.isfinite(a) or a <= 0:
            i += 1; continue
        if cfg["trend"] and (P["c"][i] > P["e200"][i]) != long_:
            i += 1; continue
        if cfg["kz"] and not P["kz"][i]:
            i += 1; continue
        if cfg["chop"] and P["slope"][i] < a * 0.30:
            i += 1; continue
        if not zone_ok(P, i, long_):
            i += 1; continue

        w0 = max(0, i - cfg["slbars"] + 1)
        if long_:
            sl = min(P["l"][w0:i+1].min() - a * cfg["slbuf"], P["c"][i] - a * SL_MIN_ATR)
        else:
            sl = max(P["h"][w0:i+1].max() + a * cfg["slbuf"], P["c"][i] + a * SL_MIN_ATR)
        risk = abs(P["c"][i] - sl)
        if risk <= 0:
            i += 1; continue
        tp = P["c"][i] + cfg["rr"] * risk if long_ else P["c"][i] - cfg["rr"] * risk

        j1 = min(P["n"], i + 1 + MAX_HOLD)
        hh, ll = P["h"][i+1:j1], P["l"][i+1:j1]
        if hh.size == 0:
            break
        if long_:
            hit_sl = np.argmax(ll <= sl) if (ll <= sl).any() else 10**9
            hit_tp = np.argmax(hh >= tp) if (hh >= tp).any() else 10**9
        else:
            hit_sl = np.argmax(hh >= sl) if (hh >= sl).any() else 10**9
            hit_tp = np.argmax(ll <= tp) if (ll <= tp).any() else 10**9
        if hit_sl == 10**9 and hit_tp == 10**9:
            ex = P["c"][j1-1]
            r = ((ex - P["c"][i]) if long_ else (P["c"][i] - ex)) / risk
        else:
            r = -1.0 if hit_sl <= hit_tp else cfg["rr"]
        res.append(r - SPREAD_R)
        i += 12                      # cooldown
    if not res:
        return 0, 0.0, 0.0
    arr = np.array(res)
    gw, gl = arr[arr > 0].sum(), -arr[arr <= 0].sum()
    pf = gw / gl if gl > 0 else float("inf")
    return len(arr), pf, arr.mean()


def main():
    out = []
    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    out.append(f"OPTIMALIZACIA  {len(combos)} kombinacii x {len(TICKERS)} trhov")
    out.append(f"delenie: prvych {int(TRAIN_FRAC*100)} % ladenie / zvysok overenie")
    out.append("=" * 62)

    for t, name in TICKERS.items():
        df = fetch(t)
        if df is None:
            out.append(f"\n{name} ({t}): data sa nestiahli"); continue
        P = prepare(df)
        split = int(P["n"] * TRAIN_FRAC)
        rows = []
        for cfg in combos:
            n1, pf1, e1 = run(P, cfg, 0, split)
            if n1 < 15:                       # prilis mala vzorka na ladenie
                continue
            n2, pf2, e2 = run(P, cfg, split, P["n"])
            rows.append((pf1, n1, e1, pf2, n2, e2, cfg))
        rows.sort(key=lambda x: -x[0])

        out.append(f"\n{name}  ({t})   {P['n']} sviecok, "
                   f"{df.index[0]:%d.%m.} - {df.index[-1]:%d.%m.%Y}")
        out.append(f"{'RR':>4}{'SLb':>5}{'buf':>6}{'T':>2}{'K':>2}{'C':>2}"
                   f"{'| n':>6}{'PF':>7}{'expR':>7}{'|| n':>6}{'PF':>7}{'expR':>7}")
        out.append("-" * 62)
        for pf1, n1, e1, pf2, n2, e2, cfg in rows[:8]:
            out.append(f"{cfg['rr']:>4.1f}{cfg['slbars']:>5}{cfg['slbuf']:>6.2f}"
                       f"{'X' if cfg['trend'] else '-':>2}{'X' if cfg['kz'] else '-':>2}"
                       f"{'X' if cfg['chop'] else '-':>2}"
                       f"{n1:>6}{pf1:>7.2f}{e1:>+7.2f}{n2:>6}{pf2:>7.2f}{e2:>+7.2f}")
        if rows:
            top = rows[:8]
            prof1 = sum(1 for r in rows if r[0] > 1.10 and r[1] >= 30)
            survived = sum(1 for r in top if r[0] > 1.10 and r[3] > 1.10 and r[4] >= 20)
            out.append(f"kombinacii ziskovych uz na LADENI (PF>1.10, n>=30): "
                       f"{prof1}/{len(rows)}")
            out.append(f"z 8 najlepsich obstalo aj na OVERENI: {survived}/8")

    out.append("")
    out.append("STLPCE:  vlavo od || = LADENIE,  vpravo = OVERENIE")
    out.append("T/K/C = trend / killzone / chop filter")
    out.append("Zaujima nas VYHRADNE prava strana. Lava len ukazuje,")
    out.append("ako dobre sa da naladit na minulost.")
    txt = "\n".join(out)
    print("\n" + txt + "\n")
    if SEND_TG and TG_TOKEN and TG_CHAT:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      json={"chat_id": TG_CHAT, "text": txt[:4000]}, timeout=20)


if __name__ == "__main__":
    main()
