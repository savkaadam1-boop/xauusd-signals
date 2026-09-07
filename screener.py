#!/usr/bin/env python3
"""
SCREENER STRATEGII

Netestuje jednu strategiu s 288 nastaveniami. Testuje 5 roznych
strategii (+ nahodny vstup ako kontrolu) s malou mriezkou 9 nastaveni.

Obchody sa ZLUCUJU cez vsetkych 5 trhov -> namiesto 25 obchodov na trh
mame stovky. Az to je vzorka, z ktorej sa da nieco tvrdit.

Prva polovica obdobia = LADENIE, druha = OVERENIE.
"NAHODA" je kontrolna skupina. Strategia, ktora ju neprekona, nema hranu.
"""
import os, math
import numpy as np, pandas as pd, yfinance as yf

TICKERS  = {"NQ=F": "Nasdaq", "ES=F": "S&P 500", "GC=F": "Zlato",
            "YM=F": "Dow", "CL=F": "Ropa"}
INTERVAL = os.environ.get("INTERVAL", "1h")
PERIOD   = os.environ.get("PERIOD", "730d")
TRAIN_FRAC = 0.60
MAX_HOLD   = 48
SPREAD_R   = 0.05
RRS = [1.0, 1.5, 2.0, 3.0]
SLS = [1.0, 1.5, 2.0]


def fetch(t):
    try:
        df = yf.download(t, interval=INTERVAL, period=PERIOD,
                         progress=False, auto_adjust=False)
    except Exception as e:
        print(f"  CHYBA pri stahovani {t}: {type(e).__name__}: {e}")
        return None
    if df is None or len(df) == 0:
        print(f"  {t}: yfinance vratil prazdne data")
        return None
    if len(df) < 1000:
        print(f"  {t}: len {len(df)} sviecok, to je malo (treba 1000+)")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().copy()
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df


def prepare(df):
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    n = len(c)
    s = pd.Series(c)

    prev = np.roll(c, 1); prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = pd.Series(tr).rolling(14).mean().values

    P = dict(c=c, h=h, l=l, n=n, atr=atr)
    for sp in (9, 20, 21, 50, 200):
        P[f"e{sp}"] = s.ewm(span=sp, adjust=False).mean().values
    P["sma20"] = s.rolling(20).mean().values
    P["sd20"]  = s.rolling(20).std().values
    # Donchian z UZAVRETYCH predoslych sviecok (bez lookahead)
    P["dch"] = pd.Series(h).rolling(20).max().shift(1).values
    P["dcl"] = pd.Series(l).rolling(20).min().shift(1).values
    # vcerajsie High/Low
    d = pd.Series(df.index.date, index=df.index)
    P["pdh"] = pd.Series(h, index=df.index).groupby(d).transform("max").groupby(d).transform("first").shift(1).values
    dayh = pd.Series(h, index=df.index).groupby(d).max()
    dayl = pd.Series(l, index=df.index).groupby(d).min()
    pdh = d.map(dayh.shift(1)).values.astype(float)
    pdl = d.map(dayl.shift(1)).values.astype(float)
    P["pdh"], P["pdl"] = pdh, pdl
    return P


# ---------- strategie: vracaju pole smerov (-1 / 0 / +1) ----------

def s_ema_cross(P):
    a, b = P["e9"], P["e21"]
    sig = np.zeros(P["n"], dtype=int)
    up = (a[:-1] <= b[:-1]) & (a[1:] > b[1:])
    dn = (a[:-1] >= b[:-1]) & (a[1:] < b[1:])
    sig[1:][up] = 1
    sig[1:][dn] = -1
    return sig


def s_donchian(P):
    c, sig = P["c"], np.zeros(P["n"], dtype=int)
    sig[c > P["dch"]] = 1
    sig[c < P["dcl"]] = -1
    return sig


def s_bb_fade(P):
    c = P["c"]
    up = P["sma20"] + 2.0 * P["sd20"]
    lo = P["sma20"] - 2.0 * P["sd20"]
    sig = np.zeros(P["n"], dtype=int)
    sig[c < lo] = 1
    sig[c > up] = -1
    return sig


def s_pullback(P):
    c, e20, e50, e200 = P["c"], P["e20"], P["e50"], P["e200"]
    sig = np.zeros(P["n"], dtype=int)
    bull = (e50 > e200)
    bear = (e50 < e200)
    cu = np.zeros(P["n"], dtype=bool); cd = np.zeros(P["n"], dtype=bool)
    cu[1:] = (c[:-1] <= e20[:-1]) & (c[1:] > e20[1:])
    cd[1:] = (c[:-1] >= e20[:-1]) & (c[1:] < e20[1:])
    sig[bull & cu] = 1
    sig[bear & cd] = -1
    return sig


def s_pdh_pdl(P):
    c, sig = P["c"], np.zeros(P["n"], dtype=int)
    with np.errstate(invalid="ignore"):
        sig[c > P["pdh"]] = 1
        sig[c < P["pdl"]] = -1
    return sig


def s_random(P):
    rng = np.random.default_rng(20260906)
    sig = np.zeros(P["n"], dtype=int)
    k = rng.random(P["n"])
    sig[k < 0.010] = 1
    sig[k > 0.990] = -1
    return sig


STRATS = [
    ("EMA 9/21 cross",      s_ema_cross),
    ("Donchian 20 breakout", s_donchian),
    ("Bollinger fade",      s_bb_fade),
    ("Pullback do EMA20",   s_pullback),
    ("Prerazenie PDH/PDL",  s_pdh_pdl),
    ("NAHODA (kontrola)",   s_random),
]


def simulate(P, sig, slm, rr, i0, i1):
    """Vracia zoznam vysledkov v R. Naraz max 1 pozicia."""
    c, h, l, atr, n = P["c"], P["h"], P["l"], P["atr"], P["n"]
    out = []
    i = max(i0, 210)
    end = min(i1, n - 2)
    while i < end:
        d = sig[i]
        if d == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        entry = c[i]
        dist = slm * atr[i]
        sl = entry - d * dist
        tp = entry + d * dist * rr
        r = None
        j = i + 1
        last = min(i + MAX_HOLD, end)
        while j <= last:
            if d == 1:
                if l[j] <= sl:
                    r = -1.0; break
                if h[j] >= tp:
                    r = rr; break
            else:
                if h[j] >= sl:
                    r = -1.0; break
                if l[j] <= tp:
                    r = rr; break
            j += 1
        if r is None:
            r = d * (c[min(j, n - 1)] - entry) / dist
        out.append(r - SPREAD_R)
        i = j + 1
    return out


def stats(rs):
    if len(rs) < 5:
        n = float("nan")
        return len(rs), n, n, n, n
    a = np.array(rs, dtype=float)
    w, ls = a[a > 0].sum(), -a[a < 0].sum()
    pf = w / ls if ls > 0 else float("inf")
    sd = a.std(ddof=1)
    t = a.mean() / (sd / math.sqrt(len(a))) if sd > 0 else 0.0
    wr = (a > 0).sum() / len(a) * 100.0
    return len(a), pf, a.mean(), t, wr


def fmt(x, w=6, dec=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return " " * (w - 1) + "-"
    s = f"{x:+.{dec}f}" if sign else f"{x:.{dec}f}"
    return s.rjust(w)


def main():
    data = {}
    for t, name in TICKERS.items():
        df = fetch(t)
        if df is None:
            print(f"{name}: data sa nestiahli")
            continue
        data[name] = prepare(df)
        print(f"{name}: {data[name]['n']} sviecok  ({df.index[0]:%d.%m.%Y} - {df.index[-1]:%d.%m.%Y})")
    if not data:
        return

    print("\n" + "=" * 74)
    print(f"SCREENER  |  {len(STRATS)} strategii x {len(RRS)*len(SLS)} nastaveni")
    print(f"obchody zlucene cez {len(data)} trhov, {INTERVAL} sviecky")
    print(f"prvych {int(TRAIN_FRAC*100)} % = LADENIE, zvysok = OVERENIE")
    print("=" * 74)
    print(f"{'STRATEGIA':<22}{'RR':>4}{'SLxATR':>7}"
          f"{'| n':>7}{'PF':>6}{'expR':>7}"
          f"{'|| n':>7}{'PF':>6}{'expR':>7}{'WR%':>6}{'t':>6}")
    print("-" * 80)

    for label, fn in STRATS:
        sigs = {k: fn(P) for k, P in data.items()}
        best = None
        for rr in RRS:
            for slm in SLS:
                tr, te = [], []
                for k, P in data.items():
                    sp = int(P["n"] * TRAIN_FRAC)
                    tr += simulate(P, sigs[k], slm, rr, 0, sp)
                    te += simulate(P, sigs[k], slm, rr, sp, P["n"])
                n1, pf1, e1, _, w1 = stats(tr)
                if n1 < 40:
                    continue
                n2, pf2, e2, t2, w2 = stats(te)
                cand = (pf1, rr, slm, n1, e1, n2, pf2, e2, t2, w2)
                if best is None or cand[0] > best[0]:
                    best = cand
        if best is None:
            print(f"{label:<22}   malo obchodov")
            continue
        pf1, rr, slm, n1, e1, n2, pf2, e2, t2, w2 = best
        print(f"{label:<22}{rr:>4.1f}{slm:>7.1f}"
              f"{n1:>7}{fmt(pf1)}{fmt(e1, 7, 2, True)}"
              f"{n2:>7}{fmt(pf2)}{fmt(e2, 7, 2, True)}"
              f"{fmt(w2, 6, 1)}{fmt(t2, 6, 1, True)}")

    print("-" * 80)
    print("vlavo od |  = LADENIE (vyber nastavenia)")
    print("vpravo od || = OVERENIE (jedine, co plati)")
    print("t = t-statistika overenia. Pod 2.0 = neodlisitelne od nahody.")
    print("NAHODA je kontrola: strategia pod nou nema ziadnu hodnotu.")
    print("WR% = win rate na overeni. Vsimni si, ze vysoky WR (RR 1.0)")
    print("NEZNAMENA vyssi zisk - expR je to, na com zalezi.")


if __name__ == "__main__":
    main()
