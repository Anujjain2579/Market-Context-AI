from __future__ import annotations
import os
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import string, math
from commentary_schema import (
    MarketContextInput,
    Macro,
    BreadthConcentration,
    SectorTheme,
    StyleRotation,
    IndexEvent,
    MarketContextInput,
    validate_payload,
)
load_dotenv()

# Access the API key
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY")
# Utilities

def _iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")

def _retry_get(url: str, params: Dict[str, str], tries: int = 3, backoff: float = 1.5) -> Dict:
    for i in range(tries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(backoff * (i + 1))
    raise RuntimeError(f"GET failed: {url} {params}")

def _to_df_time_series_alpha(payload: Dict, key: str) -> pd.DataFrame:
    if key not in payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload[key]).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for c in df.columns:
        with np.errstate(all='ignore'):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _period_bounds(period: str) -> Tuple[date, date]:
    if period.startswith("Q"):
        q, y = period.split()
        year = int(y)
        qn = int(q[1])
        start_month = 3*(qn-1)+1
        start = date(year, start_month, 1)
        end = (start + relativedelta(months=3)) - relativedelta(days=1)
        return start, end
    # Custom range: YYYY-MM-DD to YYYY-MM-DD
    s, _, e = period.partition(" to ")
    return datetime.fromisoformat(s).date(), datetime.fromisoformat(e).date()

# Source adapters (Alpha Vantage)

def av_cpi_series() -> Optional[str]:
    if not ALPHAVANTAGE_API_KEY:
        return None
    url = "https://www.alphavantage.co/query"
    params = {"function": "CPI", "interval": "monthly", "apikey": ALPHAVANTAGE_API_KEY}
    js = _retry_get(url, params)
    data = js.get("data")
    if not data:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    # Year-over-year calculation if not provided
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "value" not in df or df["value"].isna().all():
        return None
    latest = df.iloc[-1]
    idx = df.index.get_loc(latest.name)
    if idx >= 12:
        yoy = (latest["value"] / df.iloc[idx-12]["value"] - 1.0) * 100.0
        return f"Headline CPI {yoy:.1f}% YoY"
    return "Headline CPI not provided"

def av_fed_funds_rate() -> Optional[str]:
    if not ALPHAVANTAGE_API_KEY:
        return None
    url = "https://www.alphavantage.co/query"
    params = {"function": "FEDERAL_FUNDS_RATE", "interval": "monthly", "apikey": ALPHAVANTAGE_API_KEY}
    js = _retry_get(url, params)
    data = js.get("data")
    if not data:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    latest = df.dropna(subset=["value"]).iloc[-1]["value"]
    return f"Fed funds rate {latest:.2f}%"

def av_treasury_10y_yield() -> Optional[str]:
    if not ALPHAVANTAGE_API_KEY:
        return None
    url = "https://www.alphavantage.co/query"
    params = {"function": "TREASURY_YIELD", "interval": "daily", "maturity": "10year", "apikey": ALPHAVANTAGE_API_KEY}
    js = _retry_get(url, params)
    data = js.get("data")
    if not data:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    latest = df.dropna(subset=["value"]).iloc[-1]["value"]
    return f"10y UST ~{latest:.2f}%"

# expose this so step 3 can reuse the price series
def daily_prices_for_benchmark(period: str, benchmark: str) -> pd.Series:
    start, end = _period_bounds(period)
    info = BENCHMARK_MAP.get(benchmark)
    if not info:
        return pd.Series(dtype=float)
    df = _download_daily_from_alpha(info["symbol"], start, end)
    if df.empty:
        return pd.Series(dtype=float)
    s = df["adj_close"].copy()
    s.name = info["symbol"]
    return s

def detect_outsized_moves_from_prices(prices: pd.Series, z: float = 1.5, max_events: int = 3) -> List[Dict]:
    if prices is None or prices.size < 5:
        return []
    rets = prices.pct_change().dropna()
    sigma = rets.std(ddof=1)
    if not np.isfinite(sigma) or sigma == 0:
        return []
    mask = rets.abs() >= (z * sigma)
    flagged = rets.loc[mask].sort_values(key=lambda x: -x.abs())
    events: List[Dict] = []
    for dt, r in flagged.head(max_events).items():
        events.append({"date": dt.strftime("%Y-%m-%d"), "pct": round(r * 100.0, 2)})
    return events

def curated_macro_calendar(period: str) -> List[str]:
    hints: List[str] = []
    cpi = av_cpi_series()
    if cpi:
        hints.append(f"CPI:{cpi}")
    ffr = av_fed_funds_rate()
    if ffr:
        hints.append(f"FOMC:{ffr}")
    y10 = av_treasury_10y_yield()
    if y10:
        hints.append(f"UST10Y:{y10}")
    return hints

#Additions

# FX pairs to consider by region keyword
FX_REGION_MAP = {
    "u.s.": [("USD","JPY"), ("EUR","USD")],
    "us ": [("USD","JPY"), ("EUR","USD")],
    "united states": [("USD","JPY"), ("EUR","USD")],
    "emerging": [("USD","CNY"), ("USD","INR"), ("USD","BRL")],
    "europe": [("EUR","USD"), ("EUR","GBP"), ("USD","CHF")],
    "japan": [("USD","JPY")],
    "uk": [("GBP","USD"), ("EUR","GBP")],
}

def _region_fx_pairs(market_region: str) -> list[tuple[str,str]]:
    mr = market_region.lower()
    for k, pairs in FX_REGION_MAP.items():
        if k in mr:
            return pairs
    return [("EUR","USD"), ("USD","JPY")]

def av_unemployment_rate() -> Optional[str]:
    if not ALPHAVANTAGE_API_KEY:
        return None
    js = _retry_get("https://www.alphavantage.co/query",
                    {"function":"UNEMPLOYMENT","apikey":ALPHAVANTAGE_API_KEY})
    data = js.get("data")
    if not data:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    latest = df.dropna(subset=["value"]).iloc[-1]
    return f"Unemployment {latest['value']:.1f}%"

def _fx_daily_pair_df(base: str, quote: str) -> pd.DataFrame:
    if not ALPHAVANTAGE_API_KEY:
        return pd.DataFrame()
    js = _retry_get("https://www.alphavantage.co/query", {
        "function":"FX_DAILY", "from_symbol":base, "to_symbol":quote,
        "outputsize":"full", "apikey":ALPHAVANTAGE_API_KEY
    })
    key = "Time Series FX (Daily)"
    if key not in js:
        return pd.DataFrame()
    df = pd.DataFrame(js[key]).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    col = "4. close" if "4. close" in df.columns else list(df.columns)[-1]
    df = df[[col]].rename(columns={col:"close"})
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df

def fx_changes_for_period(period: str, pairs: list[tuple[str,str]]) -> list[tuple[str,float]]:
    start, end = _period_bounds(period)
    out = []
    for base, quote in pairs:
        df = _fx_daily_pair_df(base, quote)
        if df.empty:
            continue
        s = df.loc[(df.index.date >= start) & (df.index.date <= end), "close"]
        if len(s) < 2:
            continue
        ret = (s.iloc[-1] / s.iloc[0] - 1.0) * 100.0
        out.append((f"{base}/{quote}", float(round(ret, 2))))
    return out

def av_all_commodities_mom_yoy(period: str) -> Optional[str]:
    if not ALPHAVANTAGE_API_KEY:
        return None
    js = _retry_get("https://www.alphavantage.co/query",
                    {"function":"ALL_COMMODITIES","interval":"monthly","apikey":ALPHAVANTAGE_API_KEY})
    data = js.get("data")
    if not data:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date")
    start, end = _period_bounds(period)
    in_q = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]
    if in_q.empty:
        latest = df.iloc[-1]
        return f"All Commodities index {latest['value']:.1f} (period data not provided)"
    last = in_q.iloc[-1]
    # MoM vs previous monthly obs
    prev_idx = df.index.get_loc(last.name) - 1
    mom = None
    if prev_idx >= 0:
        prev_val = df.iloc[prev_idx]["value"]
        mom = (last["value"]/prev_val - 1.0)*100.0
    # YoY vs 12 months back
    idx = df.index.get_loc(last.name)
    yoy = None
    if idx >= 12:
        yoy = (last["value"]/df.iloc[idx-12]["value"] - 1.0)*100.0
    parts = [f"All Commodities index {last['value']:.1f}"]
    if mom is not None and np.isfinite(mom):
        parts.append(f"MoM {mom:.1f}%")
    if yoy is not None and np.isfinite(yoy):
        parts.append(f"YoY {yoy:.1f}%")
    return "; ".join(parts)

def av_real_gdp_yoy() -> Optional[str]:
    if not ALPHAVANTAGE_API_KEY:
        return None
    js = _retry_get("https://www.alphavantage.co/query",
                    {"function":"REAL_GDP","interval":"annual","apikey":ALPHAVANTAGE_API_KEY})
    data = js.get("data")
    if not data:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date")
    if len(df) < 2:
        return f"Real GDP {df.iloc[-1]['value']:.1f} (YoY not provided)"
    yoy = (df.iloc[-1]["value"]/df.iloc[-2]["value"] - 1.0)*100.0
    return f"Real GDP YoY {yoy:.1f}% (annual)"

def news_topics_from_av(time_from_iso: str, tickers: list[str], limit: int = 100) -> list[str]:
    if not ALPHAVANTAGE_API_KEY:
        return []
    js = _retry_get("https://www.alphavantage.co/query", {
        "function":"NEWS_SENTIMENT",
        "tickers": ",".join(tickers),
        "time_from": time_from_iso,
        "limit": str(limit),
        "apikey": ALPHAVANTAGE_API_KEY
    })
    feed = js.get("feed", [])
    if not feed:
        return []
    texts = []
    for it in feed:
        t = (it.get("title") or "") + " " + (it.get("summary") or "")
        texts.append(t)
    blob = " ".join(texts).lower()
    stop = set("""a an the of for to and or in on with by from into over under amid during as at vs versus via
    is are was were be been being this that these those it its their our your his her them they we you
    stocks shares market markets policy economic federal central bank rate rates inflation deflation
    growth recession expansion gdp cpi ppi""".split())
    tokens = [w.strip(string.punctuation) for w in blob.split()]
    tokens = [w for w in tokens if len(w) > 3 and w not in stop]
    if not tokens:
        return []
    vc = pd.Series(tokens).value_counts()
    return [w for w in vc.index[:5]]

def _period_time_from(period: str) -> str:
    start, _ = _period_bounds(period)
    return f"{start.strftime('%Y%m%d')}T0000"

def _region_news_tickers(market_region: str) -> list[str]:
    mr = market_region.lower()
    if "emerging" in mr:
        return ["FOREX:USD", "FOREX:CNY", "FOREX:BRL", "CRYPTO:BTC"]
    if "europe" in mr:
        return ["FOREX:EUR", "FOREX:GBP", "FOREX:USD"]
    return ["FOREX:USD", "CRYPTO:BTC"]

# Benchmarks & returns 

BENCHMARK_MAP = {
    # extend as needed
    "S&P 500": {"symbol": "SPY", "source": "databento_or_alpha"},
    "S&P 500 Equal Weight": {"symbol": "RSP", "source": "databento_or_alpha"},
    "Russell 3000": {"symbol": "IWV", "source": "databento_or_alpha"},
    "Russell 1000": {"symbol": "IWB", "source": "databento_or_alpha"},
    "Russell 1000 Growth": {"symbol": "IWF", "source": "databento_or_alpha"},
    "Russell 1000 Value": {"symbol": "IWD", "source": "databento_or_alpha"},
    "Russell Midcap": {"symbol": "IWR", "source": "databento_or_alpha"},
    "Russell 2000": {"symbol": "IWM", "source": "databento_or_alpha"},
    "MSCI ACWI": {"symbol": "ACWI", "source": "databento_or_alpha"},
    "MSCI EAFE": {"symbol": "EFA", "source": "databento_or_alpha"},
    "MSCI Emerging Markets": {"symbol": "EEM", "source": "databento_or_alpha"},
    "NASDAQ-100": {"symbol": "QQQ", "source": "databento_or_alpha"},
}

def _download_daily_from_alpha(symbol: str, start: date, end: date) -> pd.DataFrame:
    if not ALPHAVANTAGE_API_KEY:
        return pd.DataFrame()
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol, "outputsize": "full", "apikey": ALPHAVANTAGE_API_KEY}
    js = _retry_get(url, params)
    df = _to_df_time_series_alpha(js, "Time Series (Daily)")
    if df.empty:
        return df
    # Use adjusted close (field '5. adjusted close') if available, else '4. close'
    col = "5. adjusted close" if "5. adjusted close" in df.columns else "4. close"
    out = df[[col]].rename(columns={col: "adj_close"})
    out = out.loc[(out.index.date >= start) & (out.index.date <= end)]
    return out

def _total_return_pct_from_prices(prices: pd.Series) -> Optional[float]:
    if prices is None or len(prices) < 2:
        return None
    start, end = prices.iloc[0], prices.iloc[-1]
    if pd.isna(start) or pd.isna(end) or start <= 0:
        return None
    return (end / start - 1.0) * 100.0

def compute_benchmark_return(period: str, benchmark: str) -> Optional[float]:
    start, end = _period_bounds(period)
    info = BENCHMARK_MAP.get(benchmark)
    if not info:
        return None
    sym = info["symbol"]
    # Try Alpha Vantage first (simple, no Databento dependency for ETFs)
    df = _download_daily_from_alpha(sym, start, end)
    if df.empty:
        return None
    return _total_return_pct_from_prices(df["adj_close"])

#  simple sector proxies via SPDRs (if user wants sector color)

SECTOR_ETFS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

def sector_total_returns(period: str) -> List[Tuple[str, float]]:
    start, end = _period_bounds(period)
    out: List[Tuple[str, float]] = []
    for sector, ticker in SECTOR_ETFS.items():
        df = _download_daily_from_alpha(ticker, start, end)
        if df.empty:
            continue
        tr = _total_return_pct_from_prices(df["adj_close"])
        if tr is not None:
            out.append((sector, tr))
    return out

def build_market_context_payload(
    period: str,
    market_region: str,
    benchmark: str,
    index_level_events: Optional[List[IndexEvent]] = None,
    qualitative_market_move: Optional[str] = None,
) -> MarketContextInput:
    macro_lines: Dict[str, Optional[str]] = {
        "inflation": av_cpi_series() or "not provided",
        "policy_rate": av_fed_funds_rate() or "not provided",
        "yields": av_treasury_10y_yield() or "not provided",
        "growth": av_real_gdp_yoy() or "not provided",
        "employment": av_unemployment_rate() or "not provided",
        "fx": "not provided",
        "credit": "not provided",
        "commodities": av_all_commodities_mom_yoy(period) or "not provided",
        "policy_geopolitics": None,
    }

    bench_ret = compute_benchmark_return(period, benchmark)

    fx_pairs = _region_fx_pairs(market_region)
    fx_stats = fx_changes_for_period(period, fx_pairs)
    if fx_stats:
        fx_bits = [f"{p} {v:+.2f}%" for p, v in fx_stats]
        macro_lines["fx"] = "; ".join(fx_bits)

    # news → geopolitics hints (keywords only, no claims/numbers)
    time_from = _period_time_from(period)
    topics = news_topics_from_av(time_from, _region_news_tickers(market_region), limit=200)
    macro_lines["policy_geopolitics"] = topics if topics else ["not provided"]

    sector_rets = sector_total_returns(period)
    sector_themes: List[SectorTheme] = []
    if sector_rets:
        srtd = sorted(sector_rets, key=lambda x: x[1], reverse=True)
        for s, v in srtd[:2]:
            sector_themes.append(SectorTheme(sector=s, theme=f"{s} led; proxy ETF return {v:.1f}%"))
        for s, v in srtd[-2:]:
            sector_themes.append(SectorTheme(sector=s, theme=f"{s} lagged; proxy ETF return {v:.1f}%"))

    payload_dict = {
        "period": period,
        "market_region": market_region,
        "benchmark": benchmark,
        "benchmark_return_total_pct": bench_ret,
        "qualitative_market_move": qualitative_market_move,
        "index_level_events": index_level_events,
        "macro": Macro(**macro_lines).model_dump(),
        "breadth_concentration": BreadthConcentration(
            description="not provided",
            top_names_contribution_pct="not provided",
            advance_decline="not provided",
        ).model_dump(),
        "sector_themes": [st.model_dump() for st in sector_themes] if sector_themes else None,
        "style_rotation": StyleRotation(
            value_vs_growth="not provided",
            size="not provided",
            quality="not provided",
            volatility="not provided",
        ).model_dump(),
        "disclaimers": "No forecasts; monitoring policy path and earnings dispersion",
    }

    obj, errs = validate_payload(payload_dict)
    if errs:
        raise ValueError("Payload validation failed:\n" + "\n".join(f"- {e}" for e in errs))
    return obj