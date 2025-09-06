from __future__ import annotations
import os, json, re, math, textwrap
from typing import List, Tuple
from pydantic import BaseModel
from typing import Optional
from commentary_schema import MarketContextInput, validate_payload

FORBIDDEN = re.compile(r'\b(we|our|us|the fund|portfolio|overweight|underweight|selection|allocation)\b', re.I)
api_key = os.getenv("OPENAI_API_KEY")

def to_compact_json(model: MarketContextInput) -> str:
    return json.dumps(model.model_dump(), separators=(",", ":"), ensure_ascii=False)

def _region_guard(model: MarketContextInput) -> str:
    r = model.market_region.strip()
    return (
        f"Region focus: {r}. Tailor all context to this region and the stated benchmark. "
        "Avoid U.S.-centric details if region is not U.S.; if inputs are U.S.-only, treat them as global backdrop and mark local items as 'not provided'. "
        "Do not generalize across regions."
    )

def assemble_messages(payload: MarketContextInput, events_narrative: Optional[str], para_min=2, para_max=4) -> List[dict]:
    system_rules = (
        "You are an investment commentator. Write the Market Context for the specified period. "
        "Use only the data in INPUT and EVENTS. No fund/portfolio mentions. "
        "If a data point is missing, write “not provided”. Do not invent numbers. "
        f"Deliverable: Begin with '## Market Context'. 150–250 words; {para_min}–{para_max} paragraphs. "
        "Para 1: Macro & policy (inflation, rates, growth, FX/credit, notable policy/geopolitics). "
        "Para 2: Market behavior (index return, volatility, sector trends, factor/style rotation, breadth/concentration). "
        "Include an extra paragraph only if EVENTS is non-empty, summarizing the outsized move(s) neutrally. "
        "Style: factual, concise, client-friendly. Numbers must match INPUT exactly."
        "Select the most relevant macro items for the specified market region; less relevant items may be omitted."
    )
    region_guard = _region_guard(payload)
    salience = "Salience guidance: " + " ".join(_salience_hints(payload))
    content = f"INPUT JSON:\n{to_compact_json(payload)}\n\nEVENTS:\n{events_narrative or '[]'}\n\n{region_guard}"
    return [{"role": "system", "content": system_rules}, {"role": "user", "content": content}]


def _salience_hints(payload: MarketContextInput) -> list[str]:
    mr = payload.market_region.lower()
    hints = []
    if "emerging" in mr:
        hints = [
            "Prioritize USD vs EM FX changes and commodities.",
            "Use global rates as backdrop; avoid U.S.-centric labor unless explicitly relevant.",
            "If policy_geopolitics keywords exist, reference them generically."
        ]
    elif "europe" in mr:
        hints = [
            "Prioritize EUR/USD and EUR/GBP and changes in rates that affect Europe.",
            "Commodities relevant to Europe can be mentioned briefly.",
        ]
    else:
        hints = [
            "Prioritize domestic inflation, policy rate, unemployment, 10y yield.",
            "Include USD FX only as secondary context."
        ]
    return hints


def basic_checks(text: str, para_min=2, para_max=4) -> List[str]:
    errs: List[str] = []
    if not text.lstrip().startswith("## Market Context"):
        errs.append("must start with '## Market Context'")
    if FORBIDDEN.search(text):
        errs.append("contains forbidden fund/attribution terms")
    words = re.findall(r"\b\w[\w%\-/.]*\b", text)
    n = len(words)
    if n < 150 or n > 300:
        errs.append(f"word count {n} not in [150,250]")
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paras) < para_min or len(paras) > para_max:
        errs.append(f"paragraphs not in [{para_min},{para_max}]")
    if re.search(r"\[(?:.+?)\]", text):
        errs.append("found bracketed placeholder")
    return errs

def make_critique(errs: List[str]) -> str:
    return "Fix these issues without changing facts: " + "; ".join(errs) + ". Keep 150–250 words."

def _client():
    from openai import OpenAI
    return OpenAI(api_key=api_key)

def _chat(messages: List[dict], model: Optional[str] = None, temperature: float = 0.2) -> str:
    client = _client()
    mdl = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    resp = client.chat.completions.create(model=mdl, messages=messages, temperature=temperature)
    return resp.choices[0].message.content.strip()

# Event narration
def narrate_events_with_llm(events: List[dict], calendar_hints: List[str], market_region: str, benchmark: str,
                            model: Optional[str] = None) -> str:
    if not events:
        return "[]"
    sys = (
        "You convert index move flags into short neutral narratives for a 'notable events' paragraph. "
        "Avoid fund terms. Do not fabricate causes. If unsure, attribute to 'headline-driven volatility' or 'macro releases'. "
        "One compact sentence per event; include the date and the magnitude with % sign. Keep region context."
    )
    usr = {
        "events": events,                      # e.g., [{'date':'2025-04-08','pct':3.2}, ...]
        "calendar_hints": calendar_hints,      # e.g., ['CPI:Headline CPI 3.1% YoY', 'FOMC:Fed funds rate 5.25%']
        "region": market_region,
        "benchmark": benchmark
    }
    messages = [{"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(usr, ensure_ascii=False)}]
    text = _chat(messages, model=model, temperature=0.2)
    lines = [ln.strip() for ln in re.split(r"[;\n]+", text) if ln.strip()]
    # Return as JSON-ish list to feed into the main generator
    return json.dumps(lines, ensure_ascii=False)

def generate_market_context(payload: MarketContextInput,
                            events: List[dict],
                            calendar_hints: List[str],
                            max_retries: int = 2,
                            para_min: int = 2,
                            para_max: int = 4) -> str:
    events_narr = narrate_events_with_llm(events, calendar_hints, payload.market_region, payload.benchmark)
    msgs = assemble_messages(payload, events_narr, para_min=para_min, para_max=para_max)
    text = _chat(msgs, temperature=0.2)
    errs = basic_checks(text, para_min=para_min, para_max=para_max)
    retries = 0
    while errs and retries < max_retries:
        critique = make_critique(errs)
        msgs = msgs + [{"role": "system", "content": critique}]
        text = _chat(msgs, temperature=0.2)
        errs = basic_checks(text, para_min=para_min, para_max=para_max)
        retries += 1
    if errs:
        raise ValueError("Post-checks failed:\n" + "\n".join(f"- {e}" for e in errs))
    return text

