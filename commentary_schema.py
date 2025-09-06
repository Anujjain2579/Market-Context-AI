from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, ValidationError, root_validator, validator, field_validator, model_validator
import re
from datetime import datetime
import json
import string

FUND_TERMS_RE = re.compile(
    r'\b(we|our|us|the fund|portfolio|overweight|underweight|selection|allocation)\b',
    flags=re.IGNORECASE
)

PERIOD_Q_RE = re.compile(r'^(Q[1-4]\s20\d{2})$')
PERIOD_RANGE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}$')

def _reject_fund_terms(text: Optional[str], field: str, errors: List[str]) -> None:
    if not text:
        return
    if FUND_TERMS_RE.search(text):
        errors.append(f"{field}: contains forbidden fund-specific language")

def _date_iso(s: str) -> bool:
    try:
        datetime.fromisoformat(s)
        return True
    except Exception:
        return False
    
class IndexEvent(BaseModel):
    date: str = Field(..., description="ISO date, e.g., 2025-04-08")
    description: str
    one_day_move_pct: Optional[float] = None

    @field_validator("date")
    def date_must_be_iso(cls, v):
        if not _date_iso(v):
            raise ValueError("date must be ISO format YYYY-MM-DD")
        return v

class Macro(BaseModel):
    inflation: Optional[str] = None
    policy_rate: Optional[str] = None
    yields: Optional[str] = None
    growth: Optional[str] = None
    employment: Optional[str] = None
    fx: Optional[str] = None
    credit: Optional[str] = None
    commodities: Optional[str] = None
    policy_geopolitics: Optional[List[str]] = None

class BreadthConcentration(BaseModel):
    description: Optional[str] = None
    top_names_contribution_pct: Optional[str] = None
    advance_decline: Optional[str] = None

class SectorTheme(BaseModel):
    sector: str
    theme: str

class StyleRotation(BaseModel):
    value_vs_growth: Optional[str] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    volatility: Optional[str] = None

class MarketContextInput(BaseModel):
    period: str
    market_region: str
    benchmark: str
    benchmark_return_total_pct: Optional[float] = Field(
        None, description="Total return % for the period (e.g., 8.5 for +8.5%)"
    )
    qualitative_market_move: Optional[str] = None
    index_level_events: Optional[List[IndexEvent]] = None
    macro: Optional[Macro] = None
    breadth_concentration: Optional[BreadthConcentration] = None
    sector_themes: Optional[List[SectorTheme]] = None
    style_rotation: Optional[StyleRotation] = None
    disclaimers: Optional[str] = None

    @field_validator("period")
    def period_format(cls, v):
        if PERIOD_Q_RE.match(v) or PERIOD_RANGE_RE.match(v):
            return v
        raise ValueError("period must be 'Q# YYYY' or 'YYYY-MM-DD to YYYY-MM-DD'")

    @field_validator("benchmark_return_total_pct")
    def return_bounds(cls, v):
        if v is None:
            return v
        if not (-100.0 <= v <= 100.0):
            raise ValueError("benchmark_return_total_pct must be between -100 and 100")
        return v

    @model_validator(mode='before')
    def required_combo(cls, values, info):
        period = values.get("period")
        benchmark = values.get("benchmark")
        ret = values.get("benchmark_return_total_pct")
        qual = values.get("qualitative_market_move")
        if not period or not benchmark:
            raise ValueError("required: period and benchmark")
        if ret is None and not qual:
            raise ValueError("required: at least one of benchmark_return_total_pct or qualitative_market_move")
        return values



def validate_payload(payload: Dict[str, Any]) -> Tuple[Optional[MarketContextInput], List[str]]:
    errors: List[str] = []

    # Schema validation
    try:
        obj = MarketContextInput(**payload)
    except ValidationError as ve:
        errors.extend([e['msg'] if isinstance(e, dict) else str(e) for e in ve.errors()])
        return None, errors

    # Content sanitation (forbidden fund-specific language)
    _reject_fund_terms(obj.market_region, "market_region", errors)
    _reject_fund_terms(obj.benchmark, "benchmark", errors)
    _reject_fund_terms(obj.qualitative_market_move, "qualitative_market_move", errors)
    if obj.macro:
        for k, v in obj.macro.model_dump().items():
            if isinstance(v, str):
                _reject_fund_terms(v, f"macro.{k}", errors)
            elif isinstance(v, list):
                for i, s in enumerate(v):
                    _reject_fund_terms(s, f"macro.policy_geopolitics[{i}]", errors)
    if obj.breadth_concentration:
        for k, v in obj.breadth_concentration.model_dump().items():
            if isinstance(v, str):
                _reject_fund_terms(v, f"breadth_concentration.{k}", errors)
    if obj.sector_themes:
        for i, st in enumerate(obj.sector_themes):
            _reject_fund_terms(st.sector, f"sector_themes[{i}].sector", errors)
            _reject_fund_terms(st.theme, f"sector_themes[{i}].theme", errors)
    if obj.style_rotation:
        for k, v in obj.style_rotation.model_dump().items():
            if isinstance(v, str):
                _reject_fund_terms(v, f"style_rotation.{k}", errors)
    _reject_fund_terms(obj.disclaimers, "disclaimers", errors)

    # Numeric sanity checks beyond schema
    if obj.index_level_events:
        for i, ev in enumerate(obj.index_level_events):
            if ev.one_day_move_pct is not None and not (-100.0 <= ev.one_day_move_pct <= 100.0):
                errors.append(f"index_level_events[{i}].one_day_move_pct must be between -100 and 100")

    # If period is a date range, ensure valid ordering
    if PERIOD_RANGE_RE.match(obj.period):
        start_s, _, end_s = obj.period.partition(" to ")
        if datetime.fromisoformat(start_s) > datetime.fromisoformat(end_s):
            errors.append("period range start must be <= end")

    return (obj if not errors else None), errors


if __name__ == "__main__":
    # Minimal smoke test with your example payload
    sample: Dict[str, Any] = {
      "period": "Q2 2025",
      "market_region": "U.S. equities (small-cap)",
      "benchmark": "Russell 2000",
      "benchmark_return_total_pct": -1.79,
      "index_level_events": [
        {"date": "2025-04-08", "description": "Tariff pause announced", "one_day_move_pct": 9.5}
      ],
      "macro": {
        "inflation": "Headline CPI 3.1% YoY, easing",
        "policy_rate": "Fed on hold; markets price 2–3 cuts in 2025",
        "yields": "10y UST ~4.22%",
        "growth": "EPS revisions trimmed; GDP resilient",
        "employment": "not provided",
        "fx": "USD weaker vs. basket",
        "credit": "High-yield spreads narrowed",
        "commodities": "not provided",
        "policy_geopolitics": ["Tariffs narrative volatility", "Geopolitical flare-ups"]
      },
      "breadth_concentration": {
        "description": "Risk-on, high beta outperformed; market narrowness increased",
        "top_names_contribution_pct": "not provided",
        "advance_decline": "not provided"
      },
      "sector_themes": [
        {"sector": "Information Technology", "theme": "AI-related demand; strong rally"},
        {"sector": "Industrials", "theme": "mixed; machinery resilient"},
        {"sector": "Health Care", "theme": "Biotech weakness"}
      ],
      "style_rotation": {
        "value_vs_growth": "Growth outperformed",
        "size": "Large > Small",
        "quality": "Low-quality beta led",
        "volatility": "Elevated headline-driven swings; VIX level not provided"
      },
      "disclaimers": "No forecasts; monitoring policy path and earnings dispersion"
    }

    obj, errs = validate_payload(sample)
    if errs:
        print("INVALID\n")
        for e in errs:
            print(f"- {e}")
    else:
        print("VALID\n")
        print(json.dumps(obj.model_dump(), indent=2))