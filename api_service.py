from __future__ import annotations
import os
import json
import logging
import re
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from commentary_schema import MarketContextInput
from ingest_normalize import (
    build_market_context_payload,
    daily_prices_for_benchmark,
    detect_outsized_moves_from_prices,
    curated_macro_calendar,
)
from generate_market_context import generate_market_context
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from datetime import date
from fastapi.responses import PlainTextResponse, HTMLResponse

PRESETS = [
    {"id": "us_large_core", "label": "U.S. equities (large-cap core)", "benchmark": "S&P 500"},
    {"id": "us_all_cap_core", "label": "U.S. equities (all-cap core)", "benchmark": "Russell 3000"},
    {"id": "us_small_cap", "label": "U.S. equities (small-cap)", "benchmark": "Russell 2000"},
    {"id": "us_large_growth", "label": "U.S. equities (large-cap growth)", "benchmark": "Russell 1000 Growth"},
    {"id": "us_large_value", "label": "U.S. equities (large-cap value)", "benchmark": "Russell 1000 Value"},
    {"id": "us_mid_cap", "label": "U.S. equities (mid-cap)", "benchmark": "Russell Midcap"},
    {"id": "global_equity", "label": "Global equities", "benchmark": "MSCI ACWI"},
    {"id": "intl_dev", "label": "International developed equities", "benchmark": "MSCI EAFE"},
    {"id": "em_equity", "label": "Emerging markets equities", "benchmark": "MSCI Emerging Markets"},
    {"id": "us_equal_weight", "label": "U.S. equities (equal-weight)", "benchmark": "S&P 500 Equal Weight"},
    {"id": "global_growth_tech", "label": "Global large-cap growth (tech tilt)", "benchmark": "NASDAQ-100"},
]

BENCHMARK_MAP = {
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


app = FastAPI(title="Market Context Generator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", response_class=PlainTextResponse)
def root():
    return "Market Context Generator API\n\nTry:\n- GET /health\n- GET /docs\n- GET /market-context/raw?period=Q2%202025&market_region=U.S.%20equities%20(small-cap)&benchmark=Russell%202000\n- GET /ui\n"

@app.get("/market-context/raw", response_class=PlainTextResponse)
def market_context_raw(
    period: str = Query(..., description="Q# YYYY or YYYY-MM-DD to YYYY-MM-DD"),
    market_region: str = Query(...),
    benchmark: str = Query(...),
    para_min: int = Query(2),
    para_max: int = Query(4),
    z_threshold: float = Query(1.5),
    max_events: int = Query(3),
):
    payload = build_market_context_payload(
        period=period,
        market_region=market_region,
        benchmark=benchmark,
        index_level_events=None,
        qualitative_market_move=None,
    )
    px = daily_prices_for_benchmark(payload.period, payload.benchmark)
    events = detect_outsized_moves_from_prices(px, z=z_threshold, max_events=max_events)
    cal = curated_macro_calendar(payload.period)
    md = generate_market_context(
        payload=payload,
        events=events,
        calendar_hints=cal,
        max_retries=2,
        para_min=para_min,
        para_max=para_max,
    )
    return md + "\n"

def _quarters_desc(start_q="Q2 2025", end_q="Q1 2023"):
    def parse(qs):
        q, y = qs.split()
        return int(y), int(q[1])
    def step_back(y, q):
        if q > 1: return y, q-1
        return y-1, 4
    y, q = parse(start_q)
    ye, qe = parse(end_q)
    out = []
    while (y > ye) or (y == ye and q >= qe):
        out.append(f"Q{q} {y}")
        y, q = step_back(y, q)
    return out

@app.get("/presets")
def presets():
    return {
        "quarters": _quarters_desc("Q2 2025", "Q1 2023"),
        "regions": PRESETS,
        "benchmarks": list(BENCHMARK_MAP.keys()),
    }


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcg")

FORBIDDEN = re.compile(r'\b(we|our|us|the fund|portfolio|overweight|underweight|selection|allocation)\b', re.I)

class GenerateRequest(BaseModel):
    period: str = Field(..., description="Q# YYYY or YYYY-MM-DD to YYYY-MM-DD")
    market_region: str = Field(..., description="e.g., U.S. equities (small-cap)")
    benchmark: str = Field(..., description="e.g., Russell 2000")
    para_min: int = 2
    para_max: int = 4
    z_threshold: float = 1.5
    max_events: int = 3
    openai_model: Optional[str] = None

class GenerateResponse(BaseModel):
    market_context_markdown: str
    payload: Dict[str, Any]
    events: List[Dict[str, Any]]
    stats: Dict[str, int]

def _word_count(s: str) -> int:
    return len(re.findall(r"\b\w[\w%\-/.]*\b", s))

def _para_count(s: str) -> int:
    return len([p for p in re.split(r"\n\s*\n", s.strip()) if p.strip()])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/market-context", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        payload: MarketContextInput = build_market_context_payload(
            period=req.period,
            market_region=req.market_region,
            benchmark=req.benchmark,
            index_level_events=None,
            qualitative_market_move=None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"payload build/validation error: {str(e)}")

    try:
        px = daily_prices_for_benchmark(payload.period, payload.benchmark)
        events = detect_outsized_moves_from_prices(px, z=req.z_threshold, max_events=req.max_events)
        cal = curated_macro_calendar(payload.period)
        md = generate_market_context(
            payload=payload,
            events=events,
            calendar_hints=cal,
            max_retries=2,
            para_min=req.para_min,
            para_max=req.para_max,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation error: {str(e)}")

    return GenerateResponse(
        market_context_markdown=md,
        payload=payload.model_dump(),
        events=events,
        stats={"words": _word_count(md), "paragraphs": _para_count(md)},
    )

# api_service.py (replace your /app handler with this)
@app.get("/app", response_class=HTMLResponse)
def app_page():
    return """
<!doctype html>
<meta charset="utf-8">
<title>Market Context Generator</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css">
<div class="min-h-screen bg-gray-50">
  <header class="bg-white shadow">
    <div class="max-w-5xl mx-auto py-6 px-4">
      <h1 class="text-2xl font-semibold">Market Context Generator</h1>
      <p class="text-gray-600">Choose a quarter and market region, then generate the commentary.</p>
    </div>
  </header>

  <main class="max-w-5xl mx-auto p-4 space-y-6">
    <section class="bg-white rounded-2xl shadow p-6">
      <div class="grid md:grid-cols-3 gap-4">
        <div>
          <label class="block text-sm font-medium">Period (quarter)</label>
          <select id="period" class="mt-1 w-full border rounded-lg px-3 py-2"></select>
        </div>
        <div>
          <label class="block text-sm font-medium">Market region (preset)</label>
          <select id="region" class="mt-1 w-full border rounded-lg px-3 py-2"></select>
        </div>
        <div>
          <label class="block text-sm font-medium">Benchmark</label>
          <select id="bench" class="mt-1 w-full border rounded-lg px-3 py-2"></select>
        </div>
      </div>

      <div class="grid md:grid-cols-4 gap-4 mt-4">
        <div>
          <label class="block text-sm font-medium">Paragraphs min</label>
          <input id="pmin" type="number" class="mt-1 w-full border rounded-lg px-3 py-2" value="2">
        </div>
        <div>
          <label class="block text-sm font-medium">Paragraphs max</label>
          <input id="pmax" type="number" class="mt-1 w-full border rounded-lg px-3 py-2" value="4">
        </div>
        <div>
          <label class="block text-sm font-medium">Z threshold</label>
          <input id="z" type="number" step="0.1" class="mt-1 w-full border rounded-lg px-3 py-2" value="1.5">
        </div>
        <div>
          <label class="block text-sm font-medium">Max events</label>
          <input id="me" type="number" class="mt-1 w-full border rounded-lg px-3 py-2" value="3">
        </div>
      </div>

      <div class="mt-6 flex items-center gap-3">
        <button id="go" class="bg-black text-white rounded-xl px-4 py-2">Generate</button>
        <button id="copy" class="border rounded-xl px-4 py-2">Copy markdown</button>
        <a id="raw" class="text-blue-600 underline" href="#" target="_blank">Open raw</a>
        <span id="msg" class="text-sm text-gray-600"></span>
      </div>
    </section>

    <section class="bg-white rounded-2xl shadow p-6">
      <h2 class="text-xl font-semibold mb-2">Market Context (markdown)</h2>
      <div class="grid md:grid-cols-2 gap-6">
        <div><pre id="out-md" class="whitespace-pre-wrap text-sm bg-gray-50 p-3 rounded-lg"></pre></div>
        <div><h3 class="font-medium mb-2">Preview</h3><article id="preview" class="prose max-w-none"></article></div>
      </div>
    </section>

    <section class="grid md:grid-cols-3 gap-6">
      <div class="bg-white rounded-2xl shadow p-6">
        <h3 class="text-lg font-semibold">Stats</h3>
        <pre id="out-stats" class="text-sm bg-gray-50 p-3 rounded-lg"></pre>
      </div>
      <div class="bg-white rounded-2xl shadow p-6 md:col-span-2">
        <h3 class="text-lg font-semibold">Detected Events</h3>
        <pre id="out-events" class="text-sm bg-gray-50 p-3 rounded-lg"></pre>
      </div>
    </section>

    <section class="bg-white rounded-2xl shadow p-6">
      <h3 class="text-lg font-semibold">Payload</h3>
      <pre id="out-payload" class="text-sm bg-gray-50 p-3 rounded-lg"></pre>
    </section>
  </main>
</div>

<script>
const el = (id)=>document.getElementById(id);

async function loadPresets(){
  const r = await fetch('/presets');
  const j = await r.json();

  // quarters
  const periodSel = el('period');
  j.quarters.forEach(q => {
    const o = document.createElement('option');
    o.textContent = q; o.value = q;
    periodSel.appendChild(o);
  });

  // benchmarks
  const benchSel = el('bench');
  benchSel.innerHTML = '';
  j.benchmarks.forEach(b => {
    const o = document.createElement('option');
    o.textContent = b; o.value = b;
    benchSel.appendChild(o);
  });

  // regions (preset)
  const regionSel = el('region');
  j.regions.forEach(p => {
    const o = document.createElement('option');
    o.textContent = p.label; o.value = p.id; o.dataset.benchmark = p.benchmark;
    regionSel.appendChild(o);
  });

  // set defaults
  regionSel.addEventListener('change', ()=>{
    const bm = regionSel.options[regionSel.selectedIndex].dataset.benchmark;
    if (bm) benchSel.value = bm;
  });
  // trigger once
  regionSel.dispatchEvent(new Event('change'));
}

async function generate(){
  const period = el('period').value;
  const regionOpt = el('region').options[el('region').selectedIndex];
  const market_region = regionOpt.textContent; // use the label from presets
  const benchmark = el('bench').value;
  const para_min = parseInt(el('pmin').value || "2");
  const para_max = parseInt(el('pmax').value || "4");
  const z_threshold = parseFloat(el('z').value || "1.5");
  const max_events = parseInt(el('me').value || "3");

  el('raw').href = "/market-context/raw?" + new URLSearchParams({period, market_region, benchmark}).toString();
  el('msg').textContent = "Working...";
  el('go').disabled = true;

  try {
    const r = await fetch('/market-context', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ period, market_region, benchmark, para_min, para_max, z_threshold, max_events })
    });
    const j = await r.json();
    if(!r.ok) throw new Error(j.detail || "Request failed");
    el('out-md').textContent = j.market_context_markdown || '';
    el('preview').innerHTML = marked.parse(j.market_context_markdown || '');
    el('out-stats').textContent = JSON.stringify(j.stats, null, 2);
    el('out-events').textContent = JSON.stringify(j.events, null, 2);
    el('out-payload').textContent = JSON.stringify(j.payload, null, 2);
    el('msg').textContent = "Done.";
  } catch(e) {
    el('msg').textContent = "Error: " + e.message;
  } finally {
    el('go').disabled = false;
  }
}

document.getElementById('go').addEventListener('click', (e)=>{ e.preventDefault(); generate(); });
document.getElementById('copy').addEventListener('click', async ()=>{
  await navigator.clipboard.writeText(document.getElementById('out-md').textContent || '');
  document.getElementById('msg').textContent = "Copied.";
});

loadPresets();
</script>
"""

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<meta charset="utf-8">
<title>Market Context Generator</title>
<style>
body { font-family: system-ui, Arial, sans-serif; max-width: 980px; margin: 2rem auto; }
label { display:block; margin-top: .5rem; font-weight: 600; }
input { width: 100%; padding: .5rem; font-size: 1rem; }
button { margin-top: 1rem; padding: .6rem 1rem; font-size: 1rem; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
pre { background: #f6f8fa; padding: 1rem; border-radius: 8px; overflow:auto; }
h2 { margin-top: 2rem; }
</style>
<h1>Market Context Generator</h1>
<form id="f">
<label>Period</label>
<input id="period" placeholder="Q2 2025" value="Q2 2025">
<label>Market Region</label>
<input id="region" placeholder="U.S. equities (small-cap)" value="U.S. equities (small-cap)">
<label>Benchmark</label>
<input id="bench" placeholder="Russell 2000" value="Russell 2000">
<div class="cols">
    <div>
    <label>Paragraphs min</label>
    <input id="pmin" type="number" value="2">
    </div>
    <div>
    <label>Paragraphs max</label>
    <input id="pmax" type="number" value="4">
    </div>
</div>
<div class="cols">
    <div>
    <label>Z threshold</label>
    <input id="z" type="number" step="0.1" value="1.5">
    </div>
    <div>
    <label>Max events</label>
    <input id="me" type="number" value="3">
    </div>
</div>
<button type="submit">Generate</button>
</form>

<h2>Market Context (markdown)</h2>
<pre id="out-md">(result prints here)</pre>
<h2>Stats</h2>
<pre id="out-stats"></pre>
<h2>Detected Events</h2>
<pre id="out-events"></pre>
<h2>Payload</h2>
<pre id="out-payload"></pre>

<script>
const f = document.getElementById('f');
async function run(e){
e.preventDefault();
const body = {
    period: document.getElementById('period').value,
    market_region: document.getElementById('region').value,
    benchmark: document.getElementById('bench').value,
    para_min: parseInt(document.getElementById('pmin').value || "2"),
    para_max: parseInt(document.getElementById('pmax').value || "4"),
    z_threshold: parseFloat(document.getElementById('z').value || "1.5"),
    max_events: parseInt(document.getElementById('me').value || "3")
};
const r = await fetch('/market-context', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
});
const j = await r.json();
document.getElementById('out-md').textContent = j.market_context_markdown || '(no text)';
document.getElementById('out-stats').textContent = JSON.stringify(j.stats, null, 2);
document.getElementById('out-events').textContent = JSON.stringify(j.events, null, 2);
document.getElementById('out-payload').textContent = JSON.stringify(j.payload, null, 2);
}
f.addEventListener('submit', run);
</script>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_service:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)