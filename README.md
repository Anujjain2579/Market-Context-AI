# Setup
## Sample .env
ALPHAVANTAGE_API_KEY="YOUR_KEY"

OPENAI_API_KEY="YOUR_KEY"

OPENAI_MODEL="gpt-4.1-mini"

## Libraries
Recommended

conda create -n mcg python=3.11 "pydantic=2.8.*" "fastapi=0.115.*" "uvicorn=0.30.*" "pandas=2.2.*" "numpy=1.26.*" "python-dateutil=2.9.*" "requests=2.32.*" -c conda-forge -y

pip install "openai>=1.30,<2"

OR

pip install -r requirements.txt

## Project files
commentary_schema.py

ingest_normalize.py

generate_market_context.py

api_service.py

## Run the API
python api_service.py

## Go to app

http://localhost:8000/app

### You may also download results by command like
python cli_generate.py --period "Q2 2025" --market_region "U.S. equities (small-cap)" --benchmark "Russell 2000"


# The Problem I Tackled (and Why)

I set out to automate the "Market Context" section of monthly or quarterly portfolio commentaries. The idea was to pull macro and benchmark data, normalize it into a strict schema, and use a large language model (LLM) to write a 150–250 word section that’s region-specific, strictly factual (no made-up numbers), and avoids any fund-specific or attribution terms. This helps remove a tedious bottleneck, and also makes the market commentary more consistent and compliant.

# Architecture and Design Decisions

- **Schema-first Gatekeeper (Pydantic v2):** I used a schema with strict validators for required fields, numeric bounds, special placeholders (like “not provided”), and language filters to make sure certain words like “we”, “our”, and portfolio-specific jargon didn’t sneak in.
- **Data Ingestion Layer:** There are Alpha Vantage adapters set up to pull CPI, Fed Funds, 10Y UST, GDP (YoY), unemployment rates, FX (region-specific pairs), commodities; and for benchmarks, ETF proxies (e.g., Russell 2000 → IWM).
- **Event Detection:** Big moves in benchmarks get flagged (anything where the daily return's absolute value is at least 1.5 standard deviations). The LLM is prompted to give a neutral, factual mini-narrative about notable events.
- **Region Salience Controls:** The prompt is injected with guards and hints so the LLM puts extra focus on the requested market region (e.g., for Emerging Markets, it'll emphasize USD/EM-FX and commodity effects).
- **Hard Post-Checks & Retrying:** After generation, the output is checked for things like headings, word/paragraph counts (now 2–4), and forbidden words. If it fails, the system critiques and fixes it automatically.
- **Service Boundary:** The FastAPI endpoint `POST /market-context` spits out the markdown commentary, flagged events, and the exact, normalized payload.

# Key Assumptions & Trade-Offs

- **ETF proxies for benchmarks:** Using tickers like IWM or SPY keeps integration simple, but isn’t a perfect match for the true index TR. Convenience vs. precision.
- **Alpha Vantage Coverage/Latency:** Good for light macro and daily series, but their rate limits mean you need caching/backoff for scale.
- **FX relevance by region:** I used basic heuristics, like mapping Emerging Markets to USD/CNY, USD/INR, USD/BRL. Simpler, but might miss edge cases.
- **Event Detection Threshold (1.5σ):** This threshold aims to pick up real moves without flagging random noise. It can be tuned.
- **LLM Summarization Constraints:** To avoid the model making stuff up, I only feed in approved data and block any invented numbers — but it’s still prompt/check-dependent.

# What I’d Add With More Time (Future Work)

- **Geopolitical Stream:** Scrape and parse White House presidential actions and bring in BlackRock’s Geopolitical Risk dashboard labels. The goal is to surface dated, region-relevant policy events as `policy_geopolitics`.
- **Caching and Quotas:** Add local/Redis caches for Alpha Vantage, and save daily snapshots for reproducibility.
- **Source Fallbacks:** Integrate Databento for more benchmarks and sector returns, plus FRED/ECB/BOE for more global macro data.
- **Observability & QA:** Add structured logging, error/latency dashboards, golden prompts, and unit tests for validators and event detectors.
- **UI & Workflow:** Build a lightweight web UI so users can compare drafts across regions or time periods and accept or edit the generated sections.

# Prompts Used during development

## Prompt 1
My task is to design and implement a solution to streamline or automate the creation of portfolio commentaries for the latest month or quarter similar to the previous commentaries. For this assignment, focus solely on the Market Context section. Attached are 3 examples from US Equity Fund, All Cap Core portfolio and Genesis fund.

**Below is my thoughts on workflow to do this using data provider APIs (Alphavantage, Databento) and OpenAI APIs** 

Frameworks we can use are Python, FastAPI and Flask 

Inputs from user would be Period (Q or custom date range) Market region (so we know which benchmark/ETF to map) 
### Workflow
Ingest & normalize

Pull macro series (CPI, policy rate, 10y yield, FX, HY spreads), benchmark total return, sector returns, factor/style stats, notable events.

Map each feed into the JSON schema below.

Validate Required:

period, benchmark, and at least one of benchmark_return_total_pct or a qualitative market move. 

Numeric sanity checks (e.g., return is between -100 and +100). Strip fund-specific tokens from inputs (reject sentences containing: \b(we|our|the fund|portfolio|overweight|underweight|selection|allocation)\b). 

Assemble prompt 

System rules (no fund mentions, word count, no fabrication) + the JSON payload. 

Generate (LLM call) 

Output must start with ## Market Context.

150–250 words; 2–3 paragraphs.

System / Developer message (rules)

You are an investment commentator. Write the Market Context for the specified period. Use only the data in INPUT. Do not mention the fund, portfolio, “we/our,” positioning, allocation, or stock selection. If a data point is missing, write “not provided.” Do not invent numbers. Keep tone factual, concise, and client-friendly. Deliverable Return one section titled “## Market Context”, 150–250 words, 2–3 paragraphs: Macro & policy: inflation, rates, growth, FX/credit, notable policy/geopolitics. Market behavior: index return, volatility, sector trends, factor/style rotation, market breadth/concentration. 

(Optional) 3rd paragraph only if INPUT includes a notable single-day move or event. Style rules No fund names or attribution terms (allocation/selection/overweight/underweight). Define acronyms once. Numbers must match INPUT exactly. 

Style rules

No fund names or attribution terms (allocation/selection/overweight/underweight).

Define acronyms once.

Numbers must match INPUT exactly.

{ "period": "Q2 2025", "market_region": "U.S. equities (small-cap)", "benchmark": "Russell 2000", "benchmark_return_total_pct": -1.79, "index_level_events": [ {"date": "2025-04-08", "description": "Tariff pause announced", "one_day_move_pct": 9.5} ], "macro": { "inflation": "Headline CPI 3.1% YoY, easing", "policy_rate": "Fed on hold; markets price 2–3 cuts in 2025", "yields": "10y UST ~4.22%", "growth": "EPS revisions trimmed; GDP resilient", "employment": "not provided", "fx": "USD weaker vs. basket", "credit": "High-yield spreads narrowed", "commodities": "not provided", "policy_geopolitics": ["Tariffs narrative volatility", "Geopolitical flare-ups"] }, "breadth_concentration": { "description": "Risk-on, high beta outperformed; market narrowness increased", "top_names_contribution_pct": "not provided", "advance_decline": "not provided" }, "sector_themes": [ {"sector": "Information Technology", "theme": "AI-related demand; strong rally"}, {"sector": "Industrials", "theme": "mixed; machinery resilient"}, {"sector": "Health Care", "theme": "Biotech weakness"} ], "style_rotation": { "value_vs_growth": "Growth outperformed", "size": "Large > Small", "quality": "Low-quality beta led", "volatility": "Elevated headline-driven swings; VIX level not provided" }, "disclaimers": "No forecasts; monitoring policy path and earnings dispersion" } 

I would like develop code for above properly in steps. Limit comments to only where required. Do not have === or other characters for printing. If needed to print, just print \n in next lines.
 
Give me code for 1st step we should take

## Prompt 2

Key Modifications
 
I made Validator Signatures:

Updated all field and model validators to use the correct function signatures for Pydantic v2 (e.g., model_validator(mode='wrap') expects (cls, self, info); model_validator(mode='before') expects (cls, values)). 

Validator Return Values: Ensured top-level model validators return self (for 'wrap' mode) or values (for 'before' mode) to avoid Pydantic validation warnings and attribute errors. 

Serialization Methods: Replaced all deprecated .dict() and .json() calls with .model_dump() and .model_dump_json(), or used Python’s json.dumps(model.model_dump(), indent=2) for pretty-printing. 

Field and Model Access: When using model-level validators, accessed attributes via self instead of values when appropriate. 

**If above modifications sounds good, proceed to step 2 code**

## Prompt 3
proceed to step 3 code

## Prompt 4

I allowed for paragraphs to be in 2-4 in validation (to be flexible)
I would like Notable events not to be input. It should be Either based on data or through an API. Let's try asking to ChatGPT itself. 

Detection: From your benchmark daily returns (AV IWM adj close or a DB-built index), flag outsized one-day moves (e.g., |return| ≥ 1.5σ). 

Description: Use OpenAI to convert “2025-04-08: +3.2% R2K” into a short narrative (no fund mentions, neutral tone). You can optionally pass in a small curated calendar (CPI, FOMC, payrolls) you already grabbed from AV macro endpoints to help the model attribute correctly. 

On running above code, My output was mostly similar for US equities small cap vs emerging markets equity. Although we expect some overlap for big news, I would like to have context similar to market region being discussed

## Prompt 5

Great, now proceed to next step code

## Prompt 6

For macroeconomic parts, I mostly get similar output (...) for Equity, Emerging Markets, Genesis fund. 

We can actually get other available macroeconomic data from alpha vantage https://www.alphavantage.co/documentation/
https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey=demo

US Dollar to Japanese Yen:
https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=JPY&apikey=demo

Bitcoin to Euro:
https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=BTC&to_currency=EUR&apikey=demo

https://www.alphavantage.co/query?function=ALL_COMMODITIES&interval=monthly&apikey=demo

For geopolitical events, we can use proxy of Alpha intelligence on alphavantage - Like Querying news articles that simultaneously mention the Coinbase stock (COIN), Bitcoin (CRYPTO:BTC), and US Dollar (FOREX:USD) and are published on or after 2022-04-10, 1:30am UTC.

https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=COIN,CRYPTO:BTC,FOREX:USD&time_from=20220410T0130&limit=1000&apikey=demo
https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey=demo

Give code to modify. We somehow need to make OpenAI LLM choose which data is most relevant to for a given market region query. Are we already doing this? I'm ok calling all data but when writing market context, we need most relevant ones.

## Prompt 7

Generating Summary 
