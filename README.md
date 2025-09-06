# Setup
## Env vars
export ALPHAVANTAGE_API_KEY="YOUR_KEY"
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4.1-mini"

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