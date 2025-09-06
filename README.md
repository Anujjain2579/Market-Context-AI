# Setup
## Env vars
export ALPHAVANTAGE_API_KEY="YOUR_KEY"

export OPENAI_API_KEY="YOUR_KEY"

export OPENAI_MODEL="gpt-4.1-mini"

## Libraries
pip install -r requirements.txt

or

conda create -n mcg python=3.11 "pydantic=2.8.*" "fastapi=0.115.*" "uvicorn=0.30.*" "pandas=2.2.*" "numpy=1.26.*" "python-dateutil=2.9.*" "requests=2.32.*" -c conda-forge -y

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
