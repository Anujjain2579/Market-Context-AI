# cli_generate.py
import os, argparse, json
from ingest_normalize import build_market_context_payload, daily_prices_for_benchmark, detect_outsized_moves_from_prices, curated_macro_calendar
from generate_market_context import generate_market_context

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--period", required=True)
    p.add_argument("--market_region", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--outdir", default="out")
    p.add_argument("--z", type=float, default=1.5)
    p.add_argument("--max_events", type=int, default=3)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    payload = build_market_context_payload(args.period, args.market_region, args.benchmark)
    px = daily_prices_for_benchmark(payload.period, payload.benchmark)
    events = detect_outsized_moves_from_prices(px, z=args.z, max_events=args.max_events)
    cal = curated_macro_calendar(args.period)
    md = generate_market_context(payload, events, cal, para_min=2, para_max=4)

    fname = f"market_context_{args.period.replace(' ','_')}_{args.benchmark.replace(' ','_')}.md"
    path = os.path.join(args.outdir, fname)
    with open(path, "w") as f:
        f.write(md + "\n")
    print(path)

if __name__ == "__main__":
    main()
