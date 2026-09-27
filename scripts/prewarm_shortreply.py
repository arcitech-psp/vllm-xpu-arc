"""Run one discardable warmup request for the XPU short-reply path."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

from tfast_bench import prompt_for
from transformers import AutoTokenizer


def main() -> None:
    if os.getenv("TIEL_K3_PREWARM") != "1":
        raise SystemExit("set TIEL_K3_PREWARM=1 to run the discardable warmup")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="tiel-coder")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt_for(tokenizer, args.prompt_tokens, 0)}],
        "temperature": 0.0,
        "seed": 1729,
        "max_tokens": args.max_tokens,
        "min_tokens": args.max_tokens,
        "ignore_eos": True,
        "stream": False,
    }
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        response.read()
    print("Tiel XPU prewarm complete; response discarded")


if __name__ == "__main__":
    main()
