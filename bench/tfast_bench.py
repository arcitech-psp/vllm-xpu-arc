#!/usr/bin/env python3
"""Cookbook-style client benchmark for a local OpenAI-compatible server."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def request(url, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 1729,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.monotonic()
    end = time.monotonic()
    tokens = int(usage.get("completion_tokens", 0) or 0)
    boundary = first or end
    return {
        "ttft_s": boundary - t0,
        "post_first_tok_s": (tokens - 1) / max(end - boundary, 1e-9),
        "completion_tokens": tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
    }


def prompt_for(tokenizer, target, nonce):
    prefix = f"TFAST-COLD-{nonce:04d}-{target}-"
    unit = (" This is a neutral deterministic benchmark sentence. "
            "It contains no instructions and should not be continued as a task.")

    def token_count(content):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "get") and ids.get("input_ids") is not None:
            values = ids["input_ids"]
            return len(values[0]) if values and isinstance(values[0], (list, tuple)) else len(values)
        if ids and isinstance(ids[0], list):
            return len(ids[0])
        return len(ids)

    lo, hi = 0, 64
    while token_count(prefix + unit * hi) < target:
        hi *= 2
        if hi > 200000:
            raise RuntimeError("tokenizer did not return a usable 1-D chat token list")
    while lo < hi:
        mid = (lo + hi) // 2
        if token_count(prefix + unit * mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return prefix + unit * lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="tiel-coder")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = []
    for target in (512, 8192):
        for generation in (32, 128, 512):
            warm = request(args.url, args.model, prompt_for(tok, target, 0), generation)
            results = []
            for rep in range(args.reps):
                result = request(args.url, args.model, prompt_for(tok, target, rep + 1), generation)
                result["rep"] = rep + 1
                results.append(result)
            rates = [x["post_first_tok_s"] for x in results]
            row = {
                "target_prompt_tokens": target,
                "generation_tokens": generation,
                "warmup": warm,
                "runs": results,
                "median_post_first_tok_s": statistics.median(rates),
                "mean_post_first_tok_s": statistics.mean(rates),
                "actual_prompt_tokens": [x.get("prompt_tokens") for x in results],
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
