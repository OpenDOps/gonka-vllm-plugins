#!/usr/bin/env python3
"""Decode-PoC rate and artifact dump for MetaX C550 phase 0.5.

run_pow_generation.py never sets params.scheme, so it only measures prefill
(PoC v2). This client posts scheme=decode and max_tokens to
/api/v1/pow/init/generate.

Point it at the decode image (Dockerfile.vllm-decode-metax), served with
--enforce-eager, in its own container. The prefill spike has no decode seams.

A usable artifact has len(k_points_steps) >= max_tokens + 1 (one prefill step
plus max_tokens decode steps). Shorter chains are saved and counted, and they
are not the rate gate.

Writes under --out:
  input.json                 replay inputs (seeds, scheme, max_tokens)
  sweep.json                 nonces/min per batch (warmup discarded)
  collect/nonces.jsonl       one callback artifact per line
  collect/nonces.json        dump for archive / later compare
  collect/rate.json          collect-pass nonces/min

Requires: requests
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    import requests
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install requests") from e

NVIDIA_0731_BLOCK_HASH = (
    "bce35dd795a5a622a9092ec1234eecbc157f4d03bdf117527b8a877bec6489b3"
)
NVIDIA_0731_PUBLIC_KEY = (
    "4dfe8269e29777a44cd5ec412aabed42afd9a410cef2dbefe6b5cd6b983d9d5c"
)
GENERIC_BLOCK_HASH = "artifact_collection_block_v1"
GENERIC_PUBLIC_KEY = "artifact_collection_pk_v1"

# 0.5 spike pin. A full trajectory is this many decode steps after the prefill.
DEFAULT_MAX_TOKENS = 8


class CallbackState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.artifacts: List[dict] = []
        self.jsonl_fp = None

    def set_jsonl(self, path: Optional[str]) -> None:
        if self.jsonl_fp is not None:
            self.jsonl_fp.close()
            self.jsonl_fp = None
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.jsonl_fp = open(path, "w", encoding="utf-8")

    def clear(self) -> None:
        with self.lock:
            self.artifacts.clear()

    def add_batch(self, data: dict) -> int:
        arts = data.get("artifacts") or data.get("nonces") or []
        n = 0
        with self.lock:
            for a in arts:
                if not isinstance(a, dict) or "nonce" not in a:
                    continue
                self.artifacts.append(a)
                n += 1
                if self.jsonl_fp is not None:
                    self.jsonl_fp.write(json.dumps(a, separators=(",", ":")) + "\n")
                    self.jsonl_fp.flush()
        return n

    def counts(self) -> int:
        with self.lock:
            return len(self.artifacts)

    def snapshot(self) -> List[dict]:
        with self.lock:
            return list(self.artifacts)

    def close(self) -> None:
        if self.jsonl_fp is not None:
            self.jsonl_fp.close()
            self.jsonl_fp = None


STATE = CallbackState()


def _make_handler() -> type:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/health":
                self._json({"status": "OK"})
                return
            if path == "/stats":
                self._json({"total_nonces": STATE.counts()})
                return
            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            try:
                data = json.loads(raw.decode() or "{}")
            except json.JSONDecodeError:
                self._json({"error": "invalid json"}, 400)
                return
            if path in ("/generated", "/validated", "/"):
                added = STATE.add_batch(data)
                self._json({"message": "OK", "accepted": added})
                return
            if path == "/clear":
                STATE.clear()
                self._json({"message": "Cleared"})
                return
            self._json({"error": "not found"}, 404)

    return Handler


def start_callback_server(bind: str, port: int) -> HTTPServer:
    httpd = HTTPServer((bind, port), _make_handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
            if r.status_code == 200:
                return httpd
        except requests.RequestException:
            time.sleep(0.2)
    raise SystemExit(f"callback server did not come up on {bind}:{port}")


def pow_url(base: str, prefix: str, path: str) -> str:
    return f"{base.rstrip('/')}{prefix.rstrip('/')}/{path.lstrip('/')}"


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def resolve_seeds(seed_set: str, block_hash: str, public_key: str) -> Tuple[str, str]:
    if seed_set == "nvidia-0731":
        bh, pk = NVIDIA_0731_BLOCK_HASH, NVIDIA_0731_PUBLIC_KEY
    elif seed_set == "generic":
        bh, pk = GENERIC_BLOCK_HASH, GENERIC_PUBLIC_KEY
    else:
        raise SystemExit(f"unknown --seed-set {seed_set}")
    return block_hash or bh, public_key or pk


def trajectory_len(artifact: dict) -> int:
    steps = artifact.get("k_points_steps") or []
    return len(steps) if isinstance(steps, list) else 0


def usable(artifact: dict, max_tokens: int) -> bool:
    return trajectory_len(artifact) >= max_tokens + 1


def pow_stop(base: str, prefix: str) -> None:
    try:
        requests.post(pow_url(base, prefix, "stop"), json={}, timeout=15)
    except requests.RequestException:
        pass


def pow_status(base: str, prefix: str) -> dict:
    r = requests.get(pow_url(base, prefix, "status"), timeout=15)
    r.raise_for_status()
    return r.json()


def processed_from_status(st: dict) -> int:
    stats = st.get("stats") or {}
    if not stats and isinstance(st.get("backends"), list) and st["backends"]:
        stats = (st["backends"][0] or {}).get("stats") or {}
    return int(stats.get("total_processed") or 0)


def check_engine(base: str) -> None:
    errors = []
    for path in ("/health", "/v1/models"):
        try:
            r = requests.get(f"{base.rstrip('/')}{path}", timeout=10)
            if r.status_code < 500:
                print(f"engine {path}: HTTP {r.status_code}", flush=True)
                return
            errors.append(f"{path} HTTP {r.status_code}")
        except requests.RequestException as e:
            errors.append(f"{path}: {e}")
    raise SystemExit(f"engine not reachable at {base}: {'; '.join(errors)}")


def decode_params(args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "seq_len": args.seq_len,
        "k_dim": args.k_dim,
        "scheme": "decode",
        "decode": True,
        "max_tokens": args.max_tokens,
    }


def pow_init(args: argparse.Namespace, batch_size: int, callback_url: str, block_hash: str, public_key: str) -> dict:
    payload = {
        "block_hash": block_hash,
        "block_height": args.block_height,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "group_id": 0,
        "n_groups": 1,
        "batch_size": batch_size,
        "url": callback_url,
        "params": decode_params(args),
    }
    r = requests.post(
        pow_url(args.url, args.pow_prefix, "init/generate"),
        json=payload,
        timeout=60,
    )
    if r.status_code >= 400:
        raise SystemExit(
            f"init/generate HTTP {r.status_code}: {r.text[:800]}\n"
            "Decode needs the overlay image (S2D). The prefill spike accepts "
            "scheme=decode and then emits no trajectory."
        )
    return r.json()


def sweep(
    args: argparse.Namespace,
    callback_url: str,
    block_hash: str,
    public_key: str,
) -> List[dict]:
    rows = []
    sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    if not sizes:
        raise SystemExit("--batch-sizes is empty")
    print(
        f"\ndecode sweep  batch={sizes}  warmup={args.warmup_s:.0f}s  "
        f"window={args.duration_s:.0f}s  max_tokens={args.max_tokens}",
        flush=True,
    )
    for batch_size in sizes:
        print(f"\nbatch_size={batch_size}", flush=True)
        pow_stop(args.url, args.pow_prefix)
        time.sleep(1)
        STATE.clear()
        print(pow_init(args, batch_size, callback_url, block_hash, public_key), flush=True)
        time.sleep(args.warmup_s)
        STATE.clear()
        try:
            before = processed_from_status(pow_status(args.url, args.pow_prefix))
        except requests.RequestException:
            before = 0
        t0 = time.time()
        deadline = t0 + args.duration_s
        while time.time() < deadline:
            time.sleep(min(10.0, deadline - time.time()))
            elapsed = time.time() - t0
            n = STATE.counts()
            print(f"  [{elapsed:4.0f}s] callbacks={n}", flush=True)
        elapsed = time.time() - t0
        time.sleep(2)
        arts = STATE.snapshot()
        n = len(arts)
        n_ok = sum(1 for a in arts if usable(a, args.max_tokens))
        try:
            after = processed_from_status(pow_status(args.url, args.pow_prefix))
        except requests.RequestException:
            after = before
        pow_stop(args.url, args.pow_prefix)
        row = {
            "batch_size": batch_size,
            "window_sec": elapsed,
            "callback_nonces": n,
            "usable_nonces": n_ok,
            "short_nonces": n - n_ok,
            "callback_nonces_per_min": (n / elapsed * 60.0) if elapsed else 0.0,
            "usable_nonces_per_min": (n_ok / elapsed * 60.0) if elapsed else 0.0,
            "engine_nonces_window": max(0, after - before),
            "engine_nonces_per_min": (max(0, after - before) / elapsed * 60.0) if elapsed else 0.0,
            "max_tokens": args.max_tokens,
            "scheme": "decode",
        }
        rows.append(row)
        print(
            f"  callbacks {n} ({row['callback_nonces_per_min']:.0f}/min)  "
            f"usable {n_ok} ({row['usable_nonces_per_min']:.0f}/min)  "
            f"engine {row['engine_nonces_window']} ({row['engine_nonces_per_min']:.0f}/min)",
            flush=True,
        )
    return rows


def collect(
    args: argparse.Namespace,
    callback_url: str,
    block_hash: str,
    public_key: str,
) -> dict:
    collect_dir = os.path.join(args.out, "collect")
    os.makedirs(collect_dir, exist_ok=True)
    STATE.set_jsonl(os.path.join(collect_dir, "nonces.jsonl"))
    pow_stop(args.url, args.pow_prefix)
    time.sleep(1)
    STATE.clear()
    print(
        f"\ndecode collect  batch={args.collect_batch_size}  "
        f"target={args.nonces}  max_tokens={args.max_tokens}",
        flush=True,
    )
    print(pow_init(args, args.collect_batch_size, callback_url, block_hash, public_key), flush=True)
    t0 = time.time()
    timed_out = False
    last_print = -1
    while True:
        n = STATE.counts()
        elapsed = time.time() - t0
        if n >= args.nonces:
            break
        if elapsed > args.timeout_s:
            timed_out = True
            break
        if int(elapsed) != last_print and int(elapsed) % 5 == 0:
            last_print = int(elapsed)
            rate = (n / elapsed * 60.0) if elapsed else 0.0
            print(f"  callbacks={n}/{args.nonces}  {rate:.0f} n/min  {elapsed:.0f}s", flush=True)
        time.sleep(0.5)
    elapsed = time.time() - t0
    pow_stop(args.url, args.pow_prefix)
    time.sleep(2)
    arts = STATE.snapshot()[: args.nonces]
    n = len(arts)
    n_ok = sum(1 for a in arts if usable(a, args.max_tokens))
    rate = (n / elapsed * 60.0) if elapsed else 0.0
    ok_rate = (n_ok / elapsed * 60.0) if elapsed else 0.0
    need = args.max_tokens + 1
    dump = {
        "scheme": "decode",
        "max_tokens": args.max_tokens,
        "trajectory_steps": need,
        "block_hash": block_hash,
        "public_key": public_key,
        "block_height": args.block_height,
        "seq_len": args.seq_len,
        "k_dim": args.k_dim,
        "model": args.model,
        "batch_size": args.collect_batch_size,
        "total_nonces": n,
        "usable_nonces": n_ok,
        "short_nonces": n - n_ok,
        "artifacts": arts,
        "generation_time_sec": elapsed,
        "nonces_per_min": rate,
        "usable_nonces_per_min": ok_rate,
        "timed_out": timed_out,
    }
    write_json(os.path.join(collect_dir, "nonces.json"), dump)
    write_json(
        os.path.join(collect_dir, "rate.json"),
        {
            "scheme": "decode",
            "max_tokens": args.max_tokens,
            "trajectory_steps": need,
            "callback_nonces": n,
            "usable_nonces": n_ok,
            "short_nonces": n - n_ok,
            "callback_nonces_per_min": rate,
            "usable_nonces_per_min": ok_rate,
            "wall_sec": elapsed,
            "batch_size": args.collect_batch_size,
            "timed_out": timed_out,
        },
    )
    STATE.close()
    print(
        f"\n{'TIMEOUT' if timed_out else 'DONE'}: {n} artifacts, {n_ok} usable "
        f"(k_points_steps >= {need}) in {elapsed:.1f}s → {ok_rate:.0f} usable n/min",
        flush=True,
    )
    print(f"  {collect_dir}/nonces.jsonl", flush=True)
    print(f"  {collect_dir}/nonces.json", flush=True)
    print(f"  {collect_dir}/rate.json", flush=True)
    return dump


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="deepseek-ai/deepseek-llm-67b-chat")
    p.add_argument("--out", required=True)
    p.add_argument("--pow-prefix", default="/api/v1/pow")
    p.add_argument("--callback-bind", default="0.0.0.0")
    p.add_argument("--callback-port", type=int, default=9997)
    p.add_argument("--callback-url", default="")
    p.add_argument("--seed-set", choices=("nvidia-0731", "generic"), default="nvidia-0731")
    p.add_argument("--block-hash", default="")
    p.add_argument("--public-key", default="")
    p.add_argument("--block-height", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--k-dim", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--batch-sizes", default="8,16,32")
    p.add_argument("--warmup-s", type=float, default=5.0)
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--collect-batch-size", type=int, default=8)
    p.add_argument("--nonces", type=int, default=256)
    p.add_argument("--timeout-s", type=int, default=1800)
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--skip-collect", action="store_true")
    args = p.parse_args()

    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be >= 1; 0 is the prefill scheme")
    if args.skip_sweep and args.skip_collect:
        raise SystemExit("nothing to do: both --skip-sweep and --skip-collect")

    block_hash, public_key = resolve_seeds(args.seed_set, args.block_hash, args.public_key)
    os.makedirs(args.out, exist_ok=True)
    advertise = args.callback_url or f"http://127.0.0.1:{args.callback_port}"
    record = {
        "scheme": "decode",
        "decode": True,
        "max_tokens": args.max_tokens,
        "trajectory_steps": args.max_tokens + 1,
        "model": args.model,
        "engine_url": args.url,
        "pow_prefix": args.pow_prefix,
        "block_hash": block_hash,
        "public_key": public_key,
        "block_height": args.block_height,
        "seq_len": args.seq_len,
        "k_dim": args.k_dim,
        "node_id": 0,
        "node_count": 1,
        "seed_set": args.seed_set,
        "batch_sizes": args.batch_sizes,
        "collect_batch_size": args.collect_batch_size,
        "target_nonces": args.nonces,
        "callback_url": advertise,
        "note": (
            "Replay this exact (scheme=decode, max_tokens, block_hash, public_key, "
            "seq_len, k_dim, nonce ids 0,1,2,…) . Do not compare these artifacts "
            "to a prefill / PoC v2 dump."
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(os.path.join(args.out, "input.json"), record)
    print(f"logged decode input → {args.out}/input.json", flush=True)
    print(f"  scheme=decode max_tokens={args.max_tokens} trajectory_steps={args.max_tokens + 1}", flush=True)
    print(f"  block_hash={block_hash}", flush=True)
    print(f"  public_key={public_key}", flush=True)

    check_engine(args.url)
    httpd = start_callback_server(args.callback_bind, args.callback_port)
    print(f"callback {args.callback_bind}:{args.callback_port} advertise={advertise}", flush=True)

    rows: List[dict] = []
    dump: Dict[str, Any] = {}
    try:
        if not args.skip_sweep:
            rows = sweep(args, advertise, block_hash, public_key)
            best = max(rows, key=lambda r: r["usable_nonces_per_min"]) if rows else {}
            write_json(os.path.join(args.out, "sweep.json"), {"scheme": "decode", "rows": rows, "best": best})
            print("\nbatch  usable n/min  callback n/min  engine n/min", flush=True)
            for r in rows:
                print(
                    f"  {r['batch_size']:>4}  {r['usable_nonces_per_min']:13.0f}  "
                    f"{r['callback_nonces_per_min']:14.0f}  {r['engine_nonces_per_min']:12.0f}",
                    flush=True,
                )
            print(f"wrote {args.out}/sweep.json", flush=True)
        if not args.skip_collect:
            dump = collect(args, advertise, block_hash, public_key)
    finally:
        STATE.close()
        httpd.shutdown()

    sweep_ok = args.skip_sweep or any(r["usable_nonces"] or r["callback_nonces"] for r in rows)
    collect_ok = args.skip_collect or int(dump.get("usable_nonces") or 0) > 0
    if not sweep_ok:
        print("FAIL: decode sweep produced zero callback nonces", flush=True)
        return 1
    if not collect_ok:
        print(
            "FAIL: decode collect has zero usable artifacts "
            f"(need k_points_steps >= {args.max_tokens + 1})",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
