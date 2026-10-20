#!/usr/bin/env python3
"""MetaX / NVIDIA PoC nonce bench: callback collect, rate, L2 compare.

Subcommands
  collect   Start a callback HTTP server, POST /api/v1/pow/init/generate with
            url=<callback>, count incoming artifacts, log rate + dump JSON.
  sweep     Same callback path; measure nonces/min per batch_size (warmup + window).
  compare   L2 + binomial fraud test of two nonce dumps (C550 vs H100/H200).

NVIDIA DeepSeek-V4-Flash-0731 dumps in kaitakuai/experiments used:
  block_hash  bce35dd795a5a622a9092ec1234eecbc157f4d03bdf117527b8a877bec6489b3
  public_key  4dfe8269e29777a44cd5ec412aabed42afd9a410cef2dbefe6b5cd6b983d9d5c
  seq_len=1024  k_dim=12  nonce 0,1,2,…

Re-use those seeds (--seed-set nvidia-0731, the default) or L2 is meaningless.

Requires: requests  (collect/sweep)
          numpy     (compare)
          scipy     (compare fraud p-value; optional)

Examples
  python3 poc_nonce_bench.py collect \\
    --url http://127.0.0.1:8000 --model deepseek-ai/DeepSeek-V4-Flash-0731 \\
    --out ./artifacts --nonces 256 --batch-size 8

  python3 poc_nonce_bench.py sweep \\
    --url http://127.0.0.1:8000 --model deepseek-ai/DeepSeek-V4-Flash-0731 \\
    --out ./artifacts --batch-sizes 8,16,32

  python3 poc_nonce_bench.py compare \\
    --ours ./artifacts/nonces.json \\
    --ref https://raw.githubusercontent.com/kaitakuai/experiments/main/2026-08/deepseek-v4-flash-0731-2xh200/artifacts/nonces_v2_off_s1.json \\
    --out ./artifacts/l2_vs_h200.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
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

# Same seeds as 2026-08 DeepSeek-V4-Flash-0731 H100/H200 dumps.
NVIDIA_0731_BLOCK_HASH = (
    "bce35dd795a5a622a9092ec1234eecbc157f4d03bdf117527b8a877bec6489b3"
)
NVIDIA_0731_PUBLIC_KEY = (
    "4dfe8269e29777a44cd5ec412aabed42afd9a410cef2dbefe6b5cd6b983d9d5c"
)
GENERIC_BLOCK_HASH = "artifact_collection_block_v1"
GENERIC_PUBLIC_KEY = "artifact_collection_pk_v1"

NVIDIA_0731_H200 = (
    "https://raw.githubusercontent.com/kaitakuai/experiments/main/"
    "2026-08/deepseek-v4-flash-0731-2xh200/artifacts/nonces_v2_off_s1.json"
)
NVIDIA_0731_H100 = (
    "https://raw.githubusercontent.com/kaitakuai/experiments/main/"
    "2026-08/deepseek-v4-flash-0731-dspark-4xh100/artifacts/nonces_dspark_off_s1.json"
)

# V4 cross-hardware numbers in experiments used 0.40 / 0.10 (H100 vs H200 floor ~0.17).
V4_DIST_THRESHOLD = 0.40
V4_P_MISMATCH = 0.10
V4_FRAUD_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Callback receiver
# ---------------------------------------------------------------------------


class CallbackState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.batches: List[dict] = []
        self.artifacts: List[dict] = []
        self.jsonl_path: Optional[str] = None
        self.jsonl_fp = None

    def set_jsonl(self, path: str) -> None:
        self.jsonl_path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.jsonl_fp = open(path, "w", encoding="utf-8")

    def clear(self) -> None:
        with self.lock:
            self.batches.clear()
            self.artifacts.clear()

    def add_batch(self, data: dict) -> int:
        arts = data.get("artifacts") or data.get("nonces") or []
        n = 0
        with self.lock:
            self.batches.append(data)
            for a in arts:
                if not isinstance(a, dict):
                    continue
                self.artifacts.append(a)
                n += 1
                if self.jsonl_fp is not None:
                    self.jsonl_fp.write(json.dumps(a, separators=(",", ":")) + "\n")
                    self.jsonl_fp.flush()
        return n

    def counts(self) -> Tuple[int, int]:
        with self.lock:
            return len(self.artifacts), len(self.batches)

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
                n, b = STATE.counts()
                self._json({"total_nonces": n, "batch_count": b})
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
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
            if r.status_code == 200:
                return httpd
        except requests.RequestException:
            time.sleep(0.2)
    raise RuntimeError(f"callback server did not come up on {bind}:{port}")


# ---------------------------------------------------------------------------
# PoC HTTP helpers
# ---------------------------------------------------------------------------


def pow_url(base: str, prefix: str, path: str) -> str:
    return f"{base.rstrip('/')}{prefix.rstrip('/')}/{path.lstrip('/')}"


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


def pow_stop(base: str, prefix: str) -> None:
    try:
        requests.post(pow_url(base, prefix, "stop"), json={}, timeout=15)
    except requests.RequestException:
        pass


def pow_status(base: str, prefix: str) -> dict:
    r = requests.get(pow_url(base, prefix, "status"), timeout=15)
    r.raise_for_status()
    return r.json()


def processed_from_status(st: dict) -> Tuple[int, float]:
    stats = st.get("stats") or {}
    if not stats and isinstance(st.get("backends"), list) and st["backends"]:
        stats = (st["backends"][0] or {}).get("stats") or {}
    total = int(stats.get("total_processed") or 0)
    nps = float(stats.get("nonces_per_second") or 0.0)
    return total, nps


def pow_init(
    base: str,
    prefix: str,
    *,
    model: str,
    batch_size: int,
    callback_url: str,
    block_hash: str,
    public_key: str,
    block_height: int,
    seq_len: int,
    k_dim: int,
    node_id: int,
    node_count: int,
) -> dict:
    payload = {
        "block_hash": block_hash,
        "block_height": block_height,
        "public_key": public_key,
        "node_id": node_id,
        "node_count": node_count,
        "group_id": 0,
        "n_groups": 1,
        "batch_size": batch_size,
        "url": callback_url,
        "params": {"model": model, "seq_len": seq_len, "k_dim": k_dim},
    }
    r = requests.post(pow_url(base, prefix, "init/generate"), json=payload, timeout=60)
    if r.status_code >= 400:
        raise SystemExit(
            f"init/generate HTTP {r.status_code}: {r.text[:800]}\n"
            "PoC router missing? 0.5 needs gonka-poc (or in-tree vllm.poc) mounted."
        )
    return r.json()


def resolve_seeds(seed_set: str, block_hash: Optional[str], public_key: Optional[str]) -> Tuple[str, str]:
    if seed_set == "nvidia-0731":
        bh, pk = NVIDIA_0731_BLOCK_HASH, NVIDIA_0731_PUBLIC_KEY
    elif seed_set == "generic":
        bh, pk = GENERIC_BLOCK_HASH, GENERIC_PUBLIC_KEY
    else:
        raise SystemExit(f"unknown --seed-set {seed_set}")
    return block_hash or bh, public_key or pk


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def input_record(args: argparse.Namespace, block_hash: str, public_key: str, extra: dict) -> dict:
    rec = {
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
        "group_id": 0,
        "n_groups": 1,
        "seed_set": args.seed_set,
        "note": (
            "Replay this exact (block_hash, public_key, seq_len, k_dim, nonce ids) "
            "on NVIDIA to L2-compare vectors. Nonce ids are 0,1,2,… with node_count=1."
        ),
        "nvidia_0731_refs": {"h200_2x": NVIDIA_0731_H200, "h100_4x": NVIDIA_0731_H100},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def cmd_collect(args: argparse.Namespace) -> int:
    block_hash, public_key = resolve_seeds(args.seed_set, args.block_hash, args.public_key)
    os.makedirs(args.out, exist_ok=True)
    jsonl = os.path.join(args.out, "nonces.jsonl")
    STATE.set_jsonl(jsonl)

    httpd = start_callback_server(args.callback_bind, args.callback_port)
    advertise = args.callback_url or f"http://127.0.0.1:{args.callback_port}"
    print(f"callback listening {args.callback_bind}:{args.callback_port} advertise={advertise}", flush=True)

    check_engine(args.url)
    pow_stop(args.url, args.pow_prefix)
    time.sleep(1)
    STATE.clear()

    inp = input_record(
        args,
        block_hash,
        public_key,
        {
            "batch_size": args.batch_size,
            "target_nonces": args.nonces,
            "callback_url": advertise,
        },
    )
    write_json(os.path.join(args.out, "input.json"), inp)
    print(f"logged generation input → {args.out}/input.json", flush=True)
    print(f"  block_hash={block_hash}", flush=True)
    print(f"  public_key={public_key}", flush=True)
    print(f"  seq_len={args.seq_len} k_dim={args.k_dim} batch_size={args.batch_size}", flush=True)

    print(f"init/generate target={args.nonces} …", flush=True)
    print(pow_init(
        args.url, args.pow_prefix,
        model=args.model, batch_size=args.batch_size, callback_url=advertise,
        block_hash=block_hash, public_key=public_key, block_height=args.block_height,
        seq_len=args.seq_len, k_dim=args.k_dim, node_id=0, node_count=1,
    ), flush=True)

    t0 = time.time()
    last_print = -1
    timed_out = False
    while True:
        n, _ = STATE.counts()
        elapsed = time.time() - t0
        if n >= args.nonces:
            break
        if elapsed > args.timeout_s:
            timed_out = True
            break
        if int(elapsed) != last_print and int(elapsed) % 5 == 0:
            last_print = int(elapsed)
            rate = (n / elapsed * 60.0) if elapsed > 0 else 0.0
            engine_n, engine_nps = 0, 0.0
            try:
                engine_n, engine_nps = processed_from_status(pow_status(args.url, args.pow_prefix))
            except requests.RequestException:
                pass
            print(
                f"  callbacks={n}/{args.nonces}  {rate:.0f} n/min  "
                f"engine_processed={engine_n} ({engine_nps * 60:.0f} n/min)  {elapsed:.0f}s",
                flush=True,
            )
        time.sleep(0.5)

    elapsed = time.time() - t0
    engine_n, engine_nps = 0, 0.0
    try:
        engine_n, engine_nps = processed_from_status(pow_status(args.url, args.pow_prefix))
    except requests.RequestException:
        pass
    pow_stop(args.url, args.pow_prefix)
    time.sleep(2)  # last callback flush (POC_CALLBACK_INTERVAL_SEC default 5s)
    arts = STATE.snapshot()[: args.nonces]
    n = len(arts)
    rate = (n / elapsed * 60.0) if elapsed > 0 else 0.0

    dump = {
        "block_hash": block_hash,
        "public_key": public_key,
        "seq_len": args.seq_len,
        "k_dim": args.k_dim,
        "total_nonces": n,
        "artifacts": arts,
        "generation_time_sec": elapsed,
        "nonces_per_min": rate,
        "timed_out": timed_out,
        "engine_total_processed": engine_n,
        "engine_nonces_per_min": engine_nps * 60.0,
    }
    write_json(os.path.join(args.out, "nonces.json"), dump)
    write_json(
        os.path.join(args.out, "rate.json"),
        {
            "callback_nonces": n,
            "callback_nonces_per_min": rate,
            "wall_sec": elapsed,
            "engine_total_processed": engine_n,
            "engine_nonces_per_min": engine_nps * 60.0,
            "batch_size": args.batch_size,
            "timed_out": timed_out,
        },
    )
    STATE.close()
    httpd.shutdown()

    print(
        f"\n{'TIMEOUT' if timed_out else 'DONE'}: {n} callback nonces in {elapsed:.1f}s "
        f"→ {rate:.0f} n/min  (engine {engine_nps * 60:.0f} n/min)",
        flush=True,
    )
    print(f"  {args.out}/input.json     generation inputs (replay on NVIDIA)", flush=True)
    print(f"  {args.out}/nonces.jsonl   one artifact per line as callbacks arrived", flush=True)
    print(f"  {args.out}/nonces.json    collect_artifacts-compatible dump", flush=True)
    print(f"  {args.out}/rate.json      nonces/min", flush=True)
    if n == 0:
        print("FAIL: zero callback nonces — PoC router, worker-extension, or callback URL", flush=True)
        return 1
    return 0


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def cmd_sweep(args: argparse.Namespace) -> int:
    block_hash, public_key = resolve_seeds(args.seed_set, args.block_hash, args.public_key)
    sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)

    httpd = start_callback_server(args.callback_bind, args.callback_port)
    advertise = args.callback_url or f"http://127.0.0.1:{args.callback_port}"
    check_engine(args.url)

    write_json(
        os.path.join(args.out, "input.json"),
        input_record(args, block_hash, public_key, {
            "batch_sizes": sizes,
            "warmup_s": args.warmup_s,
            "duration_s": args.duration_s,
            "callback_url": advertise,
        }),
    )

    rows = []
    for bs in sizes:
        print(f"\n=== batch_size={bs} warmup={args.warmup_s}s window={args.duration_s}s ===", flush=True)
        pow_stop(args.url, args.pow_prefix)
        time.sleep(1)
        STATE.clear()
        pow_init(
            args.url, args.pow_prefix,
            model=args.model, batch_size=bs, callback_url=advertise,
            block_hash=block_hash, public_key=public_key, block_height=args.block_height,
            seq_len=args.seq_len, k_dim=args.k_dim, node_id=0, node_count=1,
        )
        time.sleep(args.warmup_s)
        STATE.clear()
        n0, _ = processed_from_status(pow_status(args.url, args.pow_prefix))
        t0 = time.time()
        time.sleep(args.duration_s)
        n1, nps = processed_from_status(pow_status(args.url, args.pow_prefix))
        elapsed = time.time() - t0
        cb, batches = STATE.counts()
        engine_delta = max(n1 - n0, 0)
        engine_rate = engine_delta / elapsed * 60.0 if elapsed > 0 else 0.0
        cb_rate = cb / elapsed * 60.0 if elapsed > 0 else 0.0
        row = {
            "batch_size": bs,
            "engine_nonces_window": engine_delta,
            "engine_nonces_per_min": engine_rate,
            "engine_nonces_per_min_status": nps * 60.0,
            "callback_nonces_window": cb,
            "callback_nonces_per_min": cb_rate,
            "callback_batches": batches,
            "window_sec": elapsed,
        }
        rows.append(row)
        print(
            f"  engine {engine_delta} nonces / {elapsed:.1f}s → {engine_rate:.0f} n/min  "
            f"callbacks {cb} → {cb_rate:.0f} n/min",
            flush=True,
        )
        pow_stop(args.url, args.pow_prefix)
        time.sleep(1)

    httpd.shutdown()
    best = max(rows, key=lambda r: r["engine_nonces_per_min"]) if rows else {}
    report = {"rows": rows, "best_engine_n_per_min": best}
    write_json(os.path.join(args.out, "sweep.json"), report)
    print("\nbatch  engine n/min  callback n/min", flush=True)
    for r in rows:
        print(f"  {r['batch_size']:>4}  {r['engine_nonces_per_min']:12.0f}  {r['callback_nonces_per_min']:14.0f}", flush=True)
    if best:
        print(f"best engine: batch={best['batch_size']} → {best['engine_nonces_per_min']:.0f} n/min", flush=True)
    print(f"wrote {args.out}/sweep.json", flush=True)
    if not any(r["engine_nonces_window"] or r["callback_nonces_window"] for r in rows):
        print("FAIL: zero nonces across sweep", flush=True)
        return 1
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _load_nonce_file(src: str) -> dict:
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, timeout=60)
        r.raise_for_status()
        return r.json()
    with open(src, encoding="utf-8") as f:
        return json.load(f)


def _art_map(doc: dict) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for a in doc.get("artifacts") or []:
        if isinstance(a, dict) and "nonce" in a and "vector_b64" in a:
            out[int(a["nonce"])] = a["vector_b64"]
    return out


def _decode_f16(b64: str):
    import base64

    import numpy as np

    return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)


def _binom_greater(k: int, n: int, p: float) -> Optional[float]:
    try:
        from scipy.stats import binomtest
    except ImportError:
        return None
    if n == 0:
        return 1.0
    return float(binomtest(k, n, p, alternative="greater").pvalue)


def cmd_compare(args: argparse.Namespace) -> int:
    import numpy as np

    ours = _load_nonce_file(args.ours)
    ref = _load_nonce_file(args.ref)
    a, b = _art_map(ours), _art_map(ref)
    common = sorted(set(a) & set(b))
    if not common:
        print("FAIL: no common nonce ids — dumps must share nonce 0,1,2,…", flush=True)
        return 1

    seed_ok = (
        ours.get("block_hash") == ref.get("block_hash")
        and ours.get("public_key") == ref.get("public_key")
        and ours.get("seq_len") == ref.get("seq_len")
        and ours.get("k_dim") == ref.get("k_dim")
    )
    if not seed_ok:
        print(
            "WARN: generation inputs differ; L2 is not a chain-valid compare.\n"
            f"  ours: hash={ours.get('block_hash')!r} pk={ours.get('public_key')!r} "
            f"seq={ours.get('seq_len')} k={ours.get('k_dim')}\n"
            f"  ref:  hash={ref.get('block_hash')!r} pk={ref.get('public_key')!r} "
            f"seq={ref.get('seq_len')} k={ref.get('k_dim')}",
            flush=True,
        )

    dists = []
    for n in common:
        va, vb = _decode_f16(a[n]), _decode_f16(b[n])
        dists.append(float(np.linalg.norm(va - vb)))
    dists.sort()
    n = len(dists)
    k = sum(1 for d in dists if d > args.dist_threshold)
    p_value = _binom_greater(k, n, args.p_mismatch)
    if p_value is None:
        verdict = "UNKNOWN (pip install scipy)"
        fraud = None
    else:
        fraud = p_value < args.p_value
        verdict = "FRAUD" if fraud else "PASS"

    report = {
        "ours": args.ours,
        "ref": args.ref,
        "seed_inputs_match": seed_ok,
        "common_nonces": n,
        "l2_median": dists[n // 2],
        "l2_mean": statistics.mean(dists),
        "l2_p95": dists[int(n * 0.95)] if n else None,
        "l2_max": dists[-1],
        "dist_threshold": args.dist_threshold,
        "mismatches": k,
        "mismatch_fraction": k / n if n else None,
        "p_mismatch": args.p_mismatch,
        "binomial_p_value": p_value,
        "fraud_p_value_threshold": args.p_value,
        "verdict": verdict,
        "ours_block_hash": ours.get("block_hash"),
        "ref_block_hash": ref.get("block_hash"),
    }
    if args.out:
        write_json(args.out, report)

    print(f"common nonces      : {n}", flush=True)
    print(
        f"L2 median/mean/p95 : {report['l2_median']:.6f} / {report['l2_mean']:.6f} / {report['l2_p95']:.6f}",
        flush=True,
    )
    print(f"L2 max             : {report['l2_max']:.6f}", flush=True)
    print(
        f"mismatches (>{args.dist_threshold}) : {k}/{n} ({100 * k / n:.2f}%)",
        flush=True,
    )
    if p_value is None:
        print("binomial p-value   : n/a (scipy missing)", flush=True)
    else:
        print(f"binomial P(X>=k)   : {p_value:.3e}  (fraud if < {args.p_value})", flush=True)
    print(f"VERDICT            : {verdict}", flush=True)
    print(f"seed inputs match  : {seed_ok}", flush=True)
    if args.out:
        print(f"wrote {args.out}", flush=True)

    # 0.5 records the number even when MetaX vs NVIDIA is FRAUD (MACA ≠ CUDA).
    # Exit 2 only on hard compare failures (no overlap). FRAUD is exit 0 with warning.
    if not seed_ok:
        return 0
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url", required=True, help="vLLM/mlnode base URL, e.g. http://127.0.0.1:8000")
    p.add_argument("--model", required=True, help="served model id (must match /v1/models)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--pow-prefix", default="/api/v1/pow", help="PoC route prefix (mlnode: /api/v1/inference/pow)")
    p.add_argument("--callback-bind", default="0.0.0.0")
    p.add_argument("--callback-port", type=int, default=9998)
    p.add_argument(
        "--callback-url",
        default="",
        help="URL the engine POSTs to (default http://127.0.0.1:PORT; needs --network=host)",
    )
    p.add_argument("--seed-set", choices=("nvidia-0731", "generic"), default="nvidia-0731")
    p.add_argument("--block-hash", default="", help="override block_hash")
    p.add_argument("--public-key", default="", help="override public_key")
    p.add_argument("--block-height", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--k-dim", type=int, default=12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="callback collect + nonces/min + dump")
    _add_engine_args(c)
    c.add_argument("--nonces", type=int, default=256)
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--timeout-s", type=int, default=600)
    c.set_defaults(func=cmd_collect)

    s = sub.add_parser("sweep", help="nonces/min vs batch_size")
    _add_engine_args(s)
    s.add_argument("--batch-sizes", default="8,16,32")
    s.add_argument("--warmup-s", type=float, default=5.0)
    s.add_argument("--duration-s", type=float, default=30.0)
    s.set_defaults(func=cmd_sweep)

    q = sub.add_parser("compare", help="L2 C550 dump vs NVIDIA H100/H200 dump")
    q.add_argument("--ours", required=True, help="C550 nonces.json (or URL)")
    q.add_argument("--ref", required=True, help="NVIDIA dump path or URL")
    q.add_argument("--out", default="", help="write JSON report here")
    q.add_argument("--dist-threshold", type=float, default=V4_DIST_THRESHOLD)
    q.add_argument("--p-mismatch", type=float, default=V4_P_MISMATCH)
    q.add_argument("--p-value", type=float, default=V4_FRAUD_THRESHOLD)
    q.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    if hasattr(args, "block_hash"):
        args.block_hash = args.block_hash or None
        args.public_key = args.public_key or None
        args.callback_url = args.callback_url or None
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
