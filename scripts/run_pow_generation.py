#!/usr/bin/env python3
"""
PoC v2 Generation Test Script

Validates MLNode PoC v2 (Proof of Computation) generation and fraud detection
for the Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 model.

PHASES:
  0. Setup Check            - Verify configuration (container, ports, network)
  1. Generation + Validation - Generate PoC v2 artifacts, then validate them (self-test)
  2. Fraud Detection        - Test pre-collected honest/fraud vectors
  3. Batch Sizing           - Benchmark batch sizes to find optimal performance

USAGE:
    python run_pow_generation.py             # Run all phases
    python run_pow_generation.py --phase 0   # Only check setup (recommended first!)
    python run_pow_generation.py --phase 2   # Only fraud detection test
    python run_pow_generation.py --skip-check  # Skip setup verification

CONFIGURATION:
    Environment variables (recommended):
        export CONTAINER_NAME="join-inference-1"  # Your MLNode container name
        export HOST_IP="172.18.0.1"               # Docker gateway IP

    Or edit at top of script:
        MLNODE_URL              - MLNode API URL (default: http://localhost:8080)
        BATCH_RECEIVER_PORT     - Callback server port (default: 9999)
        WARMUP_DURATION_S       - Warmup duration before measuring (default: 5)
        GENERATION_DURATION_S   - Test duration in seconds (default: 30)
        BATCH_SIZES_TO_TEST     - Batch sizes for phase 3 benchmark

REQUIREMENTS:
    pip install requests

See POW_TEST.md for detailed troubleshooting and setup instructions.
"""

import time
import requests
import threading
import subprocess
import os
import argparse
from typing import List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import json


# =============================================================================
# Configuration
# =============================================================================

def get_docker_host_ip(container_name: str = "join-inference-1") -> str:
    """Get the gateway IP for a Docker container."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}", container_name],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "172.18.0.1"


MLNODE_URL = "http://127.0.0.1:8000"
BATCH_RECEIVER_PORT = 9999
CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "join-inference-1")
HOST_IP = os.environ.get("HOST_IP") or get_docker_host_ip(CONTAINER_NAME)
BATCH_RECEIVER_LOCAL_URL = f"http://localhost:{BATCH_RECEIVER_PORT}"
BATCH_RECEIVER_URL = f"http://{HOST_IP}:{BATCH_RECEIVER_PORT}"

# Generation parameters
API_PREFIX = "/api/v1"
WARMUP_DURATION_S = 5
GENERATION_DURATION_S = 30
BATCH_SIZES_TO_TEST = [8, 16, 32]

# =============================================================================
# Test Vectors for Phase 2 Validation
# =============================================================================
# Pre-collected vectors for Qwen/Qwen3-235B-A22B-Instruct-2507-FP8

TEST_VECTORS = {
    # Honest vectors: from FP8 model (seq_len=1024, k_dim=12)
    "honest": [
        {"nonce": 0, "vector_b64": "QjHzs0MtjK6wGwG4nC65LR42u7CYNaG4"},
        {"nonce": 1, "vector_b64": "9aokNXI28DUstOawC7GMNTaxHyO6MV84"},
        {"nonce": 2, "vector_b64": "3jHPswc49jMiLtwwmrCruJYyCjJ2tZAr"},
        {"nonce": 3, "vector_b64": "JyoltF2qTrR+NdC2CrEPNvK0wbeAsWmx"},
        {"nonce": 4, "vector_b64": "czVQMBqirbRbt/+0izBXOJQsD7Q1syEy"},
        {"nonce": 5, "vector_b64": "ZTA5NKOypjVQNAW2Ti7ONam1wrYBK5U0"},
        {"nonce": 6, "vector_b64": "IrVBtW+17rPCrpKwAbQ0McE26TSmNvGy"},
        {"nonce": 7, "vector_b64": "grBvtRG3a6YPNO4rJTTJtZE1bTSkqTe3"},
        {"nonce": 8, "vector_b64": "8rQiNT+1ha8cN4Ux0JhVOCm1wy9bsNcw"},
        {"nonce": 9, "vector_b64": "Vjdhs/us0TLmOZWugSTwMPCwKqwcsP6z"},
        {"nonce": 10, "vector_b64": "JqCgMr8zxa8XJHQtNTNFOes2jjNGtHK0"},
        {"nonce": 11, "vector_b64": "KTSQNDA49bItrf+3v7OlNEWnMS2GMpQ0"},
        {"nonce": 12, "vector_b64": "86npND4sy7R9Npm2Tq05OJoxXDSZNPmp"},
        {"nonce": 13, "vector_b64": "NSyZtg6wtiKot0G4Xy2Us3a3Paz/Lqym"},
        {"nonce": 14, "vector_b64": "wSBkJmgy+a8WNAY1aTkasuI2Iq5Xsloz"},
        {"nonce": 15, "vector_b64": "BDYQtqoxkLXTtngy7jNjtcizHjM3sFwy"},
        {"nonce": 16, "vector_b64": "1biHtRIwObBgtJGwdTU5tDYjeqQkt6Ao"},
        {"nonce": 17, "vector_b64": "WzSlNFC3wzS0LdGihrS8rhwwlzPEODAw"},
        {"nonce": 18, "vector_b64": "WjFfLCWso7ZatxQsLTIYsUox4DTYtRQ4"},
        {"nonce": 19, "vector_b64": "eTA9tpUz77QHsCovmK51MRU1fq4qsX+5"},
        {"nonce": 20, "vector_b64": "+zVzuDadkza2Mnwp4TVqtL2s3jD4NAYQ"},
        {"nonce": 21, "vector_b64": "jig7pDk1lDaNpuQyvTa4rb8tVDgHsKw2"},
        {"nonce": 22, "vector_b64": "cLW0NZS2drXCrLi217J/s621EzEKsgIr"},
        {"nonce": 23, "vector_b64": "5qZYtPCyRrbhthSyRLagNUu2riUZs8kw"},
        {"nonce": 24, "vector_b64": "gy9WOg0veKJ+K1QssrZksWe0HjPiLb6o"},
        {"nonce": 25, "vector_b64": "jra/qyO10DW8rUwuzzhENjAyKCz6LxIs"},
        {"nonce": 26, "vector_b64": "nDncMq41zDTDMpeiKLWtsN6yVK97nS6w"},
        {"nonce": 27, "vector_b64": "gbFltMQ1Up9kNHM2eDWKsCQ0oi4kLW24"},
        {"nonce": 28, "vector_b64": "PK6puEazZKNYNTW0vrXoMgWrHLKNrgm3"},
        {"nonce": 29, "vector_b64": "xjMssCExVLbeMSk2d7Y7tWkx0jbLs5Ew"},
        {"nonce": 30, "vector_b64": "1LI9No6xlzUZshGqLLGaro+2NK21NpQ3"},
        {"nonce": 31, "vector_b64": "yrS3IPQ0Cbg+Mne1CTLTLz8xSjTit2cv"},
    ],
    # Fraud vectors: from INT4 model (different quantization = different vectors)
    "fraud": [
        {"nonce": 0, "vector_b64": "jS/Es/8ukbP/rIS2kC+rIlM2IbDINCG5"},
        {"nonce": 1, "vector_b64": "Qa5qNJ42AzZTtK6yDrGfNFmzRB0CLYQ4"},
        {"nonce": 2, "vector_b64": "YaXVsho3DjSdqy0wDbJ8uBgsaDRit38x"},
        {"nonce": 3, "vector_b64": "iCh7s9+tSLSZNTO2WLCWNsG0KLgqsG+x"},
        {"nonce": 4, "vector_b64": "DTYmL8apwLQ7t1G1li8dOHYs77N1snkz"},
        {"nonce": 5, "vector_b64": "wzChMXu1KTZ1NFu2lSk/NuKzOrVTIqI1"},
        {"nonce": 6, "vector_b64": "MbUPtVK1zrOJrt6xTLS5MaU2mjTBNqiy"},
        {"nonce": 7, "vector_b64": "BrMhtVm2MCPmMS0vvjMethk2cTQ/qm63"},
        {"nonce": 8, "vector_b64": "JbIwNQi1o6vnNnk0LCybOL20oDAfsaYw"},
        {"nonce": 9, "vector_b64": "ZDdlssirTjLzOVOwwCO3MHKxfq1er/Gz"},
        {"nonce": 10, "vector_b64": "5CirLggxFadEMVUpwy9GOmo1aipntFW1"},
        {"nonce": 11, "vector_b64": "njUTNW03ArUHpri4tJ8dNHUvx6qSL1Ew"},
        {"nonce": 12, "vector_b64": "pqj3NKkx1rLENtO2YqwWODAzIjMANU6o"},
        {"nonce": 13, "vector_b64": "YC1ut4uxbrCtt3m3mzGRtEy2v6sHLj2i"},
        {"nonce": 14, "vector_b64": "KjM7KAgzzSyTNL02sThHsTQ1GK4JIzE2"},
        {"nonce": 15, "vector_b64": "PjYJuPmjV7YgtcI00S/htDKyRTRALPcw"},
        {"nonce": 16, "vector_b64": "JrkGto0uGLAKtu6iHTRfsEsr8qqKtiMw"},
        {"nonce": 17, "vector_b64": "FDTGNHa3wzTGLoqcOrQmr+wvZjPKOHEw"},
        {"nonce": 18, "vector_b64": "XDBkLXOrPbQ3taYxRTMIsQs0IjQGtWo5"},
        {"nonce": 19, "vector_b64": "6TKCtZkzZrIYsfwsT7BDMKc1eq5PsqS5"},
        {"nonce": 20, "vector_b64": "GTZHuIkqsTc/MmiQ8TVTtNmuBC/vM5cn"},
        {"nonce": 21, "vector_b64": "xCZcMHQ3yjY+rLIvojVHsMowQjeymhA3"},
        {"nonce": 22, "vector_b64": "Y7bCNTq1drC2sOi2vLJwtF+1ijTXrKQz"},
        {"nonce": 23, "vector_b64": "C6RltDWzWLaxtgey77XCNZK2uyQQs6sw"},
        {"nonce": 24, "vector_b64": "2zA2OowuzKQ5LAEu17ZNqYy0KDTXLyOq"},
        {"nonce": 25, "vector_b64": "Q7Xpm2mzfjPvsfIvUTkXN0A0CzE2JCKi"},
        {"nonce": 26, "vector_b64": "KTmdMPY1NzWCMk6otLUMsp20eay6qOCw"},
        {"nonce": 27, "vector_b64": "jrOhtr83/CU1MAU1DDUemAU1cDFwnxG3"},
        {"nonce": 28, "vector_b64": "mK+guGC0c6wTNja0S7UbLXwijrMKr7K2"},
        {"nonce": 29, "vector_b64": "5zPBr7kwmbaEMVI2GrY/tZowCTfds3Ew"},
        {"nonce": 30, "vector_b64": "4a8xNhSydTazsjqwGa5NsvO1p6npNQE4"},
        {"nonce": 31, "vector_b64": "jLNTK7I0grhMNMWyIjOiMwA0KzGQt8Up"},
    ],
}

# Model configuration for Qwen 235B
MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"
# Pin the exact HF revision so a silent re-push cannot change what we measure.
MODEL_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"

INFERENCE_PAYLOAD = {
    "model": MODEL_NAME,
    "dtype": "auto",
    "additional_args": [
        "--tensor-parallel-size", "4",
        "--max-model-len", "240000",
    ],
}

POW_CONFIG = {
    "block_hash": "TEST_BLOCK",
    "block_height": 100,
    "public_key": "test_pub_keys",
    "node_id": 0,
    "node_count": 1,
    "params": {
        "model": MODEL_NAME,
        "seq_len": 1024,
        "k_dim": 12,
    },
}


# =============================================================================
# Batch Receiver Server
# =============================================================================

_proof_batches: List[dict] = []
_validated_batches: List[dict] = []
_server_instance: Optional[HTTPServer] = None


class BatchReceiverHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    
    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _count_nonces(self, batch: dict) -> int:
        if "artifacts" in batch:
            return len(batch["artifacts"])
        elif "nonces" in batch:
            return len(batch["nonces"])
        return 0
    
    def do_GET(self):
        if self.path == '/health':
            self._send_json({"status": "OK"})
        elif self.path == '/stats':
            total_nonces = sum(self._count_nonces(b) for b in _proof_batches)
            batch_count = len(_proof_batches)
            batch_sizes = [self._count_nonces(b) for b in _proof_batches]
            avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
            self._send_json({
                "total_nonces": total_nonces,
                "batch_count": batch_count,
                "batch_sizes": batch_sizes,
                "avg_batch_size": avg_batch_size,
            })
        elif self.path == '/batches':
            self._send_json({"batches": _proof_batches})
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode() if content_length > 0 else '{}'
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return
        
        if self.path == '/generated':
            _proof_batches.append(data)
            self._send_json({"message": "OK", "batch_count": len(_proof_batches)})
        elif self.path == '/validated':
            _validated_batches.append(data)
            self._send_json({"message": "OK"})
        elif self.path == '/clear':
            _proof_batches.clear()
            _validated_batches.clear()
            self._send_json({"message": "Cleared"})
        else:
            self._send_json({"error": "Not found"}, 404)


def run_server(port: int):
    global _server_instance
    _server_instance = HTTPServer(('0.0.0.0', port), BatchReceiverHandler)
    _server_instance.serve_forever()


def start_batch_receiver() -> threading.Thread:
    thread = threading.Thread(target=run_server, args=(BATCH_RECEIVER_PORT,), daemon=True)
    thread.start()
    return thread


def stop_batch_receiver():
    global _server_instance
    if _server_instance:
        _server_instance.shutdown()


def wait_for_batch_receiver_ready(timeout_s: int = 30) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout_s:
        try:
            response = requests.get(f"{BATCH_RECEIVER_LOCAL_URL}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False


def get_batch_receiver_stats() -> dict:
    response = requests.get(f"{BATCH_RECEIVER_LOCAL_URL}/stats", timeout=10)
    response.raise_for_status()
    return response.json()


def get_collected_batches() -> List[dict]:
    response = requests.get(f"{BATCH_RECEIVER_LOCAL_URL}/batches", timeout=10)
    response.raise_for_status()
    return response.json().get("batches", [])


def clear_batch_receiver():
    response = requests.post(f"{BATCH_RECEIVER_LOCAL_URL}/clear", timeout=10)
    response.raise_for_status()


# =============================================================================
# vLLM / PoC v2 API Functions
# =============================================================================

def is_vllm_running() -> bool:
    try:
        response = requests.get(f"{MLNODE_URL}/health", timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def start_vllm_if_needed():
    if is_vllm_running():
        print("vLLM is already running.")
        return True
    
    print("Starting vLLM...")
    try:
        response = requests.post(
            f"{MLNODE_URL}{API_PREFIX}/inference/up/async",
            json=INFERENCE_PAYLOAD, timeout=60
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409:
            print("vLLM already running.")
            return True
        raise
    
    # Wait for vLLM to be ready
    print("Waiting for vLLM to be ready...")
    for i in range(60):
        time.sleep(5)
        if is_vllm_running():
            print("vLLM is ready!")
            return True
        print(f"  [{(i+1)*5}s] Still waiting...")
    
    print("Timeout waiting for vLLM")
    return False


def init_generate(batch_size: int, max_retries: int = 12, retry_delay: int = 5) -> dict:
    """Initialize PoC v2 generation with retries for 503 errors (model still loading)."""
    payload = {
        **POW_CONFIG,
        "batch_size": batch_size,
        "url": BATCH_RECEIVER_URL,
    }
    
    for attempt in range(max_retries):
        response = requests.post(
            f"{MLNODE_URL}{API_PREFIX}/pow/init/generate",
            json=payload, timeout=60
        )
        
        if response.status_code == 503:
            if attempt < max_retries - 1:
                print(f"  vLLM not ready (503), retrying in {retry_delay}s... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                print("  vLLM still not ready after retries.")
                response.raise_for_status()
        
        response.raise_for_status()
        return response.json()
    
    raise RuntimeError("Failed to initialize PoC v2 generation")


def stop_generation() -> dict:
    response = requests.post(
        f"{MLNODE_URL}{API_PREFIX}/pow/stop",
        json={}, timeout=60
    )
    response.raise_for_status()
    return response.json()


def send_validation_request(artifacts: List[dict]) -> dict:
    """Send artifacts for validation using the /generate endpoint with validation field."""
    # Extract nonces from artifacts
    nonces = [a["nonce"] for a in artifacts]
    
    payload = {
        "block_hash": POW_CONFIG["block_hash"],
        "block_height": POW_CONFIG["block_height"],
        "public_key": POW_CONFIG["public_key"],
        "node_id": POW_CONFIG["node_id"],
        "node_count": POW_CONFIG["node_count"],
        "nonces": nonces,
        "params": POW_CONFIG["params"],
        "batch_size": 32,
        "wait": True,
        "validation": {
            "artifacts": artifacts
        },
    }
    
    response = requests.post(
        f"{MLNODE_URL}{API_PREFIX}/pow/generate",
        json=payload, timeout=300
    )
    response.raise_for_status()
    return response.json()


# =============================================================================
# Phase 1: Generation + Validation
# =============================================================================

def run_phase1_generation_validation():
    """Phase 1: Generate artifacts, then validate them."""
    print("\n" + "=" * 50)
    print("  PHASE 1: Generation + Self-Validation")
    print("=" * 50)
    
    # Clear and start generation
    clear_batch_receiver()
    total_duration = WARMUP_DURATION_S + GENERATION_DURATION_S
    print(f"\nGenerating for {total_duration}s ({WARMUP_DURATION_S}s warmup + {GENERATION_DURATION_S}s measurement, batch_size=32)...")
    
    init_generate(batch_size=32)
    
    # Warmup period
    time.sleep(WARMUP_DURATION_S)
    print(f"  {WARMUP_DURATION_S}s warmup done")
    
    # Clear stats after warmup and start measuring
    clear_batch_receiver()
    start_time = time.time()
    
    # Wait for generation duration with progress (every 10s)
    for i in range(GENERATION_DURATION_S // 10):
        time.sleep(10)
        stats = get_batch_receiver_stats()
        elapsed = (i + 1) * 10
        print(f"  [{elapsed:2d}s] {stats['total_nonces']} nonces")
    
    elapsed_time = time.time() - start_time
    stop_generation()
    
    # Get final stats
    final_stats = get_batch_receiver_stats()
    print(f"\nGeneration completed:")
    print(f"  Total nonces: {final_stats['total_nonces']}")
    print(f"  Duration: {elapsed_time:.1f}s")
    print(f"  Speed: {final_stats['total_nonces'] / elapsed_time * 60:.0f} nonces/min")
    
    # Collect all artifacts for validation
    batches = get_collected_batches()
    all_artifacts = []
    for batch in batches:
        if "artifacts" in batch:
            all_artifacts.extend(batch["artifacts"])
    
    if not all_artifacts:
        print("\nNo artifacts collected - skipping validation")
        return
    
    print(f"\nSending {len(all_artifacts)} artifacts for validation...")
    
    try:
        result = send_validation_request(all_artifacts)
        
        print("\n" + "-" * 40)
        print("  VALIDATION RESULTS")
        print("-" * 40)
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Total nonces: {result.get('n_total', 0)}")
        print(f"  Mismatches: {result.get('n_mismatch', 0)}")
        print(f"  p-value: {result.get('p_value', 'N/A')}")
        print(f"  Fraud detected: {result.get('fraud_detected', 'N/A')}")
        print("-" * 40)
        
    except requests.exceptions.RequestException as e:
        print(f"\nValidation failed: {e}")


# =============================================================================
# Phase 2: Validate External
# =============================================================================

def _validate_artifacts(artifacts: List[dict], test_name: str, expect_fraud: bool) -> bool:
    """Helper to validate artifacts and check result against expectation."""
    try:
        result = send_validation_request(artifacts)
        
        print("\n" + "-" * 40)
        print(f"  {test_name}")
        print("-" * 40)
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Total nonces: {result.get('n_total', 0)}")
        print(f"  Mismatches: {result.get('n_mismatch', 0)}")
        print(f"  p-value: {result.get('p_value', 'N/A')}")
        print(f"  Fraud detected: {result.get('fraud_detected', 'N/A')}")
        print("-" * 40)
        
        fraud_detected = result.get('fraud_detected', False)
        
        if expect_fraud:
            if fraud_detected:
                print("\n  ✓ EXPECTED: Fraud correctly detected.")
                return True
            else:
                print("\n  ✗ UNEXPECTED: No fraud detected!")
                return False
        else:
            if fraud_detected:
                print("\n  ✗ UNEXPECTED: Fraud detected on valid artifacts!")
                return False
            else:
                print("\n  ✓ EXPECTED: Validation passed (no fraud).")
                return True
                
    except requests.exceptions.RequestException as e:
        print(f"\nValidation request failed: {e}")
        return False


def run_phase2_validate_external():
    """Phase 2: Validate hardcoded pre-collected vectors."""
    print("\n" + "=" * 50)
    print("  PHASE 2: Fraud Detection Test")
    print("=" * 50)
    
    honest_artifacts = TEST_VECTORS["honest"]
    fraud_artifacts = TEST_VECTORS["fraud"]
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Honest vectors: {len(honest_artifacts)}")
    print(f"Fraud vectors: {len(fraud_artifacts)}")
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Test 1: Honest artifacts (should pass validation)
    # -------------------------------------------------------------------------
    print(f"\n[Test 1] Sending HONEST artifacts...")
    results["honest"] = _validate_artifacts(
        honest_artifacts, 
        "TEST 1: HONEST VECTORS", 
        expect_fraud=False
    )
    
    # -------------------------------------------------------------------------
    # Test 2: Fraud artifacts (should FAIL validation / detect fraud)
    # -------------------------------------------------------------------------
    print(f"\n[Test 2] Sending FRAUD artifacts...")
    results["fraud"] = _validate_artifacts(
        fraud_artifacts,
        "TEST 2: FRAUD VECTORS",
        expect_fraud=True
    )
    
    # Summary
    print("\n" + "-" * 40)
    print("  SUMMARY")
    print("-" * 40)
    print(f"  Honest vectors:  {'✓ PASS' if results['honest'] else '✗ FAIL'}")
    print(f"  Fraud detection: {'✓ PASS' if results['fraud'] else '✗ FAIL'}")
    print("-" * 40)
    
    return all(results.values())


# =============================================================================
# Phase 3: Autobatch Sizing
# =============================================================================

def run_phase3_autobatch_sizing():
    """Phase 3: Test different batch sizes and compare performance."""
    print("\n" + "=" * 50)
    print("  PHASE 3: Batch Size Benchmark")
    print("=" * 50)
    print(f"  Batch sizes: {BATCH_SIZES_TO_TEST}")
    total_duration = WARMUP_DURATION_S + GENERATION_DURATION_S
    print(f"  Duration: {total_duration}s each ({WARMUP_DURATION_S}s warmup + {GENERATION_DURATION_S}s measurement)")
    
    results = []
    
    for batch_size in BATCH_SIZES_TO_TEST:
        print(f"\n{'─' * 40}")
        print(f"  Testing batch_size = {batch_size}")
        print(f"{'─' * 40}")
        
        # Clear batch receiver and verify it's empty
        clear_batch_receiver()
        time.sleep(0.5)  # Allow any in-flight batches to arrive
        clear_batch_receiver()  # Clear again to catch stragglers
        
        # Verify we start from zero
        initial_stats = get_batch_receiver_stats()
        if initial_stats['total_nonces'] != 0:
            print(f"  WARNING: Starting with {initial_stats['total_nonces']} leftover nonces, clearing again...")
            clear_batch_receiver()
        
        try:
            init_generate(batch_size=batch_size)
        except requests.exceptions.RequestException as e:
            print(f"  Failed to start: {e}")
            results.append({
                "batch_size": batch_size,
                "nonces": 0,
                "duration": 0,
                "nonces_per_min": 0,
                "error": str(e),
            })
            continue
        
        # Warmup period
        time.sleep(WARMUP_DURATION_S)
        print(f"    {WARMUP_DURATION_S}s warmup done")
        
        # Clear stats after warmup and start measuring
        clear_batch_receiver()
        start_time = time.time()
        
        # Wait for generation duration (progress every 10s)
        for i in range(GENERATION_DURATION_S // 10):
            time.sleep(10)
            stats = get_batch_receiver_stats()
            elapsed = (i + 1) * 10
            print(f"    [{elapsed:2d}s] {stats['total_nonces']} nonces")
        
        elapsed_time = time.time() - start_time
        
        try:
            stop_generation()
        except Exception:
            pass
        
        # Wait for any in-flight batches to arrive before recording final stats
        time.sleep(1.0)
        
        # Record results
        final_stats = get_batch_receiver_stats()
        nonces = final_stats['total_nonces']
        nonces_per_min = nonces / elapsed_time * 60 if elapsed_time > 0 else 0
        
        results.append({
            "batch_size": batch_size,
            "nonces": nonces,
            "duration": elapsed_time,
            "nonces_per_min": nonces_per_min,
        })
        
        print(f"  Result: {nonces} nonces, {nonces_per_min:.0f}/min")
    
    # Print summary table
    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"  {'Batch':>8} │ {'Nonces':>8} │ {'Nonces/min':>12}")
    print(f"  {'─' * 8}─┼─{'─' * 8}─┼─{'─' * 12}")
    
    best_result = max(results, key=lambda x: x['nonces_per_min']) if results else None
    
    for r in results:
        marker = " ★" if r == best_result else "  "
        if "error" in r:
            print(f"  {r['batch_size']:>8} │ {'ERROR':>8} │ {'-':>12}{marker}")
        else:
            print(f"  {r['batch_size']:>8} │ {r['nonces']:>8} │ {r['nonces_per_min']:>12.0f}{marker}")
    
    print("=" * 50)
    
    if best_result and "error" not in best_result:
        print(f"  ★ Best batch size: {best_result['batch_size']} ({best_result['nonces_per_min']:.0f} nonces/min)")


# =============================================================================
# Setup Check
# =============================================================================

def check_setup(phases: List[int]) -> bool:
    """
    Verify configuration before running tests.
    Returns True if all checks pass, False otherwise.
    """
    print("\n" + "=" * 50)
    print("  Setup Check")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # -------------------------------------------------------------------------
    # Check 1: MLNode reachable
    # -------------------------------------------------------------------------
    print(f"\n[1/5] MLNode at {MLNODE_URL}...")
    mlnode_reachable = False
    try:
        response = requests.get(f"{MLNODE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"      ✓ Reachable")
            mlnode_reachable = True
        else:
            print(f"      ✗ Returned status {response.status_code}")
            issues.append(f"MLNode returned HTTP {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"      ✗ Connection refused")
        issues.append(f"Cannot connect to MLNode at {MLNODE_URL}")
    except requests.exceptions.Timeout:
        print(f"      ✗ Timeout")
        issues.append(f"MLNode at {MLNODE_URL} timed out")
    except Exception as e:
        print(f"      ✗ Error: {e}")
        issues.append(f"MLNode check failed: {e}")
    
    # -------------------------------------------------------------------------
    # Check 2: vLLM / Model status
    # -------------------------------------------------------------------------
    print(f"\n[2/5] vLLM model status...")
    if mlnode_reachable:
        try:
            response = requests.get(f"{MLNODE_URL}{API_PREFIX}/inference/up/status", timeout=10)
            status_data = response.json()
            is_running = status_data.get("is_running", False)
            
            if is_running:
                print(f"      ✓ Model is loaded and running")
            else:
                print(f"      ⚠ Model not loaded (script will start it automatically)")
                warnings.append("vLLM model not running - will be started automatically (may take several minutes)")
        except Exception as e:
            print(f"      ? Could not check: {e}")
            warnings.append(f"Could not verify vLLM status: {e}")
    else:
        print(f"      - Skipped (MLNode not reachable)")
    
    # -------------------------------------------------------------------------
    # Check 3: Container exists (only needed for phases 1, 3)
    # -------------------------------------------------------------------------
    actual_container = None
    if 1 in phases or 3 in phases:
        print(f"\n[3/5] Container '{CONTAINER_NAME}'...")
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5
            )
            running_containers = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            if CONTAINER_NAME in running_containers:
                print(f"      ✓ Found")
                actual_container = CONTAINER_NAME
            else:
                # Try to find inference container
                inference_containers = [c for c in running_containers if 'inference' in c.lower()]
                if inference_containers:
                    actual_container = inference_containers[0]
                    print(f"      ✗ Not found")
                    print(f"      → Found '{actual_container}' instead")
                    warnings.append(f"Container mismatch: expected '{CONTAINER_NAME}', found '{actual_container}'")
                    warnings.append(f"  Fix: export CONTAINER_NAME=\"{actual_container}\"")
                else:
                    print(f"      ✗ Not found (no inference containers running)")
                    issues.append(f"Container '{CONTAINER_NAME}' not found")
        except Exception as e:
            print(f"      ✗ Docker check failed: {e}")
            warnings.append(f"Could not verify container (docker not available?)")
    else:
        print(f"\n[2/4] Container check... SKIPPED (not needed for phase 2)")
    
    # -------------------------------------------------------------------------
    # Check 4: Callback port available (only needed for phases 1, 3)
    # -------------------------------------------------------------------------
    if 1 in phases or 3 in phases:
        print(f"\n[4/5] Port {BATCH_RECEIVER_PORT} available...")
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', BATCH_RECEIVER_PORT))
            sock.close()
            print(f"      ✓ Available")
        except OSError as e:
            print(f"      ✗ In use or blocked")
            issues.append(f"Port {BATCH_RECEIVER_PORT} is not available: {e}")
    else:
        print(f"\n[4/5] Port check... SKIPPED (not needed for phase 2)")
    
    # -------------------------------------------------------------------------
    # Check 5: HOST_IP reachable from container (only needed for phases 1, 3)
    # -------------------------------------------------------------------------
    if 1 in phases or 3 in phases:
        print(f"\n[5/5] HOST_IP '{HOST_IP}' reachable from container...")
        container_to_check = actual_container or CONTAINER_NAME
        
        # First verify the expected gateway
        try:
            result = subprocess.run(
                ["docker", "inspect", container_to_check, 
                 "--format", "{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}"],
                capture_output=True, text=True, timeout=5
            )
            detected_gateway = result.stdout.strip()
            
            if detected_gateway:
                if detected_gateway == HOST_IP:
                    print(f"      ✓ Matches container gateway ({detected_gateway})")
                else:
                    print(f"      ✗ Mismatch: container gateway is '{detected_gateway}'")
                    warnings.append(f"HOST_IP mismatch: configured '{HOST_IP}', container gateway is '{detected_gateway}'")
                    warnings.append(f"  Fix: export HOST_IP=\"{detected_gateway}\"")
            else:
                print(f"      ? Could not detect gateway (using configured value)")
        except Exception as e:
            print(f"      ? Could not verify: {e}")
            warnings.append(f"Could not verify HOST_IP (docker inspect failed)")
    else:
        print(f"\n[5/5] HOST_IP check... SKIPPED (not needed for phase 2)")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    
    if warnings:
        print("  WARNINGS:")
        for w in warnings:
            print(f"    ⚠ {w}")
    
    if issues:
        print("  ERRORS:")
        for i in issues:
            print(f"    ✗ {i}")
        print("-" * 50)
        print("\n  Setup check FAILED. Fix the issues above and retry.")
        print("  See POW_TEST.md 'Custom Setup Guide' for help.\n")
        return False
    
    if warnings:
        print("-" * 50)
        print("\n  Setup check PASSED with warnings.")
        print("  The script may work, but if you see issues, fix the warnings above.\n")
    else:
        print("  ✓ All checks passed!")
        print("-" * 50)
    
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PoC v2 Generation Test")
    parser.add_argument("--phase", type=int, choices=[0, 1, 2, 3], 
                        help="Run only specific phase (0=setup check, 1=generate+validate, 2=fraud test, 3=batch sizing)")
    parser.add_argument("--skip-check", action="store_true",
                        help="Skip setup check before phases 1-3 (not recommended)")
    args = parser.parse_args()
    
    # Phase 0: Setup check only
    if args.phase == 0:
        phases_to_check = [1, 2, 3]  # Check all requirements
        if check_setup(phases_to_check):
            print("\n✓ Setup check complete. Your configuration looks good!")
            return 0
        return 1
    
    # Determine which phases will run
    if args.phase is not None:
        phases_to_run = [args.phase]
    else:
        phases_to_run = [1, 2, 3]
    
    # Run setup check (unless skipped)
    if not args.skip_check:
        if not check_setup(phases_to_run):
            return 1
    
    print("\n" + "=" * 50)
    print("  PoC v2 Generation Test")
    print("=" * 50)
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Warmup:   {WARMUP_DURATION_S}s")
    print(f"  Duration: {GENERATION_DURATION_S}s per test")
    print(f"  Callback: {BATCH_RECEIVER_URL}")
    
    receiver_thread = None
    
    try:
        # Start batch receiver (needed for phases 1 and 3)
        if args.phase is None or args.phase in [1, 3]:
            print("\nStarting batch receiver...")
            receiver_thread = start_batch_receiver()
            if not wait_for_batch_receiver_ready():
                print("Failed to start batch receiver")
                return 1
            print("Batch receiver ready.")
        
        # Ensure vLLM is running
        if not start_vllm_if_needed():
            return 1
        
        # Run phases
        if args.phase is None or args.phase == 1:
            run_phase1_generation_validation()
        
        if args.phase is None or args.phase == 2:
            run_phase2_validate_external()
        
        if args.phase is None or args.phase == 3:
            run_phase3_autobatch_sizing()
        
        print("\n✓ Benchmark complete!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130
    
    finally:
        if receiver_thread:
            print("\nStopping batch receiver...")
            stop_batch_receiver()
            receiver_thread.join(timeout=5)


if __name__ == "__main__":
    exit(main())
