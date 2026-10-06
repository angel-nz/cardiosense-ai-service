#!/usr/bin/env python3
"""CardioSense AI deployment probe.

Reads AI_SERVICE_KEY from the environment, never prints it, and exercises the
canonical /health + /predict contract against a local or remote deployment.
This is a capability probe, not a load test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

PAYLOAD = {
    "age": 55,
    "sex": 1,
    "currentSmoker": 0,
    "cigsPerDay": 0,
    "BPMeds": 1,
    "diabetes": 0,
    "totChol": 230.5,
    "sysBP": 145.0,
    "diaBP": 92.0,
    "BMI": 28.4,
    "glucose": 105.0,
}
EXPECTED_MODEL_VERSION = "Skorp-Beta-0.2"
EXPECTED_AGE_RANGE = {"min": 32, "max": 81}
KNOWN_DEVELOPMENT_KEY = "internal-dev-key"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_local_target(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in LOCAL_HOSTS


def request_json(method: str, url: str, *, key: str | None = None, payload: dict[str, Any] | None = None, timeout: float = 15.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if key is not None:
        headers["X-Internal-Key"] = key
    req = Request(url=url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        status = exc.code
    except URLError as exc:
        return {"status": None, "latency_ms": (time.perf_counter() - started) * 1000, "error": str(exc.reason), "body": None}
    elapsed = (time.perf_counter() - started) * 1000
    try:
        body = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        body = {"_non_json": raw[:500]}
    return {"status": status, "latency_ms": elapsed, "body": body}


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(x["latency_ms"]) for x in samples if x.get("latency_ms") is not None]
    p95 = statistics.quantiles(latencies, n=100, method="inclusive")[94] if len(latencies) >= 2 else (latencies[0] if latencies else None)
    return {
        "requests": len(samples),
        "successes": sum(x.get("status") == 200 for x in samples),
        "failures": sum(x.get("status") != 200 for x in samples),
        "min_ms": min(latencies) if latencies else None,
        "median_ms": statistics.median(latencies) if latencies else None,
        "mean_ms": statistics.mean(latencies) if latencies else None,
        "p95_ms": p95,
        "max_ms": max(latencies) if latencies else None,
    }


def prediction_contract(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {"valid": False, "reason": "response is not a JSON object"}
    required = ["risk_score", "risk_level", "anomaly_score", "is_anomaly", "feature_importance", "model_version"]
    missing = [k for k in required if k not in body]
    valid = not missing and body.get("model_version") == EXPECTED_MODEL_VERSION and body.get("risk_level") in {"low", "moderate", "high"}
    score = body.get("risk_score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
        valid = False
    return {"valid": valid, "missing": missing, "model_version": body.get("model_version"), "risk_score": score, "risk_level": body.get("risk_level")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Example: https://service--project.code.run")
    parser.add_argument("--output", default="DEPLOY-AI-PROBE-RUNTIME.json")
    parser.add_argument("--skip-concurrency", action="store_true")
    args = parser.parse_args()

    key = os.environ.get("AI_SERVICE_KEY")
    if not key:
        raise SystemExit("AI_SERVICE_KEY must be set in the environment. The probe never prints it.")

    base = args.base_url.rstrip("/")
    if key == KNOWN_DEVELOPMENT_KEY and not is_local_target(base):
        raise SystemExit("Refusing to probe a non-local deployment with the known development AI_SERVICE_KEY.")

    report: dict[str, Any] = {"base_url": base, "key_redacted": True, "payload": PAYLOAD}

    health = request_json("GET", base + "/health")
    report["health"] = health
    health_body = health.get("body") if isinstance(health, dict) else None
    report["health_contract_ok"] = bool(
        health.get("status") == 200
        and isinstance(health_body, dict)
        and health_body.get("status") == "ok"
        and health_body.get("model_loaded") is True
        and health_body.get("model_version") == EXPECTED_MODEL_VERSION
        and health_body.get("eligible_age_range") == EXPECTED_AGE_RANGE
    )

    report["auth"] = {
        "missing": request_json("POST", base + "/predict", payload=PAYLOAD),
        "invalid": request_json("POST", base + "/predict", key="__cardiosense_probe_invalid__", payload=PAYLOAD),
    }

    warmup = request_json("POST", base + "/predict", key=key, payload=PAYLOAD)
    report["warmup"] = warmup
    report["warmup_contract"] = prediction_contract(warmup.get("body"))

    sequential = [request_json("POST", base + "/predict", key=key, payload=PAYLOAD) for _ in range(10)]
    report["sequential"] = {"summary": summarize(sequential), "samples": sequential}
    successful_bodies = [x.get("body") for x in sequential if x.get("status") == 200]
    report["sequential"]["outputs_identical"] = bool(successful_bodies) and all(b == successful_bodies[0] for b in successful_bodies)
    report["sequential"]["contracts_valid"] = all(prediction_contract(b).get("valid") for b in successful_bodies) and len(successful_bodies) == 10

    if not args.skip_concurrency:
        concurrent: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(request_json, "POST", base + "/predict", key=key, payload=PAYLOAD) for _ in range(10)]
            for future in as_completed(futures):
                concurrent.append(future.result())
        report["concurrency_2"] = {"summary": summarize(concurrent), "samples": concurrent}
        bodies = [x.get("body") for x in concurrent if x.get("status") == 200]
        report["concurrency_2"]["contracts_valid"] = all(prediction_contract(b).get("valid") for b in bodies) and len(bodies) == 10

    auth_missing = report["auth"]["missing"].get("status") in {401, 403}
    auth_invalid = report["auth"]["invalid"].get("status") in {401, 403}
    seq_ok = report["sequential"]["summary"]["successes"] == 10 and report["sequential"]["contracts_valid"]
    conc_ok = True if args.skip_concurrency else report["concurrency_2"]["summary"]["successes"] == 10 and report["concurrency_2"]["contracts_valid"]
    report["probe_pass"] = bool(report["health_contract_ok"] and auth_missing and auth_invalid and report["warmup_contract"]["valid"] and seq_ok and conc_ok)

    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "probe_pass": report["probe_pass"],
        "health_contract_ok": report["health_contract_ok"],
        "missing_key_status": report["auth"]["missing"].get("status"),
        "invalid_key_status": report["auth"]["invalid"].get("status"),
        "warmup_ms": warmup.get("latency_ms"),
        "sequential": report["sequential"]["summary"],
        "concurrency_2": None if args.skip_concurrency else report["concurrency_2"]["summary"],
        "output": str(output),
    }, indent=2))
    return 0 if report["probe_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
