#!/usr/bin/env python3
"""Public, zero-secret, actual GGUF smoke experiment for Qwen2.5 0.5B.
Only public source/model and aggregate evidence; no ECHOR-X private sources.
All measurements are restricted to the ephemeral GitHub-hosted Linux runner.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from urllib import request as urlrequest

import psutil
from huggingface_hub import hf_hub_download, model_info

REPO = "second-state/Qwen2.5-0.5B-Instruct-GGUF"
MODEL = "Qwen2.5-0.5B-Instruct-f16.gguf"
MODES = ("F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
THREADS = 4


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tasks():
    out = []
    for i in range(40):
        a, b = 7 + (i * 17) % 71, 3 + (i * 11) % 29
        out.append((f"en{i:03}", "en_math",
                    f"Calculate {a} + {b}. Answer only with the decimal number.", str(a+b)))
    for i in range(40):
        a, b = 13 + (i * 13) % 73, 5 + (i * 7) % 31
        out.append((f"ja{i:03}", "jp_math",
                    f"{a} と {b} の和を計算してください。数字のみで答えてください。", str(a+b)))
    return out


def available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def api(port, payload=None, endpoint="/health", timeout=15):
    body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    req = urlrequest.Request(f"http://127.0.0.1:{port}{endpoint}", data=body,
                             headers={"Content-Type": "application/json"})
    with urlrequest.build_opener(urlrequest.ProxyHandler({})).open(req, timeout=timeout) as r:
        response = r.read(1 << 20)
        if r.status != 200 or len(response) >= (1 << 20):
            raise RuntimeError("BAD_LOCAL_SERVER_RESPONSE")
        return json.loads(response)


def launch_and_evaluate(server, model, prompts):
    port = available_port()
    cmd = [str(server), "-m", str(model), "--host", "127.0.0.1",
           "--port", str(port), "-t", str(THREADS), "-c", "2048", "-ngl", "0",
           "--parallel", "1"]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    peak_rss = [0]
    active = threading.Event()
    active.set()

    def memory_sampler():
        try:
            main = psutil.Process(proc.pid)
            while active.is_set():
                try:
                    family = [main] + main.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in family if p.is_running())
                    peak_rss[0] = max(rss, peak_rss[0])
                except (psutil.Error, OSError):
                    pass
                time.sleep(.04)
        except psutil.Error:
            pass
    thread = threading.Thread(target=memory_sampler, daemon=True)
    thread.start()
    started = time.monotonic()
    records = {}
    try:
        while time.monotonic() - started < 100:
            if proc.poll() is not None:
                msg = proc.stderr.read(3000).decode("utf-8", "replace")
                raise RuntimeError("SERVER_DIED " + msg[:700])
            try:
                api(port, timeout=1.5)
                break
            except Exception:
                time.sleep(.25)
        else:
            raise RuntimeError("SERVER_STARTUP_TIMEOUT")
        init_ms = (time.monotonic() - started) * 1000

        def completion(prompt):
            payload = {"model": "public_qwen_smoke",
                       "messages": [{"role": "system", "content": "Answer only with one decimal integer."},
                                    {"role": "user", "content": prompt}],
                       "temperature": 0.0, "top_k": 1, "seed": 42,
                       "max_tokens": 24, "stream": False}
            started_req = time.monotonic_ns()
            answer = api(port, payload, endpoint="/v1/chat/completions", timeout=30)
            dt = (time.monotonic_ns() - started_req) / 1e6
            choices = answer.get("choices", [])
            if not choices or not isinstance(choices[0].get("message", {}).get("content"), str):
                raise RuntimeError("INVALID_CHAT_COMPLETION")
            return choices[0]["message"]["content"].strip(), dt

        for _, _, text, _ in prompts[:2]:
            completion(text)  # warm up
        for tid, cohort, text, gold in prompts:
            response, latency = completion(text)
            numeric = re.fullmatch(r"\s*([+-]?\d+)\s*", response)
            correct = bool(numeric and int(numeric.group(1)) == int(gold))
            records[tid] = {"cohort": cohort, "correct": correct,
                            "latency_ms": round(latency, 4)}
        return {"startup_ms": round(init_ms, 3), "peak_rss_bytes": peak_rss[0],
                "responses": records}
    finally:
        active.clear()
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=8)
        thread.join(timeout=2)
        if proc.stderr:
            proc.stderr.close()


def speed_bench(binary, model):
    cmd = [str(binary), "-m", str(model), "-p", "128", "-n", "32", "-r", "2",
           "-t", str(THREADS), "-ngl", "0", "-o", "jsonl"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=160)
    if r.returncode != 0:
        return {"status": "NOT_MEASURED", "reason": r.stderr[-500:]}
    parsed = []
    for line in r.stdout.splitlines():
        try:
            data = json.loads(line)
            parsed.append({k: data[k] for k in ("n_prompt", "n_gen", "avg_ts", "stddev_ts")
                           if k in data})
        except (ValueError, TypeError):
            pass
    return {"status": "MEASURED" if parsed else "NOT_MEASURED",
            "samples": parsed[:8]}


def main():
    root = Path(os.environ["RUNNER_TEMP"]).resolve() / "echor-public-gguf"
    root.mkdir()
    info = model_info(REPO)
    source = Path(hf_hub_download(repo_id=REPO, filename=MODEL,
                                  revision=info.sha, local_dir=root / "model"))
    if source.read_bytes()[:4] != b"GGUF":
        raise RuntimeError("BASELINE_NOT_GGUF")
    sha = digest(source)
    if source.stat().st_size < 900_000_000:
        raise RuntimeError("BASELINE_TOO_SMALL")
    bin_dir = Path(os.environ["GITHUB_WORKSPACE"]) / "build" / "bin"
    quant = bin_dir / "llama-quantize"
    server = bin_dir / "llama-server"
    bench = bin_dir / "llama-bench"
    for path in (quant, server, bench):
        if not path.is_file():
            raise RuntimeError("REQUIRED_BINARY_MISSING "+str(path))
    prompts = tasks()
    task_sha = hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()
    print("GGUF_PROVENANCE " + json.dumps({
        "repository": REPO, "model_revision": info.sha,
        "gguf_sha256": sha, "gguf_bytes": source.stat().st_size,
        "task_sha256": task_sha, "task_count": len(prompts),
        "runtime": "GitHub hosted Linux CPU 4 threads",
        "model_quantization_source": "third_party_f16_conversion",
        "source_verification": "SHA256_CONTENT_ONLY_NOT_ORIGINAL_CHECKPOINT_ATTESTATION",
        "production_grant": "DENY"}, sort_keys=True), flush=True)
    models = {"F16": source}
    for mode in MODES[1:]:
        target = root / f"Qwen2.5-0.5B-Instruct-{mode}.gguf"
        print("QUANT_START " + mode, flush=True)
        result = subprocess.run([str(quant), str(source), str(target), mode, str(THREADS)],
                                capture_output=True, text=True, timeout=240)
        if result.returncode != 0 or not target.is_file() or target.open("rb").read(4) != b"GGUF":
            raise RuntimeError(f"QUANT_FAILED_{mode} {result.stderr[-700:]}")
        models[mode] = target
        print("QUANT_OK " + mode + " sha256=" + digest(target)
              + " bytes=" + str(target.stat().st_size), flush=True)
    quality = {}
    for mode in MODES:
        print("EVAL_START " + mode, flush=True)
        model = models[mode]
        result = launch_and_evaluate(server, model, prompts)
        result["precision"] = mode
        result["model_sha256"] = digest(model)
        result["model_file_bytes"] = model.stat().st_size
        result["accuracy"] = sum(v["correct"] for v in result["responses"].values()) / len(prompts)
        latencies = sorted(v["latency_ms"] for v in result["responses"].values())
        result["p95_latency_ms"] = latencies[math.ceil(.95*len(latencies))-1]
        result["llama_bench"] = speed_bench(bench, model)
        quality[mode] = result
        print("GGUF_MODEL_RESULT "+json.dumps({k:v for k,v in result.items()
                                              if k != "responses"}, sort_keys=True), flush=True)
    baseline = quality["F16"]
    for mode in MODES[1:]:
        current = quality[mode]
        regress = [tid for tid, r in current["responses"].items()
                   if baseline["responses"][tid]["correct"] and not r["correct"]]
        per_task_slows = [tid for tid,r in current["responses"].items()
                          if r["latency_ms"] > 1.05*baseline["responses"][tid]["latency_ms"]]
        report = {"precision":mode, "baseline_precision":"F16",
                  "baseline_accuracy":baseline["accuracy"],
                  "candidate_accuracy":current["accuracy"],
                  "correct_to_wrong_count":len(regress),
                  "regression_task_ids":regress,
                  "individual_latency_regressions":len(per_task_slows),
                  "p95_latency_ratio":round(current["p95_latency_ms"]/baseline["p95_latency_ms"],5),
                  "peak_rss_ratio":round(current["peak_rss_bytes"]/baseline["peak_rss_bytes"],5),
                  "size_savings_pct":round(100*(1-current["model_file_bytes"]/baseline["model_file_bytes"]),5)}
        report["strict_holdout_zero_regression"] = len(regress)==0
        report["conditional_shadow_pass"] = (len(regress)==0
                                              and current["accuracy"]>=baseline["accuracy"]
                                              and report["p95_latency_ratio"]<=1.05
                                              and report["peak_rss_ratio"]<=1.0
                                              and report["size_savings_pct"]>=10
                                              and report["individual_latency_regressions"]==0)
        report["production_grant"] = "DENY"
        print("GGUF_PAIRED_RESULT "+json.dumps(report,sort_keys=True),flush=True)
    print("GGUF_EXPERIMENT_COMPLETED " + json.dumps({
        "real_model": True, "quality": "THIS_80_TASK_SMOKE_ONLY",
        "cpu_speed": "LLAMA_BENCH_HOSTED_RUNNER",
        "ram": "PEAK_RSS_LOCAL_PROCESS_FAMILY",
        "energy": "NOT_MEASURED", "population_noninferiority": "NOT_PROVEN",
        "production_grant": "DENY"}, sort_keys=True),flush=True)


if __name__ == "__main__":
    import math
    main()
