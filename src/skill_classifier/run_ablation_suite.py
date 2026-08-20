"""Run a resumable classifier ablation manifest with bounded parallelism."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def encode_override(key, value):
    if isinstance(value, bool):
        value = str(value).lower()
    elif isinstance(value, list):
        value = ",".join(map(str, value))
    return f"{key}={value}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    suite = yaml.safe_load(args.suite.read_text())
    output_dir = Path(suite["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_parallel = args.max_parallel or int(suite.get("max_parallel", 1))
    if max_parallel <= 0:
        raise ValueError("max_parallel must be positive")

    common = dict(suite.get("common", {}))
    wanted = set(args.only)
    pending = []
    for experiment in suite["experiments"]:
        exp_id = str(experiment["id"])
        if wanted and exp_id not in wanted:
            continue
        exp_dir = output_dir / exp_id
        if (exp_dir / "evaluation_summary.json").is_file():
            print(f"SKIP completed {exp_id}", flush=True)
            continue
        settings = common | dict(experiment.get("set", {}))
        command = [
            sys.executable,
            "-m",
            "skill_classifier.train",
            "--config",
            str(suite["base_config"]),
            "--exp_id",
            exp_id,
            "--set",
            f"output_dir={output_dir}",
            *[encode_override(key, value) for key, value in settings.items()],
        ]
        pending.append((experiment, command, exp_dir))

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    active = []
    failures = []
    while pending or active:
        while pending and len(active) < max_parallel:
            experiment, command, exp_dir = pending.pop(0)
            exp_dir.mkdir(parents=True, exist_ok=True)
            log_path = exp_dir / "run.log"
            stream = log_path.open("a", buffering=1)
            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            active.append((experiment, process, stream, log_path))
            print(
                f"START {experiment['id']} pid={process.pid} log={log_path}",
                flush=True,
            )
        time.sleep(2)
        still_active = []
        for experiment, process, stream, log_path in active:
            status = process.poll()
            if status is None:
                still_active.append((experiment, process, stream, log_path))
                continue
            stream.close()
            print(f"DONE {experiment['id']} exit={status}", flush=True)
            if status != 0:
                failures.append((experiment["id"], status, str(log_path)))
        active = still_active

    if failures:
        for exp_id, status, log_path in failures:
            print(f"FAILED {exp_id} exit={status} log={log_path}")
        raise SystemExit(1)
    print("Ablation suite complete", flush=True)


if __name__ == "__main__":
    main()
