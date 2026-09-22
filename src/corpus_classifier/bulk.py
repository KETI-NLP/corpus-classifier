"""One process/model per GPU with validated, resumable shard commits."""
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .storage import child, code_digest, digest, load, locked, sync_dir, verify_model, write
from .prepare import read_records


def job_manifest(path):
    job = load(path)
    if job.get("format") != "corpus-classifier-job-v1" or not job.get("shards"):
        raise ValueError("Unsupported or empty job manifest")
    ids = set()
    for item in job["shards"]:
        if not re.fullmatch(r"shard-[0-9]{6,}", item["id"]) or item["id"] in ids or item["rows"] < 1:
            raise ValueError("Invalid/duplicate shard")
        if item["path"] != f"inputs/{item['id']}.jsonl":
            raise ValueError("Unexpected shard input path")
        ids.add(item["id"])
        child(Path(path).parent, item["path"])
    if sum(s["rows"] for s in job["shards"]) != job["rows"]:
        raise ValueError("Manifest row count mismatch")
    return job


def completed(folder, item, binding):
    receipt = load(folder / "receipt.json")
    if receipt["binding"] != binding or receipt["input_sha256"] != item["sha256"] or receipt["rows"] != item["rows"]:
        raise ValueError("Completed shard provenance mismatch")
    if digest(folder / "predictions.jsonl") != receipt["predictions_sha256"]:
        raise ValueError("Completed shard is corrupt")
    return receipt


def bind(output, binding):
    output.mkdir(parents=True, exist_ok=True)
    with locked(output / "binding.lock", blocking=True):
        target = output / "run_binding.json"
        if target.exists():
            if load(target) != binding:
                raise ValueError("Output belongs to another input/model/code/batch configuration")
        else:
            temp = output / ".binding.tmp"
            if temp.exists():
                temp.unlink()
            write(temp, binding)
            temp.rename(target); sync_dir(output)


def process_partition(manifest_path, model_dir, output, partition, partitions, binding, factory=None, fault=None):
    """Internal worker; tests inject a deterministic CPU model and commit faults."""
    path, output = Path(manifest_path), Path(output)
    if partitions < 1 or not 0 <= partition < partitions:
        raise ValueError("Invalid partition")
    if digest(path) != binding["manifest_sha256"] or code_digest() != binding["code_sha256"]:
        raise ValueError("Input manifest or runner changed after preflight")
    job = job_manifest(path)
    if load(output / "run_binding.json") != binding:
        raise ValueError("Run binding changed")
    pending, skipped = [], 0
    for index, item in enumerate(job["shards"]):
        if index % partitions != partition:
            continue
        source = child(path.parent, item["path"])
        if source.stat().st_size != item["bytes"] or digest(source) != item["sha256"]:
            raise ValueError("Input shard checksum mismatch")
        final = output / item["id"]
        if final.exists():
            completed(final, item, binding)
            skipped += item["rows"]
        else:
            pending.append(item)
    if not pending:
        return dict(status="already_complete", processed_rows=0, skipped_rows=skipped, model_forwards=0)
    if factory is None:
        from .model import Classifier
        import torch
        torch.set_num_threads(2)
        factory = lambda: Classifier(model_dir, _verified=True)
    model = None
    processed = forwards = 0
    for item in pending:
        with locked(output / (item["id"] + ".lock")):
            final = output / item["id"]
            if final.exists():
                completed(final, item, binding)
                skipped += item["rows"]
                continue
            if model is None:
                model = factory()
            temp = output / ("." + item["id"] + ".inprogress")
            if temp.exists():
                shutil.rmtree(temp)
            temp.mkdir()
            start = time.perf_counter()
            source = child(path.parent, item["path"])
            count, seen = 0, set()
            # Stream batches, not a whole corpus or a whole shard, into memory.
            records = read_records(source, max_record_chars=100_000_000)
            with (temp / "predictions.jsonl").open("x", encoding="utf-8") as stream:
                while batch := list(itertools.islice(records, binding["batch_size"])):
                    ids = [r["sample_id"] for r in batch]
                    if len(set(ids)) != len(ids) or seen.intersection(ids):
                        raise ValueError("Duplicate document in shard")
                    seen.update(ids)
                    predictions = model.predict(batch)
                    if [p["sample_id"] for p in predictions] != ids:
                        raise ValueError("Prediction identity/order mismatch")
                    forwards += 1
                    for result in predictions:
                        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    count += len(batch)
                stream.flush(); os.fsync(stream.fileno())
            if count != item["rows"] or digest(source) != item["sha256"]:
                raise ValueError("Input changed during classification")
            receipt = dict(binding=binding, rows=count, input_sha256=item["sha256"],
                           predictions_sha256=digest(temp / "predictions.jsonl"), seconds=time.perf_counter() - start)
            write(temp / "receipt.json", receipt); sync_dir(temp)
            if fault is not None:
                fault("before_commit", item)
            temp.rename(final); sync_dir(output)
            if fault is not None:
                fault("after_commit", item)
            processed += count
            print(json.dumps(dict(committed=item["id"], rows=count, seconds=receipt["seconds"])), flush=True)
    return dict(status="complete", processed_rows=processed, skipped_rows=skipped, model_forwards=forwards)


def run(manifest_path, model_dir, output, devices="0", batch_size=32):
    path, model_dir, output = Path(manifest_path).resolve(), Path(model_dir).resolve(), Path(output).resolve()
    gpu_ids = [x.strip() for x in devices.split(",")]
    if not gpu_ids or any(not re.fullmatch(r"[0-9]+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9/-]+", x) for x in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("Use distinct comma-separated CUDA device IDs or GPU UUIDs")
    if batch_size < 1:
        raise ValueError("Positive batch size required")
    start = time.perf_counter()
    job = job_manifest(path)
    binding = dict(manifest_sha256=digest(path), model_sha256=verify_model(model_dir), code_sha256=code_digest(), batch_size=batch_size)
    bind(output, binding)
    with locked(output / "launcher.lock"):
        if all((output / s["id"]).exists() for s in job["shards"]):
            for item in job["shards"]:
                source = child(path.parent, item["path"])
                if digest(source) != item["sha256"]:
                    raise ValueError("Input shard checksum mismatch")
                completed(output / item["id"], item, binding)
            return dict(status="already_complete", skipped_rows=job["rows"], model_forwards=0)
        processes = []
        try:
            for index, gpu in enumerate(gpu_ids[:len(job["shards"])]):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
                command = [sys.executable, "-m", "corpus_classifier", "_worker", "--manifest", str(path), "--model", str(model_dir), "--output", str(output),
                           "--partition", str(index), "--partitions", str(min(len(gpu_ids), len(job["shards"])))]
                processes.append(subprocess.Popen(command, env=env))
            while processes:
                for proc in list(processes):
                    code = proc.poll()
                    if code is not None:
                        processes.remove(proc)
                        if code:
                            raise RuntimeError(f"GPU worker exited with code {code}; completed shards are retained")
                if processes:
                    time.sleep(.2)
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.terminate()
            for proc in processes:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
        for item in job["shards"]:
            completed(output / item["id"], item, binding)
        result = dict(status="complete", rows=job["rows"], seconds=time.perf_counter() - start, binding=binding)
        write(output / f"session-{time.time_ns()}.json", result)
        return result
