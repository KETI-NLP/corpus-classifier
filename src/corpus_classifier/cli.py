"""Explicit download command; all preparation, inference and export are offline."""
import argparse
import json
import os
from pathlib import Path
import re
import sys


def offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    def guard(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("Network access is disabled during local classification")
    sys.addaudithook(guard)


def parser():
    p = argparse.ArgumentParser(description="Frozen single-pass corpus classifier")
    sub = p.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download one pinned Hugging Face model snapshot")
    download.add_argument("--repo-id", required=True)
    download.add_argument("--revision", required=True, help="Full 40-character Hub commit hash")
    download.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify", help="Verify all model payload hashes")
    check.add_argument("--model", type=Path, required=True)
    prepare = sub.add_parser("prepare", help="Normalize JSONL/gzip/zstd/Parquet into immutable shards")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--shard-rows", type=int, default=4096)
    prepare.add_argument("--id-field", default="sample_id")
    prepare.add_argument("--text-field", default="text_window")
    prepare.add_argument("--max-record-chars", type=int, default=8_000_000)
    for command in ("run", "_worker"):
        run = sub.add_parser(command, help="Run/resume multi-GPU classification" if command == "run" else "Internal subprocess worker")
        run.add_argument("--manifest", type=Path, required=True)
        run.add_argument("--model", type=Path, required=True)
        run.add_argument("--output", type=Path, required=True)
        if command == "run":
            run.add_argument("--devices", default="0", help="CUDA device IDs/UUIDs, e.g. 0,1,2,3")
            run.add_argument("--batch-size", type=int, default=32)
        else:
            run.add_argument("--partition", type=int, required=True)
            run.add_argument("--partitions", type=int, required=True)
    export = sub.add_parser("export", help="Export all completed shards to normalized Parquet tables")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--predictions", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "download":
        if not re.fullmatch(r"[a-fA-F0-9]{40}", args.revision):
            raise ValueError("Pin --revision to a full 40-character Hub commit hash")
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=args.output)
        from .storage import verify_model
        result = dict(model=str(path), revision=args.revision, model_sha256=verify_model(path))
    else:
        offline()
        if args.command == "verify":
            from .storage import verify_model
            result = dict(status="passed", model_sha256=verify_model(args.model))
        elif args.command == "prepare":
            from .prepare import prepare
            result = prepare(args.input, args.output, shard_rows=args.shard_rows, id_field=args.id_field,
                             text_field=args.text_field, max_record_chars=args.max_record_chars)
        elif args.command == "run":
            from .bulk import run
            result = run(args.manifest, args.model, args.output, args.devices, args.batch_size)
        elif args.command == "_worker":
            from .bulk import process_partition
            from .storage import load, locked, digest
            binding = load(args.output / "run_binding.json")
            if digest(args.model / "MODEL_MANIFEST.json") != binding["model_sha256"]:
                raise ValueError("Model manifest changed after launcher verification")
            with locked(args.output / f"partition-{args.partition}.lock"):
                result = process_partition(args.manifest, args.model, args.output, args.partition, args.partitions, binding)
        elif args.command == "export":
            from .export import export
            result = export(args.manifest, args.predictions, args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
