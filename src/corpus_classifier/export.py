"""Export verified JSONL predictions to normalized, per-shard Parquet tables."""
import json
import os
from pathlib import Path
import shutil

from .bulk import completed, job_manifest
from .prepare import read_records
from .storage import digest, load, locked, sync_dir, write


def export(manifest_path, predictions, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    manifest_path, predictions, output = Path(manifest_path), Path(predictions), Path(output)
    binding = load(predictions / "run_binding.json")
    if binding["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Predictions do not belong to this input manifest")
    job = job_manifest(manifest_path)
    schemas = {
        "documents": pa.schema([(k, t) for k, t in [
            ("sample_id", pa.string()), ("roots", pa.list_(pa.string())), ("exact", pa.list_(pa.string())),
            ("source_family", pa.string()), ("source_locator_json", pa.string()), ("document_state_json", pa.string()),
            ("auxiliary_primary", pa.string()), ("input_scope", pa.string()), ("input_tokens", pa.int64()),
            ("visible_tokens", pa.int64()), ("truncated", pa.bool_()), ("empty_exact", pa.bool_()), ("raw_windowed", pa.bool_()),
            ("stored_text_sha256", pa.string()), ("model_input_sha256", pa.string()), ("token_ids_sha256", pa.string())]]),
        "exact_membership": pa.schema([("sample_id", pa.string()), ("code", pa.string()), ("score", pa.float64())]),
        "root_membership": pa.schema([("sample_id", pa.string()), ("code", pa.string()), ("score", pa.float64())]),
        "review_queue": pa.schema([("sample_id", pa.string()), ("reasons", pa.list_(pa.string()))]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with locked(output.with_name(output.name + ".export.lock")):
        if output.exists():
            raise FileExistsError(output)
        temp = output.with_name(output.name + ".exporting")
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        for name in schemas:
            (temp / name).mkdir()
        counts = dict.fromkeys(schemas, 0)
        for item in job["shards"]:
            completed(predictions / item["id"], item, binding)
            writers = {name: pq.ParquetWriter(temp / name / (item["id"] + ".parquet"), schema, compression="zstd") for name, schema in schemas.items()}
            buffers = {name: [] for name in schemas}
            def flush():
                for name in schemas:
                    if buffers[name]:
                        writers[name].write_table(pa.Table.from_pylist(buffers[name], schema=schemas[name]))
                        counts[name] += len(buffers[name]); buffers[name].clear()
            try:
                for row in read_records(predictions / item["id"] / "predictions.jsonl"):
                    doc = {key: row.get(key) for key in schemas["documents"].names}
                    doc["source_locator_json"] = json.dumps(row.get("source_locator"), ensure_ascii=False)
                    doc["document_state_json"] = json.dumps(row.get("document_state"), ensure_ascii=False)
                    buffers["documents"].append(doc)
                    for kind, field in (("root", "roots"), ("exact", "exact")):
                        buffers[kind + "_membership"].extend(dict(sample_id=row["sample_id"], code=c, score=row[kind + "_probabilities"][c]) for c in row[field])
                    reasons = [name for name, flag in (("no_exact_label", row["empty_exact"]), ("token_truncated", row["truncated"]), ("raw_windowed", row.get("raw_windowed"))) if flag]
                    if reasons:
                        buffers["review_queue"].append(dict(sample_id=row["sample_id"], reasons=reasons))
                    if len(buffers["documents"]) >= 256:
                        flush()
                flush()
            finally:
                for writer in writers.values():
                    writer.close()
            # Detect concurrently changed predictions before publication.
            completed(predictions / item["id"], item, binding)
        if counts["documents"] != job["rows"]:
            raise ValueError("Export row count mismatch")
        files = {}
        for p in temp.rglob("*.parquet"):
            with p.open("rb") as stream:
                os.fsync(stream.fileno())
            files[p.relative_to(temp).as_posix()] = dict(bytes=p.stat().st_size, sha256=digest(p))
        write(temp / "export_receipt.json", dict(binding=binding, counts=counts, files=files))
        for name in schemas:
            sync_dir(temp / name)
        sync_dir(temp); temp.rename(output); sync_dir(output.parent)
    return dict(status="complete", counts=counts, output=str(output))
