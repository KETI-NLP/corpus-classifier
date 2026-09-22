"""Stream normalized records into portable, immutable JSONL shards."""
from contextlib import contextmanager
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3

from .contracts import clean_metadata
from .storage import digest, write, sync_dir, locked


@contextmanager
def text_stream(path):
    with Path(path).open("rb") as raw:
        if str(path).endswith(".gz"):
            with gzip.GzipFile(fileobj=raw) as decoded, io.TextIOWrapper(decoded, encoding="utf-8") as text:
                yield text
        elif str(path).endswith((".zst", ".zstd")):
            import zstandard
            with zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True) as decoded, io.TextIOWrapper(decoded, encoding="utf-8") as text:
                yield text
        else:
            with io.TextIOWrapper(raw, encoding="utf-8") as text:
                yield text


def read_records(path, max_record_chars=8_000_000):
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
        return
    with text_stream(path) as stream:
        number = 0
        while line := stream.readline(max_record_chars + 1):
            number += 1
            if len(line) > max_record_chars:
                raise ValueError(f"Input record too large: line {number}")
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at line {number}")
            yield row


def normalize(row, id_field, text_field, max_record_chars):
    sid, text = row.get(id_field), row.get(text_field)
    if not isinstance(sid, str) or not sid or not isinstance(text, str) or not text.strip():
        raise ValueError("Each record needs a nonempty string ID and text")
    if len(text) > max_record_chars:
        raise ValueError("Document text exceeds --max-record-chars")
    flag = row.get("raw_windowed")
    if flag is not None and type(flag) is not bool:
        raise ValueError("raw_windowed must be a boolean or null")
    return dict(sample_id=sid, text_window=text, metadata_compact=clean_metadata(row.get("metadata_compact")),
                source_family=row.get("source_family") or "unspecified", source_locator=row.get("source_locator"),
                raw_windowed=flag)


def prepare(input_path, output, *, shard_rows=4096, id_field="sample_id", text_field="text_window", max_record_chars=8_000_000):
    source, output = Path(input_path).resolve(), Path(output).resolve()
    if shard_rows < 1 or max_record_chars < 1:
        raise ValueError("Positive shard size and record limit required")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".preparing")
    with locked(output.with_name(output.name + ".prepare.lock")):
        if output.exists():
            raise FileExistsError("Job already exists; reuse its manifest or choose a new output")
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        (temp / "inputs").mkdir()
        before = digest(source)
        db = sqlite3.connect(temp / "ids.sqlite")
        db.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
        stream = None
        count = 0
        shards = []
        try:
            for row in read_records(source, max_record_chars):
                row = normalize(row, id_field, text_field, max_record_chars)
                db.execute("INSERT INTO ids VALUES (?)", (row["sample_id"],))
                if stream is None:
                    name = f"shard-{len(shards):06d}"
                    path = temp / "inputs" / (name + ".jsonl")
                    stream = path.open("x", encoding="utf-8")
                    chunk = 0
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
                chunk += 1
                if chunk == shard_rows:
                    stream.flush(); os.fsync(stream.fileno()); stream.close(); stream = None
                    shards.append(dict(id=name, path=f"inputs/{name}.jsonl", rows=chunk, bytes=path.stat().st_size, sha256=digest(path)))
                    db.commit()
            if stream is not None:
                stream.flush(); os.fsync(stream.fileno()); stream.close(); stream = None
                shards.append(dict(id=name, path=f"inputs/{name}.jsonl", rows=chunk, bytes=path.stat().st_size, sha256=digest(path)))
            db.commit()
            if not count:
                raise ValueError("Input contains no documents")
            if digest(source) != before:
                raise ValueError("Input file changed during preparation")
            write(temp / "manifest.json", dict(format="corpus-classifier-job-v1", rows=count, shards=shards,
                  source=dict(name=source.name, sha256=before), id_field=id_field, text_field=text_field,
                  max_length=768, input_scope="supplied_window_prefix_768", shard_rows=shard_rows))
        finally:
            if stream is not None:
                stream.close()
            db.close()
        (temp / "ids.sqlite").unlink()
        sync_dir(temp / "inputs"); sync_dir(temp)
        temp.rename(output); sync_dir(output.parent)
    return dict(documents=count, shards=len(shards), manifest=str(output / "manifest.json"))
