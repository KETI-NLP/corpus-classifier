import gzip
import json
from pathlib import Path
import tempfile
import unittest

from corpus_classifier.bulk import bind, completed, job_manifest, process_partition
from corpus_classifier.prepare import prepare
from corpus_classifier.storage import child, code_digest, digest, load, locked, verify_files


def prediction(row):
    return dict(sample_id=row["sample_id"], roots=["SUB.A"], exact=["SUB.A.X"],
                root_probabilities={"SUB.A": .8}, exact_probabilities={"SUB.A.X": .96},
                auxiliary_primary="SUB.A.X", document_state={"topic_structure": "single_topic"},
                input_tokens=12, visible_tokens=12, truncated=False, empty_exact=False,
                raw_windowed=True, source_family="synthetic", input_scope="supplied_window_prefix_768")


class FakeModel:
    def predict(self, rows):
        return [prediction(r) for r in rows]


class BulkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.jsonl"
        self.rows = [dict(sample_id=f"synthetic-{i}", text_window=f"Document {i}", raw_windowed=True) for i in range(7)]
        self.source.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        self.job = self.root / "job"
        self.output = self.root / "predictions"

    def setup_job(self):
        prepare(self.source, self.job, shard_rows=3)
        self.path = self.job / "manifest.json"
        self.binding = dict(manifest_sha256=digest(self.path), code_sha256=code_digest(), model_sha256="synthetic-test-only", batch_size=2)
        bind(self.output, self.binding)

    def worker(self, **kwargs):
        return process_partition(self.path, self.root / "unused-model", self.output, 0, 1, self.binding, factory=FakeModel, **kwargs)

    def test_resume_after_commit_does_not_repeat_completed_shard(self):
        self.setup_job()
        def fault(stage, item):
            if stage == "after_commit":
                raise RuntimeError("simulated death")
        with self.assertRaises(RuntimeError):
            self.worker(fault=fault)
        first = self.output / "shard-000000/predictions.jsonl"
        original = (digest(first), first.stat().st_mtime_ns)
        result = self.worker()
        self.assertEqual((result["processed_rows"], result["skipped_rows"]), (4, 3))
        self.assertEqual(original, (digest(first), first.stat().st_mtime_ns))
        def no_load():
            raise AssertionError("No-op loaded model")
        result = process_partition(self.path, self.root, self.output, 0, 1, self.binding, factory=no_load)
        self.assertEqual(result["model_forwards"], 0)

    def test_uncommitted_work_is_recreated(self):
        self.setup_job()
        def fault(stage, item):
            if stage == "before_commit":
                raise RuntimeError("simulated death")
        with self.assertRaises(RuntimeError):
            self.worker(fault=fault)
        self.assertFalse((self.output / "shard-000000").exists())
        self.assertEqual(self.worker()["processed_rows"], 7)
        ids = [r["sample_id"] for p in sorted(self.output.glob("shard-*/predictions.jsonl")) for r in map(json.loads, p.read_text().splitlines())]
        self.assertEqual(ids, [r["sample_id"] for r in self.rows])

    def test_input_corruption_rejected_before_model_load(self):
        self.setup_job()
        p = self.job / "inputs/shard-000000.jsonl"
        p.write_text(p.read_text().replace("Document", "Changed!"))
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.worker()

    def test_output_corruption_rejected(self):
        self.setup_job(); self.worker()
        (self.output / "shard-000000/predictions.jsonl").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self.worker()

    def test_manifest_binding_and_code_change_rejected(self):
        self.setup_job()
        with self.assertRaises(ValueError):
            bind(self.output, {**self.binding, "batch_size": 4})
        with self.assertRaisesRegex(ValueError, "runner changed"):
            process_partition(self.path, self.root, self.output, 0, 1, {**self.binding, "code_sha256": "different"}, factory=FakeModel)

    def test_global_duplicate_ids_never_publish_manifest(self):
        with self.source.open("a") as stream:
            stream.write(json.dumps(self.rows[0]) + "\n")
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            prepare(self.source, self.job, shard_rows=3)
        self.assertFalse((self.job / "manifest.json").exists())

    def test_formats_and_custom_field_names(self):
        import zstandard
        import pyarrow as pa
        import pyarrow.parquet as pq
        raw = self.source.read_bytes()
        gz, zst, parquet = (self.root / name for name in ("data.jsonl.gz", "data.jsonl.zst", "data.parquet"))
        gz.write_bytes(gzip.compress(raw))
        # Two independent zstd frames must both be consumed.
        split = raw.find(b"\n") + 1
        zst.write_bytes(zstandard.ZstdCompressor().compress(raw[:split]) + zstandard.ZstdCompressor().compress(raw[split:]))
        pq.write_table(pa.Table.from_pylist(self.rows), parquet)
        for i, source in enumerate((self.source, gz, zst, parquet)):
            output = self.root / f"format-{i}"
            self.assertEqual(prepare(source, output, shard_rows=3)["documents"], 7)
            self.assertEqual([r["sample_id"] for p in sorted((output / "inputs").glob("*.jsonl")) for r in map(json.loads, p.read_text().splitlines())], [r["sample_id"] for r in self.rows])
        self.source.write_text(json.dumps({"id": "one", "text": "Text"}) + "\n")
        self.assertEqual(prepare(self.source, self.job, id_field="id", text_field="text")["documents"], 1)

    def test_partitions_cover_input_once(self):
        self.setup_job()
        for i in range(2):
            process_partition(self.path, self.root, self.output, i, 2, self.binding, factory=FakeModel)
        self.assertEqual(sum(completed(self.output / s["id"], s, self.binding)["rows"] for s in job_manifest(self.path)["shards"]), 7)

    def test_lock_and_manifest_path_escape(self):
        with locked(self.root / "test.lock"):
            with self.assertRaises(BlockingIOError):
                with locked(self.root / "test.lock"):
                    pass
        for path in ("../escape", "/absolute", "a/../../b"):
            with self.assertRaises(ValueError):
                child(self.root, path)

    def test_bad_record_and_oversized_text(self):
        self.source.write_text('{"sample_id":"id","text_window":""}\n')
        with self.assertRaises(ValueError):
            prepare(self.source, self.job)
        self.source.write_text(json.dumps(self.rows[0]) + "\n")
        with self.assertRaises(ValueError):
            prepare(self.source, self.job, max_record_chars=8)

    def test_parquet_export_retains_relations(self):
        self.setup_job(); self.worker()
        from corpus_classifier.export import export
        import pyarrow.dataset as ds
        result = export(self.path, self.output, self.root / "parquet")
        self.assertEqual(result["counts"], dict(documents=7, root_membership=7, exact_membership=7, review_queue=7))
        table = ds.dataset(self.root / "parquet/exact_membership", format="parquet").to_table()
        self.assertEqual(len(set(table["sample_id"].to_pylist())), 7)
        receipt = load(self.root / "parquet/export_receipt.json")
        verify_files(self.root / "parquet", receipt["files"])


if __name__ == "__main__":
    unittest.main()
