"""Local-only Transformers v2 loader with legacy v1 bundle compatibility."""
import hashlib
import json
import os
from pathlib import Path

from .storage import load, verify_model, digest
from .contracts import document_text, clean_metadata


class Classifier:
    def __init__(self, model_dir, device="cuda:0", *, _verified=False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
        from .pooling import install_pooling
        from .model_utils import reset_classifier_head, set_pad_token_id_deep, disable_use_cache

        self.directory = Path(model_dir).resolve()
        if not _verified:
            verify_model(self.directory)
        settings = load(self.directory / "classifier_config.json")
        if settings.get("format") == "corpus-classifier-transformers-v2":
            from .hf_model import CorpusClassifierConfig, CorpusClassifierForSequenceClassification
            if (settings["max_length"], settings["pooling"], settings["root_threshold"], settings["exact_threshold"]) != (768, "last", .5, .95):
                raise ValueError("Frozen inference configuration changed")
            self.config = load(self.directory / "head_config.json")
            implementation = Path(__file__).parent / "hf_model"
            for name in ("configuration_corpus_classifier.py", "modeling_corpus_classifier.py", "corpus_processing.py"):
                if digest(self.directory / name) != digest(implementation / name):
                    raise ValueError("Model code differs from this installed release; install the matching classifier version")
            self.device = torch.device(device)
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU required; this release uses bfloat16")
            native_config = CorpusClassifierConfig.from_pretrained(self.directory, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.directory, config=native_config, local_files_only=True, trust_remote_code=False)
            self.model = CorpusClassifierForSequenceClassification.from_pretrained(
                self.directory, config=native_config, local_files_only=True, dtype=torch.bfloat16)
            if self.model.config.head_config != self.config:
                raise ValueError("Model and sidecar label layouts differ")
            self.model.to(device=self.device, dtype=torch.bfloat16).eval()
            return
        expected = ("model/base", "model/adapter", "model/adapter/head_config.json", "last", 768, .5, .95)
        if tuple(settings.get(k) for k in ("base_model", "adapter", "head_config", "pooling", "max_length", "root_threshold", "exact_threshold")) != expected:
            raise ValueError("Frozen inference configuration changed")
        self.config = load(self.directory / settings["head_config"])
        labels = self.config["output_labels"]
        if (len(labels), len(self.config["exact"]), len(self.config["roots"])) != (2100, 1029, 33):
            raise ValueError("Unexpected classification head layout")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; this release uses bfloat16")
        self.tokenizer = AutoTokenizer.from_pretrained(self.directory / "model/adapter", local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(self.directory / "model/base", local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16)
        reset_classifier_head(model, len(labels), dict(enumerate(labels)), {c: i for i, c in enumerate(labels)})
        set_pad_token_id_deep(model, self.tokenizer.pad_token_id)
        self.model = PeftModel.from_pretrained(model, self.directory / "model/adapter", is_trainable=False, local_files_only=True)
        install_pooling(self.model, "last")
        disable_use_cache(self.model)
        self.model.to(device=self.device, dtype=torch.bfloat16).eval()

    def encode(self, rows):
        texts = []
        for row in rows:
            if not isinstance(row.get("sample_id"), str) or not row["sample_id"] or not isinstance(row.get("text_window"), str) or not row["text_window"].strip():
                raise ValueError("Nonempty sample_id and text_window strings required")
            texts.append(document_text(dict(text_window=row["text_window"], metadata_compact=clean_metadata(row.get("metadata_compact")))))
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate sample_id in batch")
        tokens = self.tokenizer(texts, truncation=False)["input_ids"]
        batch = self.tokenizer.pad({"input_ids": [t[:768] for t in tokens]}, padding=True, return_tensors="pt").to(self.device)
        return batch, [len(t) for t in tokens], texts

    def logits(self, rows):
        import torch
        batch, lengths, _ = self.encode(rows)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**batch).logits.float().cpu()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite model output")
        return logits, lengths

    def predict(self, rows):
        if not rows:
            return []
        import torch
        from .model_utils import materialize
        batch, lengths, texts = self.encode(rows)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**batch).logits.float().cpu()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite model output")
        results = materialize(logits, rows, self.config)
        ids = batch["input_ids"].cpu()
        masks = batch["attention_mask"].cpu().bool()
        for i, (result, row, text, length) in enumerate(zip(results, rows, texts, lengths)):
            roots = result.pop("root_scores")
            exact = result.pop("exact_scores")
            result["root_probabilities"] = {c: roots[self.config["roots"].index(c)] for c in result["roots"]}
            result["exact_probabilities"] = {c: exact[self.config["exact"].index(c)] for c in result["exact"]}
            result.update(input_tokens=length, visible_tokens=min(length, 768), truncated=length > 768,
                          empty_exact=not result["exact"], input_scope="supplied_window_prefix_768",
                          source_family=row.get("source_family", "unspecified"), source_locator=row.get("source_locator"),
                          stored_text_sha256=hashlib.sha256(row["text_window"].encode()).hexdigest(),
                          model_input_sha256=hashlib.sha256(text.encode()).hexdigest(),
                          token_ids_sha256=hashlib.sha256(json.dumps(ids[i][masks[i]].tolist(), separators=(",", ":")).encode()).hexdigest())
            # Optional upstream scope flags are provenance, never model features.
            result["raw_windowed"] = row.get("raw_windowed")
        return results
