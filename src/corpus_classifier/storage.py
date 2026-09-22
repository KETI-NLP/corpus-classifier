"""Content checks and Linux atomic-directory commits for immutable jobs."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def locked(path, blocking=False):
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def child(root, relative):
    """Reject traversal and external symlinks in an artifact manifest."""
    p = PurePosixPath(relative)
    if p.is_absolute() or not p.parts or ".." in p.parts or "\\" in relative:
        raise ValueError("Invalid artifact path: " + relative)
    target = Path(root) / relative
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact escapes its directory: " + relative)
    return target


def verify_files(root, registry):
    for name, entry in registry.items():
        path = child(root, name)
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("Artifact checksum mismatch: " + name)


def code_digest():
    root = Path(__file__).parent
    files = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*.py"))}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def verify_model(model_dir):
    model_dir = Path(model_dir).resolve()
    manifest = load(model_dir / "MODEL_MANIFEST.json")
    if manifest.get("format") not in ("corpus-classifier-model-v1", "corpus-classifier-model-v2"):
        raise ValueError("Unsupported model package")
    verify_files(model_dir, manifest["files"])
    if manifest["format"] == "corpus-classifier-model-v2":
        required = {"classifier_config.json", "config.json", "head_config.json", "tokenizer.json", "tokenizer_config.json",
                    "configuration_corpus_classifier.py", "modeling_corpus_classifier.py", "corpus_processing.py"}
        weights = list(model_dir.glob("*.safetensors"))
        if not weights:
            raise ValueError("Missing Transformers model weights")
        required.update(p.name for p in weights)
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            weight_map = load(index).get("weight_map", {})
            if not weight_map:
                raise ValueError("Empty Transformers shard index")
            required.add(index.name)
            for name in set(weight_map.values()):
                path = child(model_dir, name)
                if not name.endswith(".safetensors") or not path.is_file():
                    raise ValueError("Missing indexed model shard: " + name)
                required.add(name)
        elif not (model_dir / "model.safetensors").is_file():
            raise ValueError("Missing Transformers shard index")
        for path in model_dir.glob("*.py"):
            required.add(path.name)
    else:
        required = {"classifier_config.json", "model/adapter/head_config.json",
                    "model/adapter/adapter_model.safetensors", "model/adapter/adapter_config.json",
                    "model/base/config.json", "model/adapter/tokenizer.json"}
    if not required <= manifest["files"].keys():
        raise ValueError("Model manifest is missing required payloads")
    for folder in ("model", "taxonomy"):
        for path in (model_dir / folder).rglob("*"):
            if path.is_file() and path.relative_to(model_dir).as_posix() not in manifest["files"]:
                raise ValueError("Unmanifested model file: " + str(path))
    return digest(model_dir / "MODEL_MANIFEST.json")
