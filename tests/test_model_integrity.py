"""CPU-only validation of public model manifests and recursive code bindings."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from corpus_classifier import storage


class ModelIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('classifier_config.json', 'config.json', 'head_config.json',
                     'tokenizer.json', 'tokenizer_config.json', 'configuration_corpus_classifier.py',
                     'modeling_corpus_classifier.py', 'corpus_processing.py', 'model-00001-of-00001.safetensors'):
            (self.root / name).write_text('{}')
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {'encoder.weight': 'model-00001-of-00001.safetensors'}}))
        self.seal()

    def seal(self):
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': storage.digest(p)}
                 for p in self.root.iterdir() if p.name != 'MODEL_MANIFEST.json'}
        (self.root / 'MODEL_MANIFEST.json').write_text(json.dumps({
            'format': 'corpus-classifier-model-v2', 'files': files}))

    def test_v2_manifest_and_code_tampering(self):
        self.assertEqual(storage.verify_model(self.root), storage.digest(self.root / 'MODEL_MANIFEST.json'))
        (self.root / 'modeling_corpus_classifier.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            storage.verify_model(self.root)

    def test_index_requires_all_shards_and_rejects_traversal(self):
        for target in ('missing.safetensors', '../outside.safetensors'):
            (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': target}}))
            self.seal()
            with self.assertRaises(ValueError):
                storage.verify_model(self.root)

    def test_manifest_cannot_omit_model_code_or_shard_index(self):
        for name in ('modeling_corpus_classifier.py', 'model.safetensors.index.json'):
            self.seal()
            manifest = storage.load(self.root / 'MODEL_MANIFEST.json')
            del manifest['files'][name]
            (self.root / 'MODEL_MANIFEST.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'missing required'):
                storage.verify_model(self.root)

    def test_code_binding_covers_nested_custom_model(self):
        package = self.root / 'package'
        (package / 'hf_model').mkdir(parents=True)
        (package / 'storage.py').write_text('# mock storage')
        code = package / 'hf_model/model.py'
        code.write_text('first')
        with patch.object(storage, '__file__', str(package / 'storage.py')):
            first = storage.code_digest()
            code.write_text('second')
            self.assertNotEqual(first, storage.code_digest())
