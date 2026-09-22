"""Apache-2.0. Full-state Transformers model; original LoRA operations stay unmerged."""
from types import MethodType
from copy import deepcopy
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, PreTrainedModel, Qwen3_5Config
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_corpus_classifier import CorpusClassifierConfig
from .corpus_processing import clean_metadata, document_text, materialize


def last_token_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                       inputs_embeds=None, labels=None, return_dict=None, **kwargs):
    if input_ids is None or inputs_embeds is not None or labels is not None:
        raise ValueError("Explicit token IDs are required; this export is for inference")
    result = self.model(input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, use_cache=False, return_dict=True)
    pad = self.config.get_text_config().pad_token_id
    if pad is None:
        raise ValueError("The frozen tokenizer padding ID is required")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
    indices = ((input_ids != pad).to(torch.int32) * positions).argmax(-1)
    hidden = result.last_hidden_state[torch.arange(input_ids.shape[0], device=input_ids.device), indices]
    return SequenceClassifierOutput(logits=self.score(hidden))


class CorpusClassifierForSequenceClassification(PreTrainedModel):
    config_class = CorpusClassifierConfig
    base_model_prefix = "encoder"
    _supports_sdpa = True
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        if (config.input_max_length, config.pooling, config.root_threshold, config.exact_threshold) != (768, "last", .5, .95):
            raise ValueError("Unsupported classifier inference contract")
        head = config.head_config
        if (len(head.get("roots", [])), len(head.get("exact", [])), len(head.get("output_labels", []))) != (33, 1029, 2100):
            raise ValueError("Expected the frozen 33/1029/2100 label layout")
        base_config = Qwen3_5Config.from_dict(config.backbone_config)
        base_config.num_labels = 2100
        base_config.id2label = config.id2label
        base_config.label2id = config.label2id
        base = AutoModelForSequenceClassification.from_config(base_config, attn_implementation="sdpa")
        self.encoder = get_peft_model(base, LoraConfig(**deepcopy(config.lora_config)), autocast_adapter_dtype=False)
        classifier = self.encoder.get_base_model()
        classifier.forward = MethodType(last_token_forward, classifier)
        for module in self.encoder.modules():
            cfg = getattr(module, "config", None)
            if cfg is not None and hasattr(cfg, "use_cache"):
                cfg.use_cache = False
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                labels=None, return_dict=None, **kwargs):
        if labels is not None:
            raise ValueError("Joint partial-label training requires a separate masked loss")
        if input_ids is None or input_ids.ndim != 2 or not all(input_ids.shape):
            raise ValueError("Nonempty [batch, sequence] input_ids are required")
        if input_ids.shape[1] > self.config.input_max_length:
            raise ValueError("Input exceeds 768 tokens; use model.prepare_inputs()")
        # The frozen runner casts floating buffers (including rotary frequencies)
        # together with parameters. Transformers restores nonpersistent buffers in
        # their constructor dtype, so reproduce that conversion once per dtype.
        if getattr(self, "_inference_buffer_dtype", None) != self.dtype:
            for module in self.encoder.modules():
                for name, value in list(module._buffers.items()):
                    if value is not None and value.is_floating_point() and value.dtype != self.dtype:
                        module._buffers[name] = value.to(dtype=self.dtype)
            self._inference_buffer_dtype = self.dtype
        logits = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              position_ids=position_ids, return_dict=True).logits
        if return_dict is False:
            return (logits,)
        return SequenceClassifierOutput(logits=logits)

    def prepare_inputs(self, tokenizer, texts, metadata=None):
        """Apply the frozen metadata/body formatting and token-window policy."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Provide nonempty text strings")
        if metadata is None:
            metadata = [None] * len(texts)
        if len(metadata) != len(texts):
            raise ValueError("Metadata count must match the text count")
        formatted = [document_text(dict(text_window=text, metadata_compact=clean_metadata(meta))) for text, meta in zip(texts, metadata)]
        return tokenizer(formatted, padding=True, truncation=True,
                         max_length=self.config.input_max_length, return_tensors="pt")

    def decode_logits(self, logits, sample_ids=None):
        """Decode joint heads; generic pipeline softmax/sigmoid is insufficient."""
        if logits.ndim != 2 or logits.shape[1] != 2100 or not torch.isfinite(logits).all():
            raise ValueError("Expected finite [batch, 2100] logits")
        if sample_ids is None:
            sample_ids = [str(i) for i in range(len(logits))]
        if len(sample_ids) != len(logits):
            raise ValueError("ID count must match logits")
        results = materialize(logits, [{"sample_id": sid} for sid in sample_ids], self.config.head_config)
        for result in results:
            roots = result.pop("root_scores")
            exact = result.pop("exact_scores")
            result["root_probabilities"] = {c: roots[self.config.head_config["roots"].index(c)] for c in result["roots"]}
            result["exact_probabilities"] = {c: exact[self.config.head_config["exact"].index(c)] for c in result["exact"]}
        return results


CorpusClassifierForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
