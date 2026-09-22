"""Apache-2.0. Native Transformers configuration for KETI-NLP's classifier."""
from copy import deepcopy
from transformers import PretrainedConfig


class CorpusClassifierConfig(PretrainedConfig):
    model_type = "keti_corpus_classifier"

    def __init__(self, backbone_config=None, lora_config=None, head_config=None,
                 input_max_length=768, pooling="last", root_threshold=0.5,
                 exact_threshold=0.95, **kwargs):
        self.backbone_config = deepcopy(backbone_config or {})
        self.lora_config = deepcopy(lora_config or {})
        self.head_config = deepcopy(head_config or {})
        self.input_max_length = input_max_length
        self.pooling = pooling
        self.root_threshold = root_threshold
        self.exact_threshold = exact_threshold
        if self.head_config:
            labels = self.head_config["output_labels"]
            kwargs["num_labels"] = len(labels)
            kwargs["id2label"] = dict(enumerate(labels))
            kwargs["label2id"] = {label: index for index, label in enumerate(labels)}
        super().__init__(**kwargs)


CorpusClassifierConfig.register_for_auto_class()
