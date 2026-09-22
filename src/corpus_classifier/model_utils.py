from typing import Any, Dict
import torch
from .contracts import predict_sets, STATE_LABELS

def reset_classifier_head(model: torch.nn.Module, num_labels: int, id_to_label: Dict[int, str], label_to_id: Dict[str, int]) -> None:
    """Some local Qwen sequence-classification checkpoints ignore num_labels on load."""
    model.config.num_labels = num_labels
    model.config.id2label = dict(id_to_label)
    model.config.label2id = dict(label_to_id)
    if hasattr(model, "num_labels"):
        model.num_labels = num_labels
    score = getattr(model, "score", None)
    if score is None:
        return
    in_features = getattr(score, "in_features", None)
    if in_features is None and hasattr(score, "weight"):
        in_features = score.weight.shape[1]
    if in_features is None:
        return
    old_weight = getattr(score, "weight", None)
    device = old_weight.device if old_weight is not None else torch.device("cpu")
    dtype = old_weight.dtype if old_weight is not None else torch.float32
    model.score = torch.nn.Linear(in_features, num_labels, bias=False, device=device, dtype=dtype)

def set_pad_token_id_deep(model: torch.nn.Module, pad_token_id: int | None) -> None:
    if pad_token_id is None:
        return
    seen_configs: set[int] = set()

    def set_config(obj: Any) -> None:
        config = getattr(obj, "config", None)
        if config is None or id(config) in seen_configs:
            return
        seen_configs.add(id(config))
        config.pad_token_id = pad_token_id
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            text_config.pad_token_id = pad_token_id

    set_config(model)
    for attr in ("base_model", "model", "language_model"):
        set_config(getattr(model, attr, None))
    for _, module in model.named_modules():
        set_config(module)
def disable_use_cache(model: torch.nn.Module) -> None:
    seen: set[int] = set()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        if hasattr(config, "use_cache"):
            config.use_cache = False
def materialize(logits, rows, config):
    logits = logits.float()
    nr, ne = len(config['roots']), len(config['exact'])
    scores = logits[:, :nr+ne].sigmoid().cpu().tolist()
    primary = logits[:, nr+ne:nr+2*ne].argmax(1).cpu().tolist()
    states, offset = {}, nr+2*ne
    for field, values in STATE_LABELS.items():
        states[field] = logits[:, offset:offset+len(values)].argmax(1).cpu().tolist()
        offset += len(values)
    result = []
    for i, row in enumerate(rows):
        roots, exact = predict_sets(scores[i][:nr], scores[i][nr:], config, .5, .95)
        result.append({'sample_id': row['sample_id'], 'root_scores': scores[i][:nr], 'exact_scores': scores[i][nr:],
                       'roots': roots, 'exact': exact, 'auxiliary_primary': config['exact'][primary[i]],
                       'document_state': {k: STATE_LABELS[k][v[i]] for k, v in states.items()}})
    return result
