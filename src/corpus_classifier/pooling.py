"""Pool before the classification head; preserve trained parameter names."""
from types import MethodType
import torch
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

def last_non_pad(input_ids, pad_token_id):
    if input_ids.ndim != 2 or not all(input_ids.shape):
        raise ValueError('Nonempty [batch, sequence] input_ids required')
    if pad_token_id is None:
        if input_ids.shape[0] != 1:
            raise ValueError('Batch size must be one when pad_token_id is absent')
        return torch.full((1,), -1, device=input_ids.device, dtype=torch.long)
    # Match GenericForSequenceClassification, including pad IDs inside text.
    positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
    return ((input_ids != pad_token_id).to(torch.int32) * positions).argmax(-1)

def pool_hidden(hidden, input_ids, attention_mask, pad_token_id, mode):
    if mode == 'last':
        indices = last_non_pad(input_ids, pad_token_id)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
    if mode != 'masked_mean':
        raise ValueError('Unknown pooling mode')
    if attention_mask is None or attention_mask.shape != hidden.shape[:2]:
        raise ValueError('Mean pooling requires matching attention_mask')
    mask = attention_mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError('Empty token row')
    # FP32 sum prevents length-dependent BF16 accumulation error. Pad gradients = 0.
    values = hidden.float().masked_fill(~mask.unsqueeze(-1), 0)
    return (values.sum(dim=1) / mask.sum(dim=1, keepdim=True)).to(hidden.dtype)

def pooled_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                   inputs_embeds=None, labels=None, return_dict=None, **kwargs):
    if input_ids is None or inputs_embeds is not None or labels is not None:
        raise ValueError('Only explicit input_ids and external masked loss supported')
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=position_ids, use_cache=False, return_dict=True)
    pooled = pool_hidden(outputs.last_hidden_state, input_ids, attention_mask,
                         self.config.get_text_config().pad_token_id, self.v1599_pooling)
    return SequenceClassifierOutputWithPast(logits=self.score(pooled))

def install_pooling(peft_model, mode):
    if mode not in ('last', 'masked_mean'):
        raise ValueError('Unknown pooling mode')
    classifier = peft_model.get_base_model()
    if type(classifier).__name__ not in ('Qwen3_5ForSequenceClassification', 'Qwen3_5TextForSequenceClassification'):
        raise ValueError('Only Qwen3.5 classifier supported')
    if hasattr(classifier, 'v1599_pooling'):
        raise ValueError('Pooling already installed')
    if peft_model.active_peft_config.is_prompt_learning:
        raise ValueError('Prompt adapters unsupported')
    classifier.v1599_pooling = mode
    classifier.forward = MethodType(pooled_forward, classifier)
    return peft_model
