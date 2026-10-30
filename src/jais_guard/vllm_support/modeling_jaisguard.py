"""HF architecture metadata for vLLM's real Transformers pooling backend.

Copied beside the pinned K2 source during export. vLLM creates this class on meta
only to discover the head; it executes its own backbone/attention and LAST pooler.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from .modeling_k2_horizon import K2HorizonModel, K2HorizonPreTrainedModel


class TrainedHead(nn.Linear):
    def forward(self, hidden):
        # vLLM 0.32.0 ClassifierWithReshape inserts the singleton sequence axis.
        if hidden.ndim == 3:
            if hidden.shape[1] != 1:
                raise ValueError("LAST pooling must select exactly one token")
            hidden = hidden[:, 0]
        with torch.autocast(hidden.device.type, enabled=False):
            return F.linear(hidden.float(), self.weight.float(), self.bias.float())


class JaisGuardForSequenceClassification(K2HorizonPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = K2HorizonModel(config)
        # Name deliberately avoids score.bias, which vLLM's LegacyMixin drops.
        self.classifier = TrainedHead(config.hidden_size, 2, bias=True, dtype=torch.float32)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs
        ).last_hidden_state
        if attention_mask is None:
            pooled = hidden[:, -1]
        else:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            last = (
                positions.expand_as(attention_mask)
                .masked_fill(~attention_mask.bool(), -1)
                .max(1)
                .values
            )
            if (last < 0).any():
                raise ValueError("Empty sequence")
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return SequenceClassifierOutput(logits=self.classifier(pooled))
