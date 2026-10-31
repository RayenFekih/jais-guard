# JaisGuard architecture and verification

The Jais 3 0.9B tokenizer and token embeddings feed 28 causal decoder layers. Each layer uses pre-attention RMSNorm, causal self-attention, a residual addition, pre-MLP RMSNorm, a dense gated SiLU MLP, and a second residual addition. A final RMSNorm produces the sequence representations with hidden dimension 1,536. The diagram expands the final layer and abbreviates intervening layers; positional encoding and attention projections are omitted for readability.

The original causal LM projects hidden states through an LM head into vocabulary logits and autoregressively generates tokens. JaisGuard instead loads AutoModel without the LM head, selects the hidden state at the last non-padding position using the attention mask, casts the representation to FP32, and applies Linear(1536, 2) with bias. Softmax supplies class probabilities. The published runtime applies a validation-locked threshold to P(GIBBERISH), or 0.5 for the default operating point; the diagram does not imply an unconditional argmax decision.

Frozen + Head: all original backbone parameters are frozen; only 1,536 × 2 + 2 = 3,074 classifier parameters train. LoRA + Head: original weights remain frozen; rank-8 adapters train in q_proj and v_proj in all 28 decoder layers, alongside the FP32 classifier head. Alpha is 16 and LoRA dropout is 0.05. The color legend refers to training, not inference; all parameters are fixed at inference. Softmax and pooling have no trainable parameters.

## Sources

Architecture dimensions and requested variants: supplied jaisguard_readme_metrics.json, architecture section.

Implementation checked at repository commit 8b5485a4bce831d98f4aa0e64a88308914377809 on cleanup/public-workflows:

- https://github.com/RayenFekih/jais-guard/blob/8b5485a4bce831d98f4aa0e64a88308914377809/src/jais_guard/finetuning/model.py
- https://github.com/RayenFekih/jais-guard/blob/8b5485a4bce831d98f4aa0e64a88308914377809/src/jais_guard/publication/model_runtime.py
- https://github.com/RayenFekih/jais-guard/blob/8b5485a4bce831d98f4aa0e64a88308914377809/configs/finetuning-iteration2.json

Pinned base implementation inspected at revision e51941b37091287fee4934e3f8a799e0d3fef37f:

- https://huggingface.co/IFM/Jais-3-0.9B/blob/e51941b37091287fee4934e3f8a799e0d3fef37f/modeling_k2_horizon.py (K2HorizonMLP, K2HorizonDecoderLayer, K2HorizonModel)
- https://huggingface.co/IFM/Jais-3-0.9B/blob/e51941b37091287fee4934e3f8a799e0d3fef37f/config.json

## Reproduction

Python 3.12 and Matplotlib 3.10.8. Install matplotlib==3.10.8, then run:

`python draw_architecture.py --metrics jaisguard_readme_metrics.json`

SVG text, shapes, and connectors are editable. PNG is 4200 × 3300 pixels at 300 dpi. No performance claims are included.
