# Ambient SFT

Ambient SFT ([paper](https://huggingface.co/papers/2611.23229)) reduces the forgetting caused by fine-tuning on small
datasets. On a random fraction of micro-batches, it replaces the SFT loss with the cross-entropy from the frozen initial
model's next-token distribution, softened at a high temperature, to the model's distribution at the same temperature,
scaled by the ratio of the initial model's entropy at that temperature to its entropy at temperature 1.

With an adapter (LoRA), the initial model is the model with its adapter disabled, so no copy is loaded. Otherwise a frozen
copy of `base_model` is loaded as the teacher.

## Usage

```yaml
plugins:
  - axolotl.integrations.ambient.AmbientPlugin

ambient_sft: true
ambient_clean_probability: 0.75  # probability that a micro-batch uses the SFT loss; 1 is standard SFT
ambient_temperature: 2.5
```

The paper's instruction-tuning setting: Qwen3-1.7B-Base on 1% of `allenai/Dolci-Instruct-SFT` for 10 epochs, learning
rate 8e-5, Adam beta2 0.95, about 1M tokens per optimizer step.

The matching loss needs full logits: `cut_cross_entropy` and `liger_fused_linear_cross_entropy` are not supported, nor is
context parallelism.
