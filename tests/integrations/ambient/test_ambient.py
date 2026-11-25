"""
CPU tests for the Ambient SFT plugin.
"""

import math

import pytest
import torch

from axolotl.cli.config import plugin_set_cfg
from axolotl.common.datasets import load_datasets
from axolotl.integrations.ambient.args import AmbientArgs
from axolotl.train import setup_model_and_trainer
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault


def _cfg(temp_dir, **overrides):
    cfg = {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "tokenizer_type": "AutoTokenizer",
        "sequence_len": 256,
        "val_set_size": 0.0,
        "special_tokens": {"pad_token": "<|endoftext|>"},
        "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
        "num_epochs": 1,
        "max_steps": 4,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "output_dir": temp_dir,
        "learning_rate": 1e-4,
        "optimizer": "adamw_torch",
        "lr_scheduler": "constant",
        "logging_steps": 1,
        "seed": 42,
        "bf16": False,
        "fp16": False,
        "save_strategy": "no",
        "save_first_step": False,
    }
    cfg.update(overrides)
    return DictDefault(cfg)


def _ambient(temp_dir, **overrides):
    return _cfg(
        temp_dir,
        plugins=["axolotl.integrations.ambient.AmbientPlugin"],
        ambient_sft=True,
        **overrides,
    )


def _trainer(cfg):
    prepare_plugins(cfg)
    cfg = validate_config(cfg)
    normalize_config(cfg)
    plugin_set_cfg(cfg)
    dataset_meta = load_datasets(cfg=cfg)
    return setup_model_and_trainer(cfg, dataset_meta)[0]


@pytest.mark.parametrize(
    "option", ["cut_cross_entropy", "liger_fused_linear_cross_entropy"]
)
def test_rejects_fused_cross_entropy(option):
    with pytest.raises(ValueError, match="full logits"):
        AmbientArgs(ambient_sft=True, **{option: True})


def test_loss_at_initialization_is_scaled_teacher_entropy(temp_dir):
    trainer = _trainer(_ambient(temp_dir, ambient_clean_probability=0.0))
    assert trainer.teacher_model is not None
    batch = next(iter(trainer.get_train_dataloader()))
    trainer.model.train()
    loss = trainer.compute_loss(trainer.model, dict(batch)).item()

    with torch.no_grad():
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        logits = trainer.teacher_model(**inputs).logits[..., :-1, :].float()
        mask = batch["labels"][..., 1:] != -100

        def entropy(temperature):
            log_probs = torch.log_softmax(logits / temperature, dim=-1)
            return (-(log_probs.exp() * log_probs).sum(-1) * mask).sum()

        temperature = trainer.args.ambient_temperature
        scale = entropy(temperature) / entropy(1.0)
        expected = scale * entropy(temperature) / mask.sum()

    assert scale > 1  # a hotter distribution has more entropy
    assert loss == pytest.approx(expected.item(), rel=1e-4)


def test_clean_probability_one_matches_sft(temp_dir):
    vanilla = _trainer(_cfg(temp_dir))
    vanilla.train()
    ambient = _trainer(_ambient(temp_dir, ambient_clean_probability=1.0))
    ambient.train()
    losses = [
        [log["loss"] for log in t.state.log_history if "loss" in log]
        for t in (vanilla, ambient)
    ]
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)


@pytest.mark.parametrize("adapter", [None, "lora"])
def test_mixed_micro_batches_train(temp_dir, adapter):
    overrides = {"ambient_clean_probability": 0.5}
    if adapter:
        overrides.update(
            adapter="lora", lora_r=8, lora_alpha=16, lora_target_linear=True
        )
    trainer = _trainer(_ambient(temp_dir, **overrides))
    # with an adapter, disabling it recovers the initial model, so no teacher copy is loaded
    assert (trainer.teacher_model is None) == (adapter == "lora")
    trainer.train()
    logs = [log for log in trainer.state.log_history if "loss" in log]
    assert logs and all(math.isfinite(log["loss"]) for log in logs)
    fractions = [log["ambient/matching_fraction"] for log in logs]
    assert any(0 < f < 1 for f in fractions)  # a step mixed both losses
