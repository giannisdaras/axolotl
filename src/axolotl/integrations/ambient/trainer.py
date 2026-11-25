"""
Trainer for Ambient SFT.
"""

import torch
import torch.nn.functional as F
from peft import PeftModel

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.loaders import ModelLoader


class AxolotlAmbientTrainer(AxolotlTrainer):
    """
    Trainer that replaces the SFT loss, on a random fraction of micro-batches, with the cross-entropy from the frozen
    initial model's tempered next-token distribution to the model's tempered distribution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = None
        # every rank draws the same coins in the same order, so they agree without communication
        self.coin_generator = torch.Generator().manual_seed(self.args.seed)

    def post_set_axolotl_cfg(self):
        super().post_set_axolotl_cfg()
        # with PEFT, disabling the adapter recovers the initial model, so no copy is loaded
        if isinstance(self.accelerator.unwrap_model(self.model), PeftModel):
            return
        teacher_model, _ = ModelLoader(
            self.axolotl_cfg, self.processing_class, reference_model=True
        ).load()
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        if self.is_deepspeed_enabled:
            from trl.models.utils import prepare_deepspeed

            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        elif self.is_fsdp_enabled:
            from trl.models.utils import prepare_fsdp

            self.teacher_model = prepare_fsdp(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(
                teacher_model, evaluation_mode=True
            )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if not model.training:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # one coin per micro-batch, so a step with gradient accumulation mixes both losses
        coin = torch.rand(1, generator=self.coin_generator).item()
        if coin < self.args.ambient_clean_probability:
            self.store_metrics({"ambient/matching_fraction": 0.0})
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if self.args.include_tkps:
            self._accumulate_token_counts(inputs)
        self._maybe_inject_mm_token_type_ids(inputs)
        if (
            self.args.sample_packing
            and self._model_type in ("gemma4", "gemma3", "gemma4_unified")
            and "position_ids" in inputs
        ):
            # same as AxolotlTrainer.compute_loss: packed masks come from position_ids
            inputs.pop("attention_mask", None)

        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        with torch.no_grad():
            if self.teacher_model is None:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    teacher_logits = model(**model_inputs).logits
            else:
                teacher_logits = self.teacher_model(**model_inputs).logits

        temperature = self.args.ambient_temperature
        loss_mask = labels[..., 1:] != -100
        teacher_probs = F.softmax(
            teacher_logits[..., :-1, :].float() / temperature, dim=-1
        )
        log_probs = F.log_softmax(
            outputs.logits[..., :-1, :].float() / temperature, dim=-1
        )
        per_token_loss = -(teacher_probs * log_probs).sum(-1) * loss_mask
        with torch.no_grad():
            entropy = -(teacher_probs * teacher_probs.clamp(min=1e-12).log()).sum(-1)
            teacher_log_probs = F.log_softmax(
                teacher_logits[..., :-1, :].float(), dim=-1
            )
            base_entropy = -(teacher_log_probs.exp() * teacher_log_probs).sum(-1)
            # keeps the matching gradient on the scale of the SFT loss as the temperature flattens the teacher
            entropy_scale = (entropy * loss_mask).sum() / (
                base_entropy * loss_mask
            ).sum().clamp(min=1e-8)

        # reduced like the SFT loss in `Trainer.compute_loss`/`training_step`
        if not self.model_accepts_loss_kwargs:
            num_items_in_batch = None
        num_tokens = (
            loss_mask.sum() if num_items_in_batch is None else num_items_in_batch
        )
        loss = (
            entropy_scale
            * per_token_loss.sum()
            / torch.as_tensor(num_tokens, device=per_token_loss.device).clamp(min=1)
        )
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss_scale = self.accelerator.num_processes
            if (
                pc := getattr(self.accelerator, "parallelism_config", None)
            ) is not None:
                loss_scale //= pc.tp_size
            loss = loss * loss_scale

        self.store_metrics(
            {
                "ambient/matching_fraction": 1.0,
                "ambient/matching_loss": (
                    per_token_loss.detach().sum() / loss_mask.sum().clamp(min=1)
                ).item(),
                "ambient/entropy_scale": entropy_scale.item(),
            }
        )
        return (loss, outputs) if return_outputs else loss
