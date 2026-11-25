"""
Ambient SFT plugin for Axolotl (https://huggingface.co/papers/2611.23229).
"""

from axolotl.integrations.base import BasePlugin

from .args import AmbientArgs as AmbientArgs


class AmbientPlugin(BasePlugin):
    """
    Plugin for Ambient SFT: matching the frozen initial model at a high temperature on a random fraction of
    micro-batches.
    """

    def get_input_args(self):
        return "axolotl.integrations.ambient.AmbientArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.ambient.args.AmbientTrainingArgsMixin"

    def get_trainer_cls(self, cfg):
        if cfg.ambient_sft and not cfg.rl:
            from .trainer import AxolotlAmbientTrainer

            return AxolotlAmbientTrainer
        return None

    def get_training_args(self, cfg):
        return {
            "ambient_clean_probability": cfg.ambient_clean_probability,
            "ambient_temperature": cfg.ambient_temperature,
        }
