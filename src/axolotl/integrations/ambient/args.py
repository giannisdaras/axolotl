"""
Config args for the Ambient SFT plugin.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field, model_validator


class AmbientArgs(BaseModel):
    """
    Input args for Ambient SFT.
    """

    ambient_sft: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable Ambient SFT: on a random fraction of micro-batches, match the initial model at a high temperature instead of the SFT targets."
        },
    )
    ambient_clean_probability: float = Field(
        default=0.75,
        json_schema_extra={
            "description": "Probability that a micro-batch uses the standard SFT loss. 1 recovers standard SFT."
        },
    )
    ambient_temperature: float = Field(
        default=2.5,
        json_schema_extra={
            "description": "Temperature at which the model matches the initial model."
        },
    )

    @model_validator(mode="before")
    @classmethod
    def check_logits_available(cls, data):
        if data.get("ambient_sft") and (
            data.get("cut_cross_entropy")
            or data.get("liger_fused_linear_cross_entropy")
        ):
            raise ValueError(
                "Ambient SFT needs the full logits of the model and its teacher. "
                "Disable `cut_cross_entropy` and `liger_fused_linear_cross_entropy`."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_no_context_parallel(cls, data):
        if data.get("ambient_sft") and (
            (data.get("context_parallel_size") or 1) > 1
            or (data.get("sequence_parallel_degree") or 1) > 1
        ):
            raise ValueError("Ambient SFT does not support context parallelism.")
        return data


@dataclass
class AmbientTrainingArgsMixin:
    """
    Additional training args for Ambient SFT.
    """

    ambient_clean_probability: float = field(default=0.75)
    ambient_temperature: float = field(default=2.5)
