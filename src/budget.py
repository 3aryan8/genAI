"""LoRA parameter-budget calculations."""

from __future__ import annotations

from collections.abc import Mapping

ModuleShape = tuple[int, int]
RankedModule = tuple[ModuleShape, int]


def lora_param_cost(module_shape: ModuleShape, rank: int) -> int:
    """Return the trainable A/B parameter count for a linear module."""
    d_in, d_out = module_shape
    if d_in <= 0 or d_out <= 0 or rank <= 0:
        raise ValueError("Module dimensions and LoRA rank must be positive.")
    return rank * (d_in + d_out)


def total_lora_params(rank_map: Mapping[str, RankedModule]) -> int:
    """Sum LoRA parameters from {module_name: ((d_in, d_out), rank)}."""
    if not rank_map:
        raise ValueError("Cannot calculate a LoRA budget for an empty rank map.")
    return sum(
        lora_param_cost(module_shape, rank)
        for module_shape, rank in rank_map.values()
    )


def uniform_test_rank_map(rank: int) -> dict[str, RankedModule]:
    """Return the verified 72-module HuBERT-base target shape map."""
    if rank <= 0:
        raise ValueError("LoRA rank must be positive.")
    result: dict[str, RankedModule] = {}
    for layer in range(12):
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            result[f"encoder.layers.{layer}.{projection}"] = ((768, 768), rank)
        result[f"encoder.layers.{layer}.feed_forward.intermediate_dense"] = (
            (768, 3072),
            rank,
        )
        result[f"encoder.layers.{layer}.feed_forward.output_dense"] = (
            (3072, 768),
            rank,
        )
    return result
