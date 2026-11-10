"""Inspect HuBERT CTC modules and verify LoRA wiring without training."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import re
from typing import Any

from peft import LoraConfig, get_peft_model
import torch
from torch import nn
import yaml
from transformers import AutoConfig, HubertForCTC

from src import data


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config.yaml"
VOCAB_PATH = REPOSITORY_ROOT / "data" / "vocab.json"
CONFIG_FIELDS = (
    "vocab_size",
    "pad_token_id",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "ctc_loss_reduction",
    "ctc_zero_infinity",
    "mask_time_prob",
    "layerdrop",
)


def _read_vocab() -> tuple[int, int, data.CharacterVocabulary]:
    with VOCAB_PATH.open(encoding="utf-8") as file:
        vocab = json.load(file)
    token_to_id = vocab["token_to_id"]
    pad_id = token_to_id[vocab["pad_token"]]
    if vocab["blank_token"] != vocab["pad_token"]:
        raise ValueError("CTC blank must use the same token as the padding token.")
    if pad_id != 0:
        raise ValueError(f"Expected <pad>/CTC blank id 0, found {pad_id}.")
    tokens = vocab["tokens"]
    if token_to_id != {token: index for index, token in enumerate(tokens)}:
        raise ValueError("data/vocab.json token IDs do not match token order.")
    vocabulary = data.CharacterVocabulary([])
    vocabulary.tokens = tokens
    vocabulary.token_to_id = token_to_id
    vocabulary.id_to_token = dict(enumerate(tokens))
    vocabulary.pad_id = pad_id
    vocabulary.unknown_id = token_to_id[vocab["unknown_token"]]
    return len(tokens), pad_id, vocabulary


def _read_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {CONFIG_PATH}")
    return config


def _print_model_config(model: HubertForCTC) -> None:
    print("config values set: vocab_size and pad_token_id only")
    for field in CONFIG_FIELDS:
        print(f"config.{field}={getattr(model.config, field, None)!r}")


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def _freeze_feature_encoder(model: HubertForCTC) -> str:
    available = {
        name: callable(getattr(model, name, None))
        for name in ("freeze_feature_encoder", "freeze_feature_extractor")
    }
    print(f"feature-freeze methods available: {available}")
    for method_name in ("freeze_feature_encoder", "freeze_feature_extractor"):
        method = getattr(model, method_name, None)
        if callable(method):
            method()
            print(f"feature-freeze method used: {method_name}")
            return method_name
    raise RuntimeError(
        "This transformers version exposes neither supported feature-encoder "
        "freezing method; refusing to guess the CNN module attribute."
    )


def _find_encoder_layers(model: HubertForCTC) -> dict[int, tuple[str, nn.Module]]:
    layers: dict[int, tuple[str, nn.Module]] = {}
    pattern = re.compile(r"(?:^|\.)encoder\.layers\.(\d+)$")
    for name, module in model.named_modules():
        match = pattern.search(name)
        if match:
            layer_index = int(match.group(1))
            if layer_index in layers:
                raise RuntimeError(f"Multiple modules found for encoder layer {layer_index}.")
            layers[layer_index] = (name, module)

    expected_count = model.config.num_hidden_layers
    expected_indices = set(range(expected_count))
    if set(layers) != expected_indices:
        raise RuntimeError(
            "Could not uniquely identify encoder layers by their actual names: "
            f"found={sorted(layers)}, expected indices={sorted(expected_indices)}"
        )
    if 0 not in layers or 11 not in layers:
        raise RuntimeError("The requested layer 0 and layer 11 were not both found.")
    return layers


def _linear_modules_in_layer(
    model: HubertForCTC, layer_name: str
) -> dict[str, nn.Linear]:
    prefix = f"{layer_name}."
    return {
        name[len(prefix) :]: module
        for name, module in model.named_modules()
        if name.startswith(prefix) and isinstance(module, nn.Linear)
    }


def _print_layer_modules_and_find_targets(
    model: HubertForCTC,
    layers: dict[int, tuple[str, nn.Module]],
    verbose: bool = True,
) -> tuple[list[str], list[tuple[nn.Linear, str]]]:
    if verbose:
        for layer_index in (0, 11):
            layer_name, _ = layers[layer_index]
            print(f"nn.Linear modules in encoder layer {layer_index} ({layer_name}):")
            for relative_name, module in sorted(
                _linear_modules_in_layer(model, layer_name).items()
            ):
                print(
                    f"  {layer_name}.{relative_name}: "
                    f"in_features={module.in_features}, "
                    f"out_features={module.out_features}"
                )

    reference_modules = _linear_modules_in_layer(model, layers[0][0])
    if not reference_modules:
        raise RuntimeError("No nn.Linear modules found in encoder layer 0.")
    reference_names = set(reference_modules)
    matches: list[tuple[nn.Linear, str]] = []
    for layer_index, (layer_name, _) in layers.items():
        layer_modules = _linear_modules_in_layer(model, layer_name)
        if set(layer_modules) != reference_names:
            raise RuntimeError(
                f"Linear module names differ in encoder layer {layer_index}: "
                f"expected={sorted(reference_names)}, got={sorted(layer_modules)}"
            )
        matches.extend(
            (module, f"{layer_name}.{relative_name}")
            for relative_name, module in layer_modules.items()
        )

    patterns = sorted({relative_name.rsplit(".", maxsplit=1)[-1] for relative_name in reference_names})
    for layer_index, (layer_name, _) in layers.items():
        layer_modules = _linear_modules_in_layer(model, layer_name)
        pattern_hits = [
            relative_name
            for relative_name in layer_modules
            if any(
                relative_name == pattern or relative_name.endswith(f".{pattern}")
                for pattern in patterns
            )
        ]
        if len(pattern_hits) != len(reference_names):
            raise RuntimeError(
                f"LoRA patterns hit {len(pattern_hits)} of "
                f"{len(reference_names)} linear modules in layer {layer_index}."
            )
        if verbose:
            print(f"LoRA pattern hits encoder layer {layer_index}: {len(pattern_hits)}")

    if verbose:
        print(f"proposed LoRA target patterns: {patterns}")
        print(f"exact target modules across encoder: {len(matches)}")
        for _, module_name in sorted(matches, key=lambda match: match[1]):
            print(f"  target: {module_name}")
    else:
        print(f"exact target modules across encoder: {len(matches)}")
    return patterns, matches


def _find_ctc_head(model: HubertForCTC) -> tuple[str, nn.Linear, int]:
    candidates = [
        (name, module)
        for name, module in model.named_children()
        if isinstance(module, nn.Linear)
        and module.out_features == model.config.vocab_size
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected one top-level nn.Linear CTC head whose output dimension "
            f"matches vocab_size={model.config.vocab_size}; found "
            f"{[name for name, _ in candidates]}."
        )
    name, head = candidates[0]
    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    print(f"CTC head name: {name}")
    print(
        f"CTC head shape: in_features={head.in_features}, "
        f"out_features={head.out_features}, parameters={parameter_count}"
    )
    return name, head, parameter_count


def _expected_trainable_parameters(
    rank: int,
    target_modules: list[tuple[nn.Linear, str]],
    head_parameters: int,
) -> int:
    adapter_parameters = sum(
        rank * (module.in_features + module.out_features)
        for module, _ in target_modules
    )
    print(
        f"r={rank}: expected LoRA parameters={adapter_parameters}, "
        f"saved CTC head parameters={head_parameters}, "
        f"expected trainable total={adapter_parameters + head_parameters}"
    )
    return adapter_parameters + head_parameters


def _attach_and_check_lora(
    base_model: HubertForCTC,
    rank: int,
    target_patterns: list[str],
    head_name: str,
    expected_parameters: int,
) -> nn.Module:
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_patterns,
        modules_to_save=[head_name],
    )
    model = get_peft_model(base_model, lora_config)
    trainable, total = _parameter_counts(model)
    agrees = trainable == expected_parameters
    print(
        f"PEFT r={rank}: trainable={trainable}, total={total}, "
        f"formula_agrees={agrees}"
    )
    if not agrees:
        raise AssertionError(
            f"PEFT trainable count {trainable} != formula count "
            f"{expected_parameters} for rank {rank}."
        )
    return model


def _make_debug_batches(
    config: dict[str, Any],
    vocabulary: data.CharacterVocabulary,
    feature_extractor: Any,
    needs_attention_mask: bool,
) -> list[dict[str, Any]]:
    data_config = config["data"]
    train_root = REPOSITORY_ROOT / data_config["librilight_root"]
    outside_characters: Counter[str] = Counter()
    records_1h, _ = data._scan_split(train_root / "1h", "1h", outside_characters)
    records_9h, _ = data._scan_split(train_root / "9h", "9h", outside_characters)
    train_records, _ = data._make_train_union(records_1h, records_9h)
    train_records = train_records[: data.DEBUG_TRAIN_UTTERANCES]
    loader = data._build_audio_loader(
        train_records,
        vocabulary,
        feature_extractor,
        float(data_config["batch_max_seconds"]),
        int(config["experiment"]["seeds"][0]),
        needs_attention_mask,
    )
    batches = list(loader)
    if not batches:
        raise RuntimeError("The M1 debug train loader produced no batches.")
    print(
        f"M1 debug training batches: {len(batches)} "
        f"from {len(train_records)} utterances"
    )
    return batches


def _model_inputs(
    batch: dict[str, Any], device: torch.device, include_labels: bool
) -> dict[str, torch.Tensor]:
    inputs = {
        "input_values": batch["input_values"].to(device=device, dtype=torch.float32)
    }
    if "attention_mask" in batch:
        inputs["attention_mask"] = batch["attention_mask"].to(device=device)
    if include_labels:
        inputs["labels"] = batch["labels"].to(device=device, dtype=torch.long)
    return inputs


def _check_losses(
    model: nn.Module, batch: dict[str, Any], device: torch.device
) -> None:
    inputs = _model_inputs(batch, device, include_labels=True)
    model.eval()
    with torch.no_grad():
        float32_loss = model(**inputs).loss
    if float32_loss is None:
        raise RuntimeError("HubertForCTC did not return a float32 CTC loss.")
    float32_finite = bool(torch.isfinite(float32_loss).item())
    print(
        f"float32 GPU forward: loss={float32_loss.item():.6f}, "
        f"finite={float32_finite}"
    )
    if not float32_finite:
        raise FloatingPointError("Float32 CTC loss is not finite.")

    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        bf16_loss = model(**inputs).loss
    if bf16_loss is None:
        raise RuntimeError("HubertForCTC did not return a bf16 CTC loss.")
    bf16_finite = bool(torch.isfinite(bf16_loss).item())
    print(f"bf16 GPU forward: loss={bf16_loss.item():.6f}, finite={bf16_finite}")
    if not bf16_finite:
        raise FloatingPointError("Bfloat16 CTC loss is not finite.")


def _check_ctc_lengths(
    model: nn.Module,
    batches: list[dict[str, Any]],
    device: torch.device,
) -> None:
    model.eval()
    for batch_index, batch in enumerate(batches, start=1):
        gaps: list[int] = []
        with torch.no_grad():
            for example_index, input_length in enumerate(batch["input_lengths"]):
                length = int(input_length.item())
                single_input = batch["input_values"][
                    example_index : example_index + 1, :length
                ].to(device=device, dtype=torch.float32)
                outputs = model(input_values=single_input)
                output_frames = int(outputs.logits.shape[1])
                label_length = int(batch["label_lengths"][example_index].item())
                gaps.append(output_frames - label_length)
        minimum_gap = min(gaps)
        print(
            f"CTC lengths batch {batch_index}: "
            f"min(frames-label_length)={minimum_gap}"
        )
        if minimum_gap <= 0:
            raise ValueError(
                f"CTC frame length is not greater than label length in debug "
                f"batch {batch_index}: minimum gap {minimum_gap}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", required=True)
    parser.parse_args()

    config = _read_config()
    vocab_size, pad_token_id, character_vocab = _read_vocab()
    if pad_token_id != 0:
        raise ValueError("Configured CTC blank/pad id must be 0.")

    model_id = config["model"]["id"]
    model_config = AutoConfig.from_pretrained(model_id)
    model_config.vocab_size = vocab_size
    model_config.pad_token_id = pad_token_id
    print(
        f"loading {model_id}; set config.vocab_size={vocab_size}, "
        f"config.pad_token_id={pad_token_id}"
    )
    base_model = HubertForCTC.from_pretrained(
        model_id,
        config=model_config,
        ignore_mismatched_sizes=True,
    )
    _print_model_config(base_model)

    before_trainable, before_total = _parameter_counts(base_model)
    print(
        f"parameters before feature freeze: trainable={before_trainable}, "
        f"total={before_total}"
    )
    _freeze_feature_encoder(base_model)
    after_trainable, after_total = _parameter_counts(base_model)
    print(
        f"parameters after feature freeze: trainable={after_trainable}, "
        f"total={after_total}"
    )

    layers = _find_encoder_layers(base_model)
    target_patterns, target_modules = _print_layer_modules_and_find_targets(
        base_model, layers
    )
    head_name, _, head_parameters = _find_ctc_head(base_model)

    expected_counts = {
        rank: _expected_trainable_parameters(rank, target_modules, head_parameters)
        for rank in (4, 16)
    }
    rank16_base = copy.deepcopy(base_model)
    model_r4 = _attach_and_check_lora(
        base_model, 4, target_patterns, head_name, expected_counts[4]
    )
    model_r16 = _attach_and_check_lora(
        rank16_base, 16, target_patterns, head_name, expected_counts[16]
    )
    del model_r16, rank16_base

    feature_extractor, _, loaded_model_id, needs_attention_mask = (
        data._load_model_preprocessing()
    )
    print(f"debug batch feature extractor model id: {loaded_model_id}")
    batches = _make_debug_batches(
        config, character_vocab, feature_extractor, needs_attention_mask
    )

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"CUDA_VISIBLE_DEVICES={visible_devices!r}")
    if visible_devices != "1":
        raise RuntimeError(
            "Run with CUDA_VISIBLE_DEVICES=1 so cuda:0 maps to physical GPU 1."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this inspection requires GPU 1.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU 1 does not support bfloat16 autocast.")
    device = torch.device("cuda:0")
    print(f"forward device: {torch.cuda.get_device_name(device)} (visible cuda:0)")
    model_r4.to(device=device, dtype=torch.float32)
    first_batch = batches[0]
    if bool((first_batch["labels"] == -100).any().item()) is False:
        print("first debug batch has no padded labels; labels remain valid CTC targets")
    _check_losses(model_r4, first_batch, device)
    _check_ctc_lengths(model_r4, batches, device)
    print("M2 inspect completed; no training was performed.")


if __name__ == "__main__":
    main()
