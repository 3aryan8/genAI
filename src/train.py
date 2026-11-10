"""Train the uniform-LoRA HuBERT CTC baseline."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from transformers import AutoConfig, HubertForCTC, get_linear_schedule_with_warmup

from src import budget, data, eval as evaluation
from src.model import (
    _find_ctc_head,
    _find_encoder_layers,
    _freeze_feature_encoder,
    _print_layer_modules_and_find_targets,
    _read_vocab,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config.yaml"
EXPECTED_TRAINABLE = {4: 686_622, 16: 2_677_278}
PERIODIC_EVAL_UTTERANCES = 200


def _read_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {CONFIG_PATH}")
    return config


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _verify_hf_ctc_fp32_path() -> None:
    from transformers.loss.loss_ctc import ForCTCLoss

    loss_source = inspect.getsource(ForCTCLoss)
    hubert_source = inspect.getsource(HubertForCTC.forward)
    if "log_softmax(logits, dim=-1, dtype=torch.float32)" not in loss_source:
        raise RuntimeError(
            "Installed Transformers CTC loss does not explicitly calculate "
            "log_softmax in float32; inspect this version before training."
        )
    if "self.loss_function(" not in hubert_source:
        raise RuntimeError(
            "HuBERT forward no longer delegates labels to loss_function; "
            "inspect this Transformers version before training."
        )
    print(
        "HF CTC loss source verified: HubertForCTC.forward delegates to "
        "ForCTCLoss, which applies log_softmax(dtype=torch.float32); "
        "runtime loss dtype will also be asserted."
    )


def _verify_budget_examples() -> None:
    expected = {4: 663_552, 16: 2_654_208}
    for rank, expected_count in expected.items():
        actual = budget.total_lora_params(budget.uniform_test_rank_map(rank))
        if actual != expected_count:
            raise AssertionError(
                f"Uniform rank-{rank} budget is {actual}, expected {expected_count}."
            )
        print(f"uniform rank-{rank} LoRA budget={actual} (verified)")
    if budget.lora_param_cost((768, 3072), 1) != 3840:
        raise AssertionError("Feed-forward rank-step cost must be 3840.")
    if budget.lora_param_cost((768, 768), 1) != 1536:
        raise AssertionError("Attention rank-step cost must be 1536.")
    print("rank-step costs verified: feed-forward=3840 attention=1536")


def _load_base_model(config: dict[str, Any]) -> HubertForCTC:
    vocab_size, pad_token_id, _ = _read_vocab()
    model_config = AutoConfig.from_pretrained(config["model"]["id"])
    model_config.vocab_size = vocab_size
    model_config.pad_token_id = pad_token_id
    model_config.ctc_loss_reduction = config["model"]["ctc_loss_reduction"]
    model_config.layerdrop = float(config["model"]["layerdrop"])
    model_config.mask_time_prob = float(config["model"]["mask_time_prob"])
    print(
        "config overrides: vocab_size, pad_token_id, "
        "ctc_loss_reduction='mean' and layerdrop=0.0 per experiment decision; "
        "mask_time_prob retained at 0.05; ctc_zero_infinity left unchanged "
        f"({model_config.ctc_zero_infinity!r})."
    )
    return HubertForCTC.from_pretrained(
        config["model"]["id"],
        config=model_config,
        ignore_mismatched_sizes=True,
    )


def _attach_lora(
    config: dict[str, Any], rank: int
) -> tuple[torch.nn.Module, dict[str, tuple[tuple[int, int], int]], int]:
    base_model = _load_base_model(config)
    _freeze_feature_encoder(base_model)
    layers = _find_encoder_layers(base_model)
    target_patterns, target_modules = _print_layer_modules_and_find_targets(
        base_model, layers, verbose=False
    )
    if len(target_modules) != 72 or len(layers) != 12:
        raise RuntimeError(
            "Expected the M2-verified 72 LoRA targets in 12 layers; "
            f"found {len(target_modules)} targets in {len(layers)} layers."
        )
    head_name, head, head_parameters = _find_ctc_head(base_model)
    rank_map = {
        name: ((module.in_features, module.out_features), rank)
        for module, name in target_modules
    }
    adapter_count = budget.total_lora_params(rank_map)
    expected_adapter_count = sum(
        rank * (module.in_features + module.out_features)
        for module, _ in target_modules
    )
    if adapter_count != expected_adapter_count:
        raise AssertionError(
            f"Budget helper returned {adapter_count}; target shapes require "
            f"{expected_adapter_count}."
        )
    expected_trainable = adapter_count + head_parameters

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * int(config["methods"]["uniform_lora"]["alpha_multiplier"]),
        lora_dropout=float(config["methods"]["uniform_lora"]["dropout"]),
        bias="none",
        target_modules=target_patterns,
        modules_to_save=[head_name],
    )
    model = get_peft_model(base_model, lora_config)
    actual_trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if actual_trainable != expected_trainable:
        raise AssertionError(
            f"PEFT trainable count {actual_trainable} != formula count "
            f"{expected_trainable} for rank {rank}."
        )
    if rank in EXPECTED_TRAINABLE and actual_trainable != EXPECTED_TRAINABLE[rank]:
        raise AssertionError(
            f"rank {rank} trainable count {actual_trainable} != "
            f"M2-verified count {EXPECTED_TRAINABLE[rank]}."
        )
    print(
        f"trainable parameters: {actual_trainable} "
        f"(LoRA={adapter_count}, saved {head_name}={head_parameters}, "
        "formula agrees with PEFT)"
    )
    return model, rank_map, actual_trainable


def _make_loader(
    config: dict[str, Any],
    split: str,
    vocabulary: data.CharacterVocabulary,
    feature_extractor: Any,
    needs_attention_mask: bool,
    seed: int,
    debug: bool,
) -> tuple[torch.utils.data.DataLoader[dict[str, Any]], int, float]:
    records = evaluation._manifest_records(split)
    if debug:
        limit = int(
            config["training"]["debug_train_utterances"]
            if split == "train10h"
            else config["training"]["debug_dev_clean_utterances"]
        )
        records = records[:limit]
    requested_batch_seconds = float(
        config["training"]["debug_batch_max_seconds"]
        if debug
        else config["data"]["batch_max_seconds"]
    )
    effective_batch_seconds = max(
        requested_batch_seconds,
        max(record.duration_sec for record in records),
    )
    loader = data._build_audio_loader(
        records,
        vocabulary,
        feature_extractor,
        effective_batch_seconds,
        seed,
        needs_attention_mask,
    )
    if len(loader) == 0:
        raise RuntimeError(f"No usable batches found for {split}.")
    if effective_batch_seconds != requested_batch_seconds:
        print(
            f"{split} batch budget raised to {effective_batch_seconds:.3f}s "
            "to accommodate its longest single utterance"
        )
    return loader, len(records), effective_batch_seconds


def _move_batch(
    batch: dict[str, Any], device: torch.device, include_labels: bool = True
) -> dict[str, torch.Tensor]:
    inputs = {
        "input_values": batch["input_values"].to(
            device=device, dtype=torch.float32
        )
    }
    if "attention_mask" in batch:
        inputs["attention_mask"] = batch["attention_mask"].to(device=device)
    if include_labels:
        inputs["labels"] = batch["labels"].to(device=device, dtype=torch.long)
    return inputs


def _loss_for_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    use_bf16: bool,
) -> torch.Tensor:
    inputs = _move_batch(batch, device)
    if use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**inputs)
    else:
        output = model(**inputs)
    if output.loss is None:
        raise RuntimeError("HuBERT CTC model returned no loss.")
    if output.loss.dtype != torch.float32:
        raise TypeError(
            f"CTC loss must be float32, received {output.loss.dtype}."
        )
    if not bool(torch.isfinite(output.loss).item()):
        raise FloatingPointError("CTC loss is not finite.")
    return output.loss


def _append_metric(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")


def _save_adapter(model: torch.nn.Module, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    signature = inspect.signature(PeftModel.save_pretrained)
    if "save_embedding_layers" not in signature.parameters:
        raise RuntimeError(
            "Installed PeftModel.save_pretrained does not expose "
            "save_embedding_layers; refusing to save with implicit embedding "
            "handling."
        )
    model.save_pretrained(path, save_embedding_layers=False)
    adapter_file = path / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(
            f"Expected PEFT safe-serialization adapter file at {adapter_file}."
        )
    with safe_open(adapter_file, framework="pt", device="cpu") as adapter:
        keys = list(adapter.keys())
    head_weights_present = any(
        "lm_head" in key and key.endswith(".weight") for key in keys
    )
    head_bias_present = any(
        "lm_head" in key and key.endswith(".bias") for key in keys
    )
    print(
        f"saved adapter tensors={len(keys)}; "
        f"lm_head weight present={head_weights_present}; "
        f"lm_head bias present={head_bias_present}"
    )
    if not head_weights_present or not head_bias_present:
        raise AssertionError(
            "Saved adapter file must contain both lm_head modules_to_save weight "
            "and bias."
        )


def _reload_loss_check(
    model: torch.nn.Module,
    batch: dict[str, Any],
    config: dict[str, Any],
    rank: int,
    checkpoint: Path,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        expected_loss = float(
            _loss_for_batch(model, batch, device, use_bf16=False).item()
        )

    base_model = _load_base_model(config)
    _freeze_feature_encoder(base_model)
    reloaded = PeftModel.from_pretrained(
        base_model, checkpoint, is_trainable=False
    ).to(device)
    reloaded.eval()
    with torch.no_grad():
        actual_loss = float(
            _loss_for_batch(reloaded, batch, device, use_bf16=False).item()
        )
    difference = abs(expected_loss - actual_loss)
    print(
        f"checkpoint reload loss: in_memory={expected_loss:.8f} "
        f"reloaded={actual_loss:.8f} absolute_difference={difference:.8g}"
    )
    del reloaded, base_model
    if difference > 1e-4:
        raise AssertionError(
            f"Reloaded adapter/head loss differs by {difference:.8g} (>1e-4) "
            f"at rank {rank}."
        )
    model.train()
    return difference


def _report_save_signature() -> None:
    signature = inspect.signature(PeftModel.save_pretrained)
    has_embedding_option = "save_embedding_layers" in signature.parameters
    print(f"installed PeftModel.save_pretrained signature: {signature}")
    print(f"save_embedding_layers parameter exists: {has_embedding_option}")
    if not has_embedding_option:
        raise RuntimeError(
            "PEFT save_pretrained lacks save_embedding_layers; the requested "
            "safe save behavior cannot be guaranteed."
        )


def _run_periodic_evaluation(
    model: torch.nn.Module,
    device: torch.device,
    epoch: int,
    global_step: int,
    train_loss: float,
    metrics_path: Path,
) -> float:
    model.eval()
    wer, examples = evaluation.evaluate_wer_with_examples(
        model,
        "dev-clean",
        max_utts=PERIODIC_EVAL_UTTERANCES,
        example_count=2,
    )
    prediction_summary = " | ".join(
        f"ref={reference!r} pred={prediction!r}"
        for reference, prediction in examples
    )
    row = {
        "type": "evaluation",
        "step": global_step,
        "epoch": epoch,
        "train_loss": train_loss,
        "dev_clean_utterances": PERIODIC_EVAL_UTTERANCES,
        "dev_clean_wer": wer,
        "examples": [
            {"reference": reference, "prediction": prediction}
            for reference, prediction in examples
        ],
    }
    print(
        f"eval epoch={epoch} step={global_step} train_loss={train_loss:.6f} "
        f"dev-clean WER={wer:.6f} {prediction_summary}"
    )
    _append_metric(
        metrics_path,
        row,
    )
    model.train()
    return wer


def _resolved_run_config(
    config: dict[str, Any],
    rank: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    debug: bool,
    steps_per_epoch: int,
) -> dict[str, Any]:
    return {
        "experiment_config": copy.deepcopy(config),
        "method": "uniform",
        "rank": rank,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lora_alpha": rank
        * int(config["methods"]["uniform_lora"]["alpha_multiplier"]),
        "lora_dropout": float(config["methods"]["uniform_lora"]["dropout"]),
        "optimizer": "AdamW",
        "weight_decay": float(config["training"]["weight_decay"]),
        "adam_betas": [
            float(config["training"]["adam_beta1"]),
            float(config["training"]["adam_beta2"]),
        ],
        "adam_epsilon": float(config["training"]["adam_epsilon"]),
        "scheduler": "linear_warmup_then_decay",
        "warmup_fraction": float(config["training"]["warmup_fraction"]),
        "gradient_clip_norm": float(config["training"]["gradient_clip_norm"]),
        "batch_max_seconds": float(
            config["training"]["debug_batch_max_seconds"]
            if debug
            else config["data"]["batch_max_seconds"]
        ),
        "debug": debug,
        "steps_per_epoch": steps_per_epoch,
        "ctc_loss_reduction": config["model"]["ctc_loss_reduction"],
        "layerdrop": float(config["model"]["layerdrop"]),
        "mask_time_prob": float(config["model"]["mask_time_prob"]),
        "ctc_zero_infinity": None,
        "mixed_precision": "bfloat16 autocast on CUDA; float32 on CPU",
        "trainable_parameter_groups": "LoRA and saved CTC head share one LR",
    }


def train_one(
    config: dict[str, Any],
    rank: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    output_dir: Path,
    debug: bool,
    eval_every: int | None = None,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    if rank <= 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("Rank, epochs, and learning rate must be positive.")
    if eval_every is not None and eval_every <= 0:
        raise ValueError("eval_every must be positive.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")
    _set_seeds(seed)
    _verify_hf_ctc_fp32_path()
    feature_extractor, _, loaded_model_id, needs_attention_mask = (
        data._load_model_preprocessing()
    )
    print(f"feature extractor loaded from {loaded_model_id}")
    _, _, vocabulary = _read_vocab()
    train_loader, train_utterances, actual_train_batch_seconds = _make_loader(
        config,
        "train10h",
        vocabulary,
        feature_extractor,
        needs_attention_mask,
        seed,
        debug,
    )
    steps_per_epoch = (
        int(config["training"]["debug_steps"]) if debug else len(train_loader)
    )
    eval_every = steps_per_epoch if eval_every is None else eval_every
    if eval_every <= 0:
        raise ValueError("eval_every must be positive.")
    if debug and epochs != int(config["training"]["debug_epochs"]):
        raise ValueError(
            f"--debug requires exactly {config['training']['debug_epochs']} epoch."
        )
    total_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    print(
        f"training data: {train_utterances} utterances, "
        f"{len(train_loader)} length-bucketed batches"
    )
    print(f"steps per epoch={steps_per_epoch}, total steps={total_steps}")

    dev_limit = PERIODIC_EVAL_UTTERANCES
    first_train_batch = next(iter(train_loader))
    evaluation.verify_label_round_trip([first_train_batch], vocabulary)

    model, rank_map, trainable_parameters = _attach_lora(config, rank)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = _resolved_run_config(
        config, rank, seed, epochs, learning_rate, debug, steps_per_epoch
    )
    run_config["trainable_parameters"] = trainable_parameters
    run_config["lora_parameters"] = budget.total_lora_params(rank_map)
    run_config["dev_clean_max_utterances"] = dev_limit
    run_config["ctc_zero_infinity"] = bool(model.config.ctc_zero_infinity)
    run_config["total_steps"] = total_steps
    run_config["warmup_steps"] = math.ceil(
        total_steps * float(config["training"]["warmup_fraction"])
    )
    run_config["needs_attention_mask"] = needs_attention_mask
    run_config["batch_max_seconds"] = actual_train_batch_seconds
    run_config["eval_every"] = eval_every
    run_config["max_steps"] = max_steps
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float32)
    use_bf16 = device.type == "cuda"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA is available but does not support bf16.")
    print(f"training device={device}, bf16 autocast={use_bf16}")
    model.train()
    print(
        f"run settings: LoRA alpha={rank * int(config['methods']['uniform_lora']['alpha_multiplier'])}, "
        f"LoRA dropout={float(config['methods']['uniform_lora']['dropout'])}, "
        f"warmup fraction={float(config['training']['warmup_fraction'])}, "
        f"weight decay={float(config['training']['weight_decay'])}, "
        f"optimizer=AdamW, batch seconds={actual_train_batch_seconds}"
    )
    print(
        f"linear schedule: warmup_steps={run_config['warmup_steps']} "
        f"total_steps={total_steps} epochs={epochs}"
    )
    initial_debug_loss: float | None = None
    if debug:
        model.eval()
        with torch.no_grad():
            initial_debug_loss = float(
                _loss_for_batch(model, first_train_batch, device, use_bf16).item()
            )
        print(f"debug fixed-batch loss before training={initial_debug_loss:.6f}")
        model.train()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        betas=(
            float(config["training"]["adam_beta1"]),
            float(config["training"]["adam_beta2"]),
        ),
        eps=float(config["training"]["adam_epsilon"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    warmup_steps = run_config["warmup_steps"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    results: list[dict[str, Any]] = []
    best_wer = float("inf")
    global_step = 0
    log_every = int(config["training"]["log_every_steps"])
    clip_norm = float(config["training"]["gradient_clip_norm"])
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        epoch_first_step = global_step
        last_evaluation_step: int | None = None
        last_evaluation_wer: float | None = None
        epoch_loader = train_loader
        if not debug:
            epoch_loader, _, _ = _make_loader(
                config,
                "train10h",
                vocabulary,
                feature_extractor,
                needs_attention_mask,
                seed + epoch,
                False,
            )
        epoch_iterator = iter(epoch_loader)
        model.train()
        for step_in_epoch in range(steps_per_epoch):
            if global_step >= total_steps:
                break
            try:
                batch = next(epoch_iterator)
            except StopIteration:
                epoch_iterator = iter(epoch_loader)
                batch = next(epoch_iterator)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_batch(model, batch, device, use_bf16)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], clip_norm
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % log_every == 0 or global_step == total_steps:
                peak_memory = (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                )
                row = {
                    "type": "step",
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach().item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "grad_norm": float(grad_norm.detach().item()),
                    "peak_gpu_memory_bytes": peak_memory,
                }
                _append_metric(metrics_path, row)
                print(
                    f"step={global_step} epoch={epoch + 1} "
                    f"loss={row['loss']:.6f} lr={row['lr']:.3g} "
                    f"grad_norm={row['grad_norm']:.4g} "
                    f"peak_gpu_memory_bytes={peak_memory}"
                )
            if global_step % eval_every == 0:
                last_evaluation_wer = _run_periodic_evaluation(
                    model,
                    device,
                    epoch + 1,
                    global_step,
                    float(loss.detach().item()),
                    metrics_path,
                )
                last_evaluation_step = global_step

        if global_step == epoch_first_step:
            break
        if last_evaluation_step != global_step:
            wer = _run_periodic_evaluation(
                model,
                device,
                epoch + 1,
                global_step,
                float(loss.detach().item()),
                metrics_path,
            )
            last_evaluation_step = global_step
            last_evaluation_wer = wer
        else:
            if last_evaluation_wer is None:
                raise RuntimeError("Missing WER for the completed training epoch.")
            wer = last_evaluation_wer
        epoch_result = {
            "epoch": epoch + 1,
            "step": global_step,
            "dev_clean_wer": wer,
        }
        results.append(epoch_result)
        _append_metric(metrics_path, {"type": "epoch", **epoch_result})
        last_path = output_dir / "last"
        _save_adapter(model, last_path)
        _reload_loss_check(
            model, first_train_batch, config, rank, last_path, device
        )
        if wer < best_wer:
            best_wer = wer
            _save_adapter(model, output_dir / "best")
        epoch_elapsed = time.perf_counter() - epoch_started
        epoch_steps = global_step - epoch_first_step
        steps_per_second = epoch_steps / epoch_elapsed if epoch_elapsed > 0 else 0.0
        _append_metric(
            metrics_path,
            {
                "type": "epoch_timing",
                "epoch": epoch + 1,
                "steps": epoch_steps,
                "elapsed_seconds": epoch_elapsed,
                "steps_per_second": steps_per_second,
            },
        )
        print(
            f"epoch={epoch + 1} time={epoch_elapsed:.2f}s "
            f"steps={epoch_steps} steps_per_second={steps_per_second:.3f}"
        )
        if global_step >= total_steps:
            break

    if debug:
        model.eval()
        with torch.no_grad():
            final_debug_loss = float(
                _loss_for_batch(model, first_train_batch, device, use_bf16).item()
            )
        print(
            f"debug fixed-batch loss after training={final_debug_loss:.6f} "
            f"(before={initial_debug_loss:.6f})"
        )
        if initial_debug_loss is None or final_debug_loss >= initial_debug_loss:
            raise AssertionError("Debug training did not decrease the fixed-batch loss.")
    return results


def _run_sweep(config: dict[str, Any]) -> None:
    rates = [float(rate) for rate in config["training"]["sweep_learning_rates"]]
    epochs = int(config["training"]["sweep_epochs"])
    sweep_result: dict[str, Any] = {
        "method": "uniform",
        "rank": 4,
        "seed": 0,
        "epochs": epochs,
        "learning_rates": rates,
        "runs": [],
    }
    for learning_rate in rates:
        rate_label = f"{learning_rate:.0e}".replace("+", "")
        run_name = f"uniform-r4-seed0-lr{rate_label}"
        run_results = train_one(
            config,
            rank=4,
            seed=0,
            epochs=epochs,
            learning_rate=learning_rate,
            output_dir=REPOSITORY_ROOT / "results" / "runs" / run_name,
            debug=False,
        )
        sweep_result["runs"].append(
            {
                "learning_rate": learning_rate,
                "output_dir": f"results/runs/{run_name}",
                "dev_clean_wer_by_epoch": [
                    item["dev_clean_wer"] for item in run_results
                ],
            }
        )
    sweep_path = REPOSITORY_ROOT / "results" / "sweep.json"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text(
        json.dumps(sweep_result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"learning-rate curves written to {sweep_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("uniform",), default="uniform")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    config = _read_config()
    _report_save_signature()
    _verify_budget_examples()

    if args.sweep:
        if args.debug or args.rank != 4 or args.seed != 0:
            parser.error("--sweep is fixed to rank 4, seed 0, and non-debug data.")
        if args.epochs is not None or args.lr is not None or args.out is not None:
            parser.error("--sweep chooses its own epochs, learning rates, and outputs.")
        _run_sweep(config)
        return

    epochs = (
        int(config["training"]["debug_epochs"])
        if args.debug and args.epochs is None
        else int(args.epochs or config["training"]["epochs"])
    )
    learning_rate = float(
        args.lr if args.lr is not None else config["training"]["learning_rate"]
    )
    output_dir = args.out
    if output_dir is None:
        output_dir = (
            REPOSITORY_ROOT
            / "results"
            / "runs"
            / f"uniform-r{args.rank}-seed{args.seed}"
        )
    elif not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    train_one(
        config,
        rank=args.rank,
        seed=args.seed,
        epochs=epochs,
        learning_rate=learning_rate,
        output_dir=output_dir,
        debug=args.debug,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
