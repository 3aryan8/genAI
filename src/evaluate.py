"""Evaluate a saved HuBERT CTC LoRA adapter on full LibriSpeech splits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import jiwer
import torch
import yaml
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoConfig, HubertForCTC

from src import budget, data
from src.eval import _manifest_records, decode_ctc_ids
from src.model import _read_vocab


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config.yaml"
EVALUATION_SPLITS = ("dev-clean", "dev-other", "test-clean", "test-other")
TEST_SPLITS = ("test-clean", "test-other")


def _read_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {CONFIG_PATH}")
    return config


def _load_model(
    adapter_dir: Path, config: dict[str, Any]
) -> tuple[torch.nn.Module, data.CharacterVocabulary, int, str]:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")
    vocab_size, pad_token_id, vocabulary = _read_vocab()
    if vocab_size != 30 or pad_token_id != 0:
        raise ValueError(
            f"Expected verified vocab_size=30 and pad_token_id=0; "
            f"found {vocab_size} and {pad_token_id}."
        )

    model_config = AutoConfig.from_pretrained(config["model"]["id"])
    model_config.vocab_size = vocab_size
    model_config.pad_token_id = pad_token_id
    model_config.ctc_loss_reduction = "mean"
    model_config.layerdrop = 0.0
    model_config.mask_time_prob = 0.0
    print(
        "model config overrides: vocab_size=30, pad_token_id=0, "
        "ctc_loss_reduction='mean', layerdrop=0.0; mask_time_prob=0.0 "
        "for inference without time masking"
    )
    base_model = HubertForCTC.from_pretrained(
        config["model"]["id"],
        config=model_config,
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model.eval()
    return model, vocabulary, pad_token_id, str(config["model"]["id"])


def _adapter_budget(adapter_dir: Path, model: torch.nn.Module) -> dict[str, int]:
    matrices: dict[str, dict[str, torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        for component in ("lora_A", "lora_B"):
            marker = f".{component}."
            if marker not in name or not name.endswith(".weight"):
                continue
            module_name = name.split(marker, maxsplit=1)[0]
            matrices.setdefault(module_name, {})[component] = parameter

    rank_map: dict[str, tuple[tuple[int, int], int]] = {}
    for module_name, components in matrices.items():
        if set(components) != {"lora_A", "lora_B"}:
            raise RuntimeError(
                f"Incomplete LoRA matrices for module {module_name}: "
                f"{sorted(components)}"
            )
        a_matrix = components["lora_A"]
        b_matrix = components["lora_B"]
        rank = int(a_matrix.shape[0])
        if int(b_matrix.shape[1]) != rank:
            raise RuntimeError(
                f"LoRA rank mismatch in {module_name}: "
                f"A={a_matrix.shape}, B={b_matrix.shape}"
            )
        rank_map[module_name] = (
            (int(a_matrix.shape[1]), int(b_matrix.shape[0])),
            rank,
        )
    if len(rank_map) != 72:
        raise RuntimeError(
            f"Expected 72 saved HuBERT LoRA modules; found {len(rank_map)}."
        )

    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(f"Missing adapter weights file: {adapter_file}")
    with safe_open(adapter_file, framework="pt", device="cpu") as adapter:
        head_keys = [key for key in adapter.keys() if "lm_head" in key]
        head_parameters = sum(
            math.prod(adapter.get_slice(key).get_shape()) for key in head_keys
        )
    if not head_keys:
        raise RuntimeError(
            f"No lm_head modules_to_save weights found in {adapter_file}."
        )
    lora_parameters = budget.total_lora_params(rank_map)
    total_parameters = lora_parameters + head_parameters
    print(
        f"adapter trainable-parameter budget: LoRA={lora_parameters}, "
        f"saved lm_head={head_parameters}, total={total_parameters}, "
        f"modules={len(rank_map)}"
    )
    return {
        "lora_parameters": lora_parameters,
        "lm_head_parameters": head_parameters,
        "total_trainable_parameters": total_parameters,
        "lora_modules": len(rank_map),
    }


def _reference_as_logits(
    reference: str,
    vocabulary: data.CharacterVocabulary,
) -> torch.Tensor:
    labels = vocabulary.encode(reference)
    frame_ids: list[int] = []
    previous: int | None = None
    for token_id in labels:
        if token_id == previous:
            frame_ids.append(vocabulary.pad_id)
        frame_ids.append(token_id)
        previous = token_id
    logits = torch.full((len(frame_ids), len(vocabulary.tokens)), -100.0)
    logits[torch.arange(len(frame_ids)), torch.tensor(frame_ids)] = 100.0
    return logits


def _verify_reference_decode(
    records: list[data.AudioRecord],
    vocabulary: data.CharacterVocabulary,
) -> None:
    references: list[str] = []
    decoded: list[str] = []
    for record in records:
        reference_logits = _reference_as_logits(record.text, vocabulary)
        references.append(record.text)
        decoded.append(
            decode_ctc_ids(reference_logits.argmax(dim=-1), vocabulary)
        )
    score = float(jiwer.wer(references, decoded))
    print(
        f"reference-label-as-logits sanity: WER={score:.6f} "
        f"utterances={len(records)}"
    )
    if score != 0.0:
        mismatch = next(
            (reference, hypothesis)
            for reference, hypothesis in zip(references, decoded, strict=True)
            if reference != hypothesis
        )
        raise AssertionError(
            f"Reference-label decode mismatch: ref={mismatch[0]!r}, "
            f"hyp={mismatch[1]!r}"
        )


def _predict_split(
    model: torch.nn.Module,
    records: list[data.AudioRecord],
    vocabulary: data.CharacterVocabulary,
    feature_extractor: Any,
    needs_attention_mask: bool,
    batch_max_seconds: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    invalid_audio = [
        record.utt_id for record in records if not record.audio_is_valid
    ]
    missing_text = [
        record.utt_id
        for record in records
        if not record.has_transcript or not record.text
    ]
    if invalid_audio or missing_text:
        raise ValueError(
            f"Refusing partial evaluation: invalid audio={len(invalid_audio)} "
            f"(examples={invalid_audio[:5]}), empty/missing transcripts="
            f"{len(missing_text)} (examples={missing_text[:5]})."
        )

    effective_batch_seconds = max(
        batch_max_seconds,
        max(record.duration_sec for record in records),
    )
    loader = data._build_audio_loader(
        records,
        vocabulary,
        feature_extractor,
        effective_batch_seconds,
        seed=0,
        needs_attention_mask=needs_attention_mask,
    )
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            inputs = {
                "input_values": batch["input_values"].to(
                    device=device, dtype=torch.float32
                )
            }
            if "attention_mask" in batch:
                inputs["attention_mask"] = batch["attention_mask"].to(
                    device=device
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits
            frame_lengths = model._get_feat_extract_output_lengths(
                batch["input_lengths"].to(device=device)
            )
            frame_ids = logits.argmax(dim=-1)
            for index, (utterance_id, path, reference) in enumerate(
                zip(batch["utt_ids"], batch["paths"], batch["texts"], strict=True)
            ):
                hypothesis = decode_ctc_ids(
                    frame_ids[index, : int(frame_lengths[index].item())],
                    vocabulary,
                )
                predictions.append(
                    {
                        "utt_id": utterance_id,
                        "path": path,
                        "reference": reference,
                        "hypothesis": hypothesis,
                    }
                )
    if len(predictions) != len(records):
        raise AssertionError(
            f"Evaluated {len(predictions)} of {len(records)} utterances."
        )
    return predictions


def _score_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    references = [row["reference"] for row in predictions]
    hypotheses = [row["hypothesis"] for row in predictions]
    word_output = jiwer.process_words(references, hypotheses)
    total_reference_words = sum(
        len(utterance) for utterance in word_output.references
    )
    total_errors = (
        word_output.substitutions
        + word_output.deletions
        + word_output.insertions
    )
    corpus_wer = float(jiwer.wer(references, hypotheses))
    error_rate = total_errors / total_reference_words if total_reference_words else 0.0
    if not math.isclose(corpus_wer, error_rate, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(
            f"Corpus WER {corpus_wer} != (S+D+I)/reference_words "
            f"{total_errors}/{total_reference_words}={error_rate}."
        )

    utterance_scores = [
        {
            **row,
            "wer": float(jiwer.wer(row["reference"], row["hypothesis"])),
        }
        for row in predictions
    ]
    worst = sorted(
        utterance_scores,
        key=lambda row: (-row["wer"], row["utt_id"]),
    )[:5]
    best = sorted(
        utterance_scores,
        key=lambda row: (row["wer"], row["utt_id"]),
    )[:5]
    print(
        "WER aggregation sanity: corpus jiwer.wer equals "
        "(substitutions+deletions+insertions)/reference words; "
        "per-utterance WER is used only to rank best/worst examples."
    )
    return {
        "utterance_count": len(predictions),
        "reference_word_count": total_reference_words,
        "wer": corpus_wer,
        "substitutions": int(word_output.substitutions),
        "deletions": int(word_output.deletions),
        "insertions": int(word_output.insertions),
        "best_5": best,
        "worst_5": worst,
    }


def _repeatability_check(
    first: list[dict[str, Any]], second: list[dict[str, Any]], split: str
) -> None:
    first_outputs = [
        (row["utt_id"], row["reference"], row["hypothesis"]) for row in first
    ]
    second_outputs = [
        (row["utt_id"], row["reference"], row["hypothesis"]) for row in second
    ]
    identical = first_outputs == second_outputs
    print(
        f"repeated-run sanity {split}: identical={identical} "
        f"utterances={len(first_outputs)}"
    )
    if not identical:
        for left, right in zip(first_outputs, second_outputs, strict=True):
            if left != right:
                raise AssertionError(
                    f"Repeated evaluation differs in {split}, "
                    f"utt_id={left[0]}: {left[2]!r} != {right[2]!r}"
                )
        raise AssertionError(f"Repeated evaluation count differs for {split}.")


def _learning_rate_for_adapter(adapter_dir: Path) -> float | None:
    run_config_path = adapter_dir.parent / "run_config.json"
    if run_config_path.is_file():
        with run_config_path.open(encoding="utf-8") as file:
            return float(json.load(file)["learning_rate"])
    return None


def evaluate(
    adapter_dir: Path,
    splits: list[str],
    output_path: Path,
    allow_test: bool,
) -> dict[str, Any]:
    unknown = sorted(set(splits) - set(EVALUATION_SPLITS))
    if unknown:
        raise ValueError(f"Unsupported split(s): {unknown}")
    requested_tests = sorted(set(splits) & set(TEST_SPLITS))
    if requested_tests and not allow_test:
        raise PermissionError(
            f"Refusing test split(s) {requested_tests}; pass --allow-test "
            "only for the final test evaluation."
        )
    if requested_tests:
        print(
            "WARNING: test splits should be used once at the end of the "
            "experiment, after all method/model choices are finalized."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("M4 evaluation requires one CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA GPU does not support bf16.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    print(f"evaluation device: {torch.cuda.get_device_name(device)}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = _read_config()
    model, vocabulary, _, model_id = _load_model(adapter_dir, config)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    budget_report = _adapter_budget(adapter_dir, model)
    feature_extractor, _, loaded_model_id, needs_attention_mask = (
        data._load_model_preprocessing()
    )
    print(
        f"base model={model_id}; preprocessing loaded from {loaded_model_id}; "
        f"attention_mask_required={needs_attention_mask}; "
        "model.eval() disables augmentation; mask_time_prob=0.0"
    )

    result: dict[str, Any] = {
        "adapter": str(adapter_dir),
        "run_name": adapter_dir.parent.name,
        "learning_rate": _learning_rate_for_adapter(adapter_dir),
        "model_id": model_id,
        "splits": {},
        "budget": budget_report,
        "sanity_checks": {
            "reference_labels_as_logits_wer_zero": True,
            "repeat_evaluation_identical": True,
            "wer_is_corpus_jiwer_equal_to_total_errors_over_reference_words": True,
        },
        "wer_method": (
            "Corpus-level jiwer.wer over concatenated reference/hypothesis "
            "lists; not an average of per-utterance WER."
        ),
    }
    for split in splits:
        records = _manifest_records(split)
        if not records:
            raise ValueError(f"Manifest split {split} is empty.")
        _verify_reference_decode(records, vocabulary)
        first = _predict_split(
            model,
            records,
            vocabulary,
            feature_extractor,
            needs_attention_mask,
            float(config["data"]["batch_max_seconds"]),
            device,
        )
        second = _predict_split(
            model,
            records,
            vocabulary,
            feature_extractor,
            needs_attention_mask,
            float(config["data"]["batch_max_seconds"]),
            device,
        )
        _repeatability_check(first, second, split)
        split_result = _score_predictions(first)
        result["splits"][split] = split_result
        print(
            f"{split}: utterances={split_result['utterance_count']} "
            f"reference_words={split_result['reference_word_count']} "
            f"WER={split_result['wer']:.6f} "
            f"S/D/I={split_result['substitutions']}/"
            f"{split_result['deletions']}/{split_result['insertions']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"evaluation results written to {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        choices=EVALUATION_SPLITS,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    adapter_dir = args.adapter
    if not adapter_dir.is_absolute():
        adapter_dir = REPOSITORY_ROOT / adapter_dir
    output_path = args.out
    if not output_path.is_absolute():
        output_path = REPOSITORY_ROOT / output_path
    evaluate(adapter_dir, args.splits, output_path, args.allow_test)


if __name__ == "__main__":
    main()
