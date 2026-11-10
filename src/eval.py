"""Greedy CTC decoding and WER evaluation."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import time
from typing import Any

import jiwer
import soundfile as sf
import torch

from src import data


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = REPOSITORY_ROOT / "data" / "vocab.json"


@lru_cache(maxsize=1)
def _evaluation_resources() -> tuple[
    dict[str, Any], data.CharacterVocabulary, Any, bool
]:
    config = data._load_config()
    with VOCAB_PATH.open(encoding="utf-8") as file:
        vocab_record = json.load(file)
    vocabulary = data.CharacterVocabulary([])
    vocabulary.tokens = vocab_record["tokens"]
    vocabulary.token_to_id = vocab_record["token_to_id"]
    vocabulary.id_to_token = dict(enumerate(vocabulary.tokens))
    vocabulary.pad_id = vocab_record["token_to_id"][vocab_record["pad_token"]]
    vocabulary.unknown_id = vocab_record["token_to_id"][vocab_record["unknown_token"]]
    feature_extractor, _, _, needs_attention_mask = data._load_model_preprocessing()
    return config, vocabulary, feature_extractor, needs_attention_mask


def _manifest_records(split: str) -> list[data.AudioRecord]:
    if split not in ("train10h", *data.EVALUATION_SPLITS):
        raise ValueError(f"Unknown manifest split: {split}")
    path = REPOSITORY_ROOT / "data" / "manifests" / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing M1 manifest: {path}")

    records: list[data.AudioRecord] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)
            audio_path = Path(row["path"])
            if not audio_path.is_absolute():
                audio_path = REPOSITORY_ROOT / audio_path
            info = sf.info(audio_path)
            records.append(
                data.AudioRecord(
                    utt_id=row["utt_id"],
                    path=audio_path,
                    duration_sec=float(row["duration_sec"]),
                    text=row["text"],
                    split=row["split"],
                    sample_rate=info.samplerate,
                    channels=info.channels,
                    has_transcript=bool(row["text"]),
                    raw_text=row["text"],
                )
            )
            if row["split"] != split:
                raise ValueError(
                    f"Manifest split mismatch at {path}:{line_number}: "
                    f"{row['split']} != {split}"
                )
    return records


def decode_ctc_ids(
    token_ids: list[int] | torch.Tensor, vocabulary: data.CharacterVocabulary
) -> str:
    """Collapse consecutive repeats, remove CTC blank, and decode characters."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id != previous and token_id != vocabulary.pad_id:
            collapsed.append(token_id)
        previous = token_id
    return vocabulary.decode(collapsed)


def decode_label_ids(
    token_ids: list[int] | torch.Tensor, vocabulary: data.CharacterVocabulary
) -> str:
    """Decode CTC targets directly; target labels are not prediction frames."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return vocabulary.decode([int(token_id) for token_id in token_ids if int(token_id) != -100])


def verify_label_round_trip(
    batches: list[dict[str, Any]], vocabulary: data.CharacterVocabulary
) -> float:
    references: list[str] = []
    decoded: list[str] = []
    for batch in batches:
        for labels, text in zip(batch["labels"], batch["texts"], strict=True):
            references.append(text)
            decoded.append(decode_label_ids(labels, vocabulary))
    score = float(jiwer.wer(references, decoded))
    print(f"batch-label round-trip WER={score:.6f} examples={len(references)}")
    if score != 0.0:
        mismatch = next(
            (reference, hypothesis)
            for reference, hypothesis in zip(references, decoded, strict=True)
            if reference != hypothesis
        )
        raise AssertionError(
            f"Batch label round-trip failed: reference={mismatch[0]!r}, "
            f"decoded={mismatch[1]!r}"
        )
    return score


def evaluate_wer(
    model: torch.nn.Module,
    split: str,
    max_utts: int | None = None,
    batch_max_seconds: float | None = None,
) -> float:
    """Evaluate greedy CTC WER from the existing M1 manifest."""
    wer, _ = evaluate_wer_with_examples(
        model,
        split,
        max_utts=max_utts,
        batch_max_seconds=batch_max_seconds,
        example_count=0,
    )
    return wer


def evaluate_wer_with_examples(
    model: torch.nn.Module,
    split: str,
    max_utts: int,
    batch_max_seconds: float | None = None,
    example_count: int = 3,
) -> tuple[float, list[tuple[str, str]]]:
    """Evaluate WER and retain a few deterministic reference/prediction pairs."""
    config, vocabulary, feature_extractor, needs_attention_mask = (
        _evaluation_resources()
    )
    records = _manifest_records(split)
    if max_utts is not None:
        if max_utts <= 0:
            raise ValueError("max_utts must be positive when provided.")
        records = records[:max_utts]
    if not records:
        raise ValueError(f"Manifest split {split!r} contains no utterances.")

    requested_batch_seconds = float(
        batch_max_seconds
        if batch_max_seconds is not None
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
        int(config["experiment"]["seeds"][0]),
        needs_attention_mask,
    )
    device = next(model.parameters()).device
    references: list[str] = []
    hypotheses: list[str] = []
    examples: list[tuple[str, str]] = []
    was_training = model.training
    model.eval()
    try:
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
                logits = model(**inputs).logits
                frame_lengths = model._get_feat_extract_output_lengths(
                    batch["input_lengths"].to(device=device)
                )
                predictions = logits.argmax(dim=-1)
                for prediction, frame_length, reference in zip(
                    predictions, frame_lengths, batch["texts"], strict=True
                ):
                    hypothesis = decode_ctc_ids(
                        prediction[: int(frame_length.item())], vocabulary
                    )
                    hypotheses.append(hypothesis)
                    references.append(reference)
                    if len(examples) < example_count:
                        examples.append((reference, hypothesis))
    finally:
        model.train(was_training)
    return float(jiwer.wer(references, hypotheses)), examples
