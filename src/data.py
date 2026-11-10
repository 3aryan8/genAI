"""Build Libri-Light/LibriSpeech manifests and inspect CTC-ready batches."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import string
from typing import Any, Iterator

from transformers import HubertForCTC

import soundfile as sf
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config.yaml"
EVALUATION_SPLITS = ("dev-clean", "dev-other", "test-clean", "test-other")
SAMPLE_RATE = 16_000
DEBUG_TRAIN_UTTERANCES = 50
DEBUG_EVAL_UTTERANCES = 20
ALLOWED_CHARACTERS = set(string.ascii_uppercase + "' ")


@dataclass(frozen=True)
class AudioRecord:
    utt_id: str
    path: Path
    duration_sec: float
    text: str
    split: str
    sample_rate: int
    channels: int
    has_transcript: bool
    raw_text: str

    @property
    def speaker_id(self) -> str:
        return self.utt_id.split("-", maxsplit=1)[0]

    @property
    def audio_is_valid(self) -> bool:
        return self.sample_rate == SAMPLE_RATE and self.channels == 1


class CharacterVocabulary:
    def __init__(self, training_texts: list[str]) -> None:
        symbols = {
            "|" if character == " " else character
            for text in training_texts
            for character in text
        }
        symbols.update({"'", "|"})
        self.tokens = ["<pad>", "<unk>", *sorted(symbols)]
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = dict(enumerate(self.tokens))
        self.pad_id = self.token_to_id["<pad>"]
        self.unknown_id = self.token_to_id["<unk>"]

    def encode(self, text: str) -> list[int]:
        return [
            self.token_to_id.get("|" if character == " " else character, self.unknown_id)
            for character in text
        ]

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        characters = [
            self.id_to_token.get(int(token_id), "<unk>")
            for token_id in token_ids
            if int(token_id) not in (self.pad_id, -100)
        ]
        return "".join(characters).replace("|", " ").replace("<unk>", "\ufffd")


def _read_transcripts(root: Path) -> tuple[dict[str, str], int]:
    transcripts: dict[str, str] = {}
    blank_lines = 0
    for path in sorted(root.rglob("*.trans.txt")):
        with path.open(encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    blank_lines += 1
                    continue
                fields = line.split(maxsplit=1)
                utt_id = fields[0]
                text = fields[1] if len(fields) == 2 else ""
                if utt_id in transcripts and transcripts[utt_id] != text:
                    raise ValueError(
                        f"Conflicting transcript text for {utt_id} at "
                        f"{path}:{line_number}"
                    )
                transcripts[utt_id] = text
    return transcripts, blank_lines


def _clean_text(
    text: str, outside_characters: Counter[str]
) -> str:
    uppercase = text.upper()
    cleaned_characters: list[str] = []
    for character in uppercase:
        if character in ALLOWED_CHARACTERS:
            cleaned_characters.append(character)
        else:
            outside_characters[character] += 1
            cleaned_characters.append(" ")
    return re.sub(r"\s+", " ", "".join(cleaned_characters)).strip()


def _scan_split(
    root: Path,
    split: str,
    outside_characters: Counter[str],
) -> tuple[list[AudioRecord], dict[str, int]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    transcripts, blank_transcript_lines = _read_transcripts(root)
    audio_paths = sorted(root.rglob("*.flac"))
    audio_ids = {path.stem for path in audio_paths}
    orphan_transcripts = set(transcripts) - audio_ids
    records: list[AudioRecord] = []
    empty_or_missing = 0

    for audio_path in audio_paths:
        raw_text = transcripts.get(audio_path.stem)
        has_transcript = raw_text is not None
        text = _clean_text(raw_text or "", outside_characters)
        if not has_transcript or not text:
            empty_or_missing += 1

        try:
            info = sf.info(audio_path)
        except RuntimeError as error:
            raise RuntimeError(f"Could not inspect audio metadata: {audio_path}") from error
        records.append(
            AudioRecord(
                utt_id=audio_path.stem,
                path=audio_path,
                duration_sec=info.frames / info.samplerate,
                text=text,
                split=split,
                sample_rate=info.samplerate,
                channels=info.channels,
                has_transcript=has_transcript,
                raw_text=raw_text or "",
            )
        )

    return records, {
        "empty_or_missing": empty_or_missing,
        "blank_transcript_lines": blank_transcript_lines,
        "orphan_transcripts": len(orphan_transcripts),
    }


def _make_train_union(
    records_1h: list[AudioRecord], records_9h: list[AudioRecord]
) -> tuple[list[AudioRecord], int]:
    train_by_id: dict[str, AudioRecord] = {}
    overlap = 0
    for record in [*records_1h, *records_9h]:
        previous = train_by_id.get(record.utt_id)
        if previous is not None:
            overlap += 1
            if (
                previous.text != record.text
                or previous.duration_sec != record.duration_sec
            ):
                raise ValueError(
                    f"Conflicting duplicate training utterance: {record.utt_id}"
                )
            continue
        train_by_id[record.utt_id] = AudioRecord(
            utt_id=record.utt_id,
            path=record.path,
            duration_sec=record.duration_sec,
            text=record.text,
            split="train10h",
            sample_rate=record.sample_rate,
            channels=record.channels,
            has_transcript=record.has_transcript,
            raw_text=record.raw_text,
        )
    return list(train_by_id.values()), overlap


def _write_manifest(name: str, records: list[AudioRecord]) -> None:
    manifest_dir = REPOSITORY_ROOT / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{name}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as file:
        for record in records:
            try:
                stored_path = record.path.resolve().relative_to(REPOSITORY_ROOT)
            except ValueError:
                stored_path = record.path.resolve()
            row = {
                "utt_id": record.utt_id,
                "path": stored_path.as_posix(),
                "duration_sec": record.duration_sec,
                "text": record.text,
                "split": record.split,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_vocabulary(vocabulary: CharacterVocabulary) -> None:
    vocab_path = REPOSITORY_ROOT / "data" / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(
        json.dumps(
            {
                "tokens": vocabulary.tokens,
                "token_to_id": vocabulary.token_to_id,
                "pad_token": "<pad>",
                "blank_token": "<pad>",
                "unknown_token": "<unk>",
                "word_delimiter_token": "|",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_model_preprocessing() -> tuple[Any, Any, str, bool]:
    from transformers import AutoConfig, AutoFeatureExtractor

    model_ids = ("facebook/HuBERT-base-ls960", "facebook/hubert-base-ls960")
    failures: list[tuple[str, Exception]] = []
    for model_id in model_ids:
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            model_config = AutoConfig.from_pretrained(model_id)
        except Exception as error:
            failures.append((model_id, error))
            continue

        feature_norm = getattr(model_config, "feat_extract_norm", None)
        if feature_norm == "layer":
            needs_attention_mask = True
        elif feature_norm == "group":
            needs_attention_mask = False
        else:
            needs_attention_mask = bool(
                getattr(feature_extractor, "return_attention_mask", False)
            )
        return feature_extractor, model_config, model_id, needs_attention_mask

    failure_details = "; ".join(
        f"{model_id}: {error}" for model_id, error in failures
    )
    raise RuntimeError(
        "Could not load HuBERT preprocessing/model config from either requested "
        f"Hub id. Attempts: {failure_details}"
    )


class AudioTextDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[AudioRecord],
        vocabulary: CharacterVocabulary,
        feature_extractor: Any,
    ) -> None:
        self.records = records
        self.vocabulary = vocabulary
        self.feature_extractor = feature_extractor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if not record.audio_is_valid:
            raise ValueError(
                f"Cannot batch non-16-kHz mono audio: {record.path} "
                f"({record.sample_rate} Hz, {record.channels} channel(s))"
            )
        if not record.has_transcript or not record.text:
            raise ValueError(f"Cannot batch empty/missing transcript: {record.utt_id}")

        waveform, sample_rate = sf.read(record.path, dtype="float32")
        if sample_rate != SAMPLE_RATE or waveform.ndim != 1:
            raise ValueError(
                f"Audio changed after manifest scan or is not mono 16 kHz: "
                f"{record.path}"
            )
        features = self.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        return {
            "utt_id": record.utt_id,
            "input_values": features["input_values"].squeeze(0),
            "labels": torch.tensor(self.vocabulary.encode(record.text), dtype=torch.long),
            "text": record.text,
            "path": str(record.path),
            "duration_sec": record.duration_sec,
        }


def collate_batch(
    examples: list[dict[str, Any]],
    padding_value: float,
    needs_attention_mask: bool,
) -> dict[str, Any]:
    input_lengths = torch.tensor(
        [example["input_values"].numel() for example in examples], dtype=torch.long
    )
    labels = pad_sequence(
        [example["labels"] for example in examples],
        batch_first=True,
        padding_value=-100,
    )
    input_values = pad_sequence(
        [example["input_values"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    batch: dict[str, Any] = {
        "input_values": input_values,
        "input_lengths": input_lengths,
        "labels": labels,
        "label_lengths": torch.tensor(
            [example["labels"].numel() for example in examples], dtype=torch.long
        ),
        "texts": [example["text"] for example in examples],
        "utt_ids": [example["utt_id"] for example in examples],
        "paths": [example["path"] for example in examples],
        "total_seconds": sum(example["duration_sec"] for example in examples),
    }
    if needs_attention_mask:
        batch["attention_mask"] = (
            torch.arange(input_values.shape[1]).unsqueeze(0)
            < input_lengths.unsqueeze(1)
        )
    return batch


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self, records: list[AudioRecord], max_seconds: float, seed: int
    ) -> None:
        self.records = records
        self.max_seconds = max_seconds
        self.seed = seed
        self.batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        if self.max_seconds <= 0:
            raise ValueError("batch_max_seconds must be greater than zero.")
        ordered_indices = sorted(
            range(len(self.records)),
            key=lambda index: (self.records[index].duration_sec, index),
        )
        batches: list[list[int]] = []
        batch: list[int] = []
        batch_seconds = 0.0
        for index in ordered_indices:
            duration = self.records[index].duration_sec
            if duration > self.max_seconds:
                raise ValueError(
                    f"Utterance {self.records[index].utt_id} is {duration:.3f}s, "
                    f"longer than batch_max_seconds={self.max_seconds:.3f}."
                )
            if batch and batch_seconds + duration > self.max_seconds:
                batches.append(batch)
                batch = []
                batch_seconds = 0.0
            batch.append(index)
            batch_seconds += duration
        if batch:
            batches.append(batch)

        random.Random(self.seed).shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {CONFIG_PATH}")
    return config


def _print_split_statistics(
    name: str, records: list[AudioRecord], issues: dict[str, int]
) -> None:
    durations = [record.duration_sec for record in records]
    hours = sum(durations) / 3600
    if durations:
        minimum, maximum = min(durations), max(durations)
        mean = sum(durations) / len(durations)
        duration_summary = (
            f"min={minimum:.3f}s max={maximum:.3f}s mean={mean:.3f}s"
        )
    else:
        duration_summary = "min=n/a max=n/a mean=n/a"
    rates = sorted({record.sample_rate for record in records})
    print(
        f"{name}: utterances={len(records)} hours={hours:.4f} "
        f"{duration_summary} sampling_rates={rates}"
    )
    print(
        f"  empty_or_missing_transcripts={issues['empty_or_missing']} "
        f"blank_transcript_lines={issues['blank_transcript_lines']} "
        f"orphan_transcript_ids={issues['orphan_transcripts']}"
    )


def _print_long_word_count(name: str, records: list[AudioRecord]) -> None:
    long_words = [
        word
        for record in records
        for word in record.text.split()
        if sum(character in string.ascii_uppercase for character in word) >= 14
    ]
    print(
        f"long words (>=14 letters) {name}: occurrences={len(long_words)} "
        f"unique={len(set(long_words))}"
    )


def _verify_text_round_trip(
    name: str,
    records: list[AudioRecord],
    vocabulary: CharacterVocabulary,
    seed: int,
) -> None:
    eligible_records = [
        record for record in records if record.has_transcript
    ]
    sample = random.Random(seed).sample(
        eligible_records, min(500, len(eligible_records))
    )
    for record in sample:
        normalized = _clean_text(record.raw_text, Counter())
        decoded = vocabulary.decode(vocabulary.encode(normalized))
        if decoded != normalized:
            raise AssertionError(
                f"Text round-trip mismatch in {name}, utt_id={record.utt_id}: "
                f"normalized={normalized!r}, decoded={decoded!r}"
            )
    print(f"text round-trip {name}: passed {len(sample)} utterances")


def _print_audio_violations(
    split_records: dict[str, list[AudioRecord]]
) -> None:
    unique_records = {
        record.path: record
        for records in split_records.values()
        for record in records
    }.values()
    invalid_audio = [
        record
        for record in unique_records
        if record.sample_rate != SAMPLE_RATE or record.channels != 1
    ]
    invalid_rate_count = sum(
        record.sample_rate != SAMPLE_RATE for record in invalid_audio
    )
    invalid_channel_count = sum(record.channels != 1 for record in invalid_audio)
    print(
        f"audio validation: non-16000-Hz={invalid_rate_count} "
        f"non-mono={invalid_channel_count}"
    )
    for record in invalid_audio:
        print(
            f"  invalid audio: {record.path} "
            f"({record.sample_rate} Hz, {record.channels} channel(s))"
        )


def _print_leakage(
    train_records: list[AudioRecord],
    evaluation_records: dict[str, list[AudioRecord]],
) -> None:
    train_ids = {record.utt_id for record in train_records}
    train_speakers = {record.speaker_id for record in train_records}
    for split, records in evaluation_records.items():
        shared_ids = train_ids & {record.utt_id for record in records}
        shared_speakers = train_speakers & {
            record.speaker_id for record in records
        }
        print(
            f"leakage {split}: shared_utt_ids={len(shared_ids)} "
            f"shared_speaker_ids={len(shared_speakers)}"
        )


def _build_audio_loader(
    records: list[AudioRecord],
    vocabulary: CharacterVocabulary,
    feature_extractor: Any,
    max_seconds: float,
    seed: int,
    needs_attention_mask: bool,
) -> DataLoader[dict[str, Any]]:
    eligible_records = [
        record
        for record in records
        if record.audio_is_valid and record.has_transcript and record.text
    ]
    dataset = AudioTextDataset(eligible_records, vocabulary, feature_extractor)
    batch_sampler = LengthBucketBatchSampler(eligible_records, max_seconds, seed)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        collate_fn=lambda examples: collate_batch(
            examples,
            padding_value=float(feature_extractor.padding_value),
            needs_attention_mask=needs_attention_mask,
        ),
    )


def _read_manifest_texts(name: str) -> dict[str, str]:
    manifest_path = REPOSITORY_ROOT / "data" / "manifests" / f"{name}.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing existing manifest {manifest_path}; run the earlier M1 "
            "manifest-generation step before verification. Verification does "
            "not regenerate manifests."
        )
    texts: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)
            utt_id = row["utt_id"]
            if utt_id in texts:
                raise ValueError(
                    f"Duplicate utt_id {utt_id} in {manifest_path}:{line_number}"
                )
            texts[utt_id] = row["text"]
    return texts


def _compare_batch_labels_to_manifest(
    name: str,
    loader: DataLoader[dict[str, Any]],
    vocabulary: CharacterVocabulary,
) -> None:
    manifest_texts = _read_manifest_texts(name)
    seen_ids: set[str] = set()
    mismatches: list[tuple[str, str, str]] = []
    mismatch_count = 0
    missing_manifest_rows: list[str] = []
    for batch in loader:
        for index, utt_id in enumerate(batch["utt_ids"]):
            seen_ids.add(utt_id)
            manifest_text = manifest_texts.get(utt_id)
            if manifest_text is None:
                missing_manifest_rows.append(utt_id)
                continue
            decoded_text = vocabulary.decode(batch["labels"][index])
            if decoded_text != manifest_text:
                mismatch_count += 1
                if len(mismatches) < 5:
                    mismatches.append((utt_id, manifest_text, decoded_text))

    missing_from_loader = set(manifest_texts) - seen_ids
    print(
        f"batch-label comparison {name}: "
        f"mismatches={mismatch_count} "
        f"batched_utterances_without_manifest_rows={len(missing_manifest_rows)} "
        f"manifest_utterances_not_batched={len(missing_from_loader)}"
    )
    for utt_id, manifest_text, decoded_text in mismatches:
        print(
            f"  mismatch {utt_id}: manifest={manifest_text!r}; "
            f"decoded={decoded_text!r}"
        )
    for utt_id in missing_manifest_rows[:5]:
        print(f"  missing manifest row for batched utt_id={utt_id}")


def _hash_label_tensors(
    loader: DataLoader[dict[str, Any]],
) -> list[str]:
    hashes: list[str] = []
    for batch in loader:
        labels = batch["labels"].detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(labels.shape)).encode("ascii"))
        digest.update(str(labels.dtype).encode("ascii"))
        digest.update(labels.numpy().tobytes())
        hashes.append(digest.hexdigest())
    return hashes


def _print_batches(
    loader: DataLoader[dict[str, Any]], vocabulary: CharacterVocabulary
) -> None:
    for batch_number, batch in enumerate(loader):
        if batch_number == 3:
            break
        print(
            f"train batch {batch_number + 1}: "
            f"input_values={tuple(batch['input_values'].shape)} "
            f"labels={tuple(batch['labels'].shape)} "
            f"attention_mask="
            f"{tuple(batch['attention_mask'].shape) if 'attention_mask' in batch else 'not used'} "
            f"total_seconds={batch['total_seconds']:.3f}"
        )
        for index, text in enumerate(batch["texts"][:2]):
            decoded = vocabulary.decode(batch["labels"][index])
            print(f"  example {index + 1} ({batch['utt_ids'][index]}): {decoded}")


def _print_preprocessing(
    feature_extractor: Any,
    model_config: Any,
    model_id: str,
    needs_attention_mask: bool,
) -> None:
    print(f"preprocessing model id: {model_id}")
    print(
        "feature extractor settings: "
        f"do_normalize={getattr(feature_extractor, 'do_normalize', None)}, "
        f"return_attention_mask="
        f"{getattr(feature_extractor, 'return_attention_mask', None)}, "
        f"feature_size={getattr(feature_extractor, 'feature_size', None)}, "
        f"sampling_rate={getattr(feature_extractor, 'sampling_rate', None)}, "
        f"padding_value={getattr(feature_extractor, 'padding_value', None)}"
    )
    print(
        f"model feature-extractor norm="
        f"{getattr(model_config, 'feat_extract_norm', None)}; "
        f"attention_mask_required={needs_attention_mask}"
    )


def main() -> None:
    print(f"HubertForCTC import: OK ({HubertForCTC.__name__})")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--write-manifests",
        action="store_true",
        help="Write manifests and vocabulary; verification is read-only by default.",
    )
    args = parser.parse_args()

    config = _load_config()
    data_config = config["data"]
    train_root = REPOSITORY_ROOT / data_config["librilight_root"]
    librispeech_root = REPOSITORY_ROOT / data_config["librispeech_root"]
    max_seconds = float(data_config["batch_max_seconds"])
    seed = int(config["experiment"]["seeds"][0])

    outside_characters: Counter[str] = Counter()
    records_1h, issues_1h = _scan_split(
        train_root / "1h", "1h", outside_characters
    )
    records_9h, issues_9h = _scan_split(
        train_root / "9h", "9h", outside_characters
    )
    train_records, train_overlap = _make_train_union(records_1h, records_9h)

    evaluation_records: dict[str, list[AudioRecord]] = {}
    evaluation_issues: dict[str, dict[str, int]] = {}
    for split in EVALUATION_SPLITS:
        evaluation_records[split], evaluation_issues[split] = _scan_split(
            librispeech_root / split, split, outside_characters
        )

    issues_by_split = {
        "1h": issues_1h,
        "9h": issues_9h,
        "train10h": {
            key: issues_1h[key] + issues_9h[key]
            for key in issues_1h
        },
        **evaluation_issues,
    }
    split_records = {
        "1h": records_1h,
        "9h": records_9h,
        "train10h": train_records,
        **evaluation_records,
    }

    for name in ("train10h", "1h", "9h", *EVALUATION_SPLITS):
        _print_split_statistics(name, split_records[name], issues_by_split[name])
        _print_long_word_count(name, split_records[name])
    print(f"utt_id shared between 1h and 9h: {train_overlap}")
    _print_audio_violations(split_records)
    all_sampling_rates = sorted(
        {
            record.sample_rate
            for records in split_records.values()
            for record in records
        }
    )
    print(f"all sampling rates found: {all_sampling_rates}")

    valid_training_texts = [
        record.text
        for record in train_records
        if record.has_transcript and record.text
    ]
    vocabulary = CharacterVocabulary(valid_training_texts)
    print(f"characters outside allowed set: {dict(sorted(outside_characters.items()))}")
    print(f"vocabulary size: {len(vocabulary.tokens)}")
    print(f"vocabulary: {vocabulary.tokens}")

    _print_leakage(train_records, evaluation_records)
    if args.write_manifests:
        _write_manifest("train10h", train_records)
        for split, records in evaluation_records.items():
            _write_manifest(split, records)
        _write_vocabulary(vocabulary)

    for name, records in (
        ("train10h", train_records),
        ("1h", records_1h),
        ("9h", records_9h),
        *evaluation_records.items(),
    ):
        _verify_text_round_trip(name, records, vocabulary, seed)

    feature_extractor, model_config, loaded_model_id, needs_attention_mask = (
        _load_model_preprocessing()
    )
    _print_preprocessing(
        feature_extractor, model_config, loaded_model_id, needs_attention_mask
    )

    if args.debug:
        train_records = train_records[:DEBUG_TRAIN_UTTERANCES]
        for split, records in evaluation_records.items():
            evaluation_records[split] = records[:DEBUG_EVAL_UTTERANCES]
        print(
            "debug subsets: "
            f"train10h={len(train_records)}; "
            + ", ".join(
                f"{split}={len(records)}"
                for split, records in evaluation_records.items()
            )
        )
        for split, records in evaluation_records.items():
            evaluation_loader = _build_audio_loader(
                records,
                vocabulary,
                feature_extractor,
                max_seconds,
                seed,
                needs_attention_mask,
            )
            loaded_examples = sum(
                len(batch["texts"]) for batch in evaluation_loader
            )
            print(f"debug {split}: loaded {loaded_examples} audio-text examples")

    loaders = {
        "train10h": _build_audio_loader(
            train_records,
            vocabulary,
            feature_extractor,
            max_seconds,
            seed,
            needs_attention_mask,
        )
    }
    loaders.update(
        {
            split: _build_audio_loader(
                records,
                vocabulary,
                feature_extractor,
                max_seconds,
                seed,
                needs_attention_mask,
            )
            for split, records in evaluation_records.items()
        }
    )

    for name, loader in loaders.items():
        _compare_batch_labels_to_manifest(name, loader, vocabulary)

    train_loader = loaders["train10h"]
    first_hashes = _hash_label_tensors(train_loader)
    second_train_loader = _build_audio_loader(
        train_records,
        vocabulary,
        feature_extractor,
        max_seconds,
        seed,
        needs_attention_mask,
    )
    second_hashes = _hash_label_tensors(second_train_loader)
    print(
        "train label batch hashes identical across same-seed builds: "
        f"{first_hashes == second_hashes} "
        f"(batches={len(first_hashes)})"
    )

    _print_batches(train_loader, vocabulary)
    if len(train_loader) < 3:
        raise ValueError(
            f"Need at least three eligible train10h batches; found "
            f"{len(train_loader)}. "
            "Check the audio/transcript reports and batch_max_seconds."
        )
    print(
        f"manifests: {REPOSITORY_ROOT / 'data' / 'manifests'}; "
        f"vocabulary: {REPOSITORY_ROOT / 'data' / 'vocab.json'}"
    )


if __name__ == "__main__":
    main()
