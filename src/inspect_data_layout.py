"""Inspect extracted dataset layout and transcript samples without decoding audio."""

from __future__ import annotations

from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data"
DATASETS = {
    "train-1h": ROOT / "librispeech_finetuning" / "1h",
    "train-9h": ROOT / "librispeech_finetuning" / "9h",
    "dev-clean": ROOT / "LibriSpeech" / "dev-clean",
    "dev-other": ROOT / "LibriSpeech" / "dev-other",
    "test-clean": ROOT / "LibriSpeech" / "test-clean",
    "test-other": ROOT / "LibriSpeech" / "test-other",
}


def inspect_dataset(name: str, root: Path) -> None:
    print(f"\n[{name}] {root} exists={root.is_dir()}")
    if not root.is_dir():
        return

    top_level = sorted(path.name + ("/" if path.is_dir() else "") for path in root.iterdir())
    print(f"top-level entries ({len(top_level)}): {top_level[:12]}")

    extensions: Counter[str] = Counter()
    samples: list[Path] = []
    transcript_samples: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = "".join(path.suffixes).lower() or "<no extension>"
        extensions[suffix] += 1
        if len(samples) < 8:
            samples.append(path)
        if len(transcript_samples) < 2 and path.suffix.lower() == ".txt":
            transcript_samples.append(path)

    print(f"file extensions/counts: {dict(sorted(extensions.items()))}")
    print("sample paths:")
    for path in samples:
        print(f"  {path.relative_to(ROOT)}")
    for path in transcript_samples:
        print(f"transcript/text sample: {path.relative_to(ROOT)}")
        try:
            with path.open(encoding="utf-8") as file:
                for _, line in zip(range(3), file):
                    print(f"  {line.rstrip()}")
        except UnicodeDecodeError:
            print("  [not UTF-8 text]")


def main() -> None:
    for name, root in DATASETS.items():
        inspect_dataset(name, root)


if __name__ == "__main__":
    main()
