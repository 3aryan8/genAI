"""Compare saved learning-rate evaluations without selecting a winner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = REPOSITORY_ROOT / "results" / "eval"


def _read_results(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def main() -> None:
    if not EVALUATION_ROOT.is_dir():
        raise FileNotFoundError(
            f"Evaluation results directory does not exist: {EVALUATION_ROOT}"
        )
    paths = sorted(EVALUATION_ROOT.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No evaluation JSON files found in {EVALUATION_ROOT}"
        )

    rows: list[tuple[str, float, float, float]] = []
    for path in paths:
        result = _read_results(path)
        splits = result.get("splits", {})
        if "dev-clean" not in splits or "dev-other" not in splits:
            continue
        clean_wer = float(splits["dev-clean"]["wer"])
        other_wer = float(splits["dev-other"]["wer"])
        lr = result.get("learning_rate")
        learning_rate = float(lr) if lr is not None else float("nan")
        rows.append((path.stem, learning_rate, clean_wer, other_wer))
    if not rows:
        raise ValueError(
            "No evaluation JSON contains both dev-clean and dev-other results."
        )

    headers = ("lr", "dev-clean WER", "dev-other WER", "mean")
    table = [
        (
            "unknown" if learning_rate != learning_rate else f"{learning_rate:g}",
            f"{clean_wer:.6f}",
            f"{other_wer:.6f}",
            f"{(clean_wer + other_wer) / 2:.6f}",
        )
        for name, learning_rate, clean_wer, other_wer in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in table))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print("No learning-rate winner was selected.")


if __name__ == "__main__":
    main()
