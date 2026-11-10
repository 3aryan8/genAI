"""Print evaluation loss and WER rows from saved training-run metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPOSITORY_ROOT / "results" / "runs"


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _evaluation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = [
        row
        for row in rows
        if row.get("type") == "step" and isinstance(row.get("step"), int)
    ]
    evaluations = [
        row
        for row in rows
        if row.get("type") == "evaluation"
        and isinstance(row.get("dev_clean_wer"), (int, float))
    ]
    if not evaluations:
        evaluations = [
            row
            for row in rows
            if row.get("type") == "epoch"
            and isinstance(row.get("dev_clean_wer"), (int, float))
        ]

    result: list[dict[str, Any]] = []
    for evaluation in evaluations:
        step = evaluation.get("step")
        loss = evaluation.get("train_loss")
        if loss is None and isinstance(step, int):
            previous_steps = [row for row in steps if row["step"] <= step]
            if previous_steps:
                loss = max(previous_steps, key=lambda row: row["step"]).get("loss")
        result.append(
            {
                "epoch": evaluation.get("epoch")
                if evaluation.get("epoch") is not None
                else "?",
                "step": step if step is not None else "?",
                "loss": loss,
                "wer": evaluation["dev_clean_wer"],
            }
        )
    return result


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    if not RUNS_ROOT.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {RUNS_ROOT}")
    metric_paths = sorted(RUNS_ROOT.glob("*/metrics.jsonl"))
    if not metric_paths:
        raise FileNotFoundError(f"No metrics.jsonl files found under {RUNS_ROOT}")

    headers = ("run", "epoch", "step", "train loss", "dev-clean WER")
    table_rows: list[tuple[str, str, str, str, str]] = []
    for metric_path in metric_paths:
        evals = _evaluation_rows(_read_metrics(metric_path))
        if not evals:
            table_rows.append((metric_path.parent.name, "-", "-", "n/a", "n/a"))
            continue
        for item in evals:
            table_rows.append(
                (
                    metric_path.parent.name,
                    str(item["epoch"]),
                    str(item["step"]),
                    _format(item["loss"]),
                    _format(item["wer"]),
                )
            )

    widths = [
        max(len(header), *(len(row[index]) for row in table_rows))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


if __name__ == "__main__":
    main()
