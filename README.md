# HuBERT LoRA rank allocation for speech recognition

This repository studies how LoRA rank allocation affects CTC speech recognition with [HuBERT base](https://huggingface.co/facebook/hubert-base-ls960). It compares uniform ranks with [PARA](https://arxiv.org/abs/2604.27796) and [LAARA](https://arxiv.org/abs/2607.19391). The main budget reference is uniform rank 4: **663,552 LoRA parameters**, plus a 23,070-parameter trainable CTC head.

The M7 comparison covers LibriSpeech dev-clean, dev-other, test-clean, and test-other for seeds 0, 1, and 2. M8 rank analysis code is present; phone-label CCA and correlations still need execution and numerical verification. See [PROGRESS.md](PROGRESS.md) for milestone status.

## Results

The table reports **WER as a fraction**, mean ± sample standard deviation across converged seeds. Values come from [the observed M7 aggregate](results/m7_table.md), which also includes every per-seed result and parameter count. Standard deviation is unavailable with one converged seed.

| Method | LoRA parameters | Converged seeds | dev-clean | dev-other | test-clean | test-other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform r4 (budget reference) | 663,552 | 3/3 | 0.130056 ± 0.001933 | 0.212995 ± 0.003624 | 0.130560 ± 0.001492 | 0.215616 ± 0.003675 |
| PARA | 662,784 / 662,784 / 663,552 | 3/3 | 0.200930 ± 0.006240 | 0.293103 ± 0.007215 | 0.205487 ± 0.005651 | 0.296595 ± 0.007154 |
| LAARA (over budget) | 941,568 / 861,696 / 947,712 | 3/3 | 0.119260 ± 0.003107 | 0.198497 ± 0.005209 | 0.120657 ± 0.004026 | 0.200313 ± 0.004601 |
| Uniform r16 (upper reference) | 2,654,208 | 1/3 | 0.107202 (seed 0) | 0.189468 (seed 0) | 0.109860 (seed 0) | 0.192729 (seed 0) |
| Uniform r3 (extra) | 497,664 | 3/3 | 0.138334 ± 0.001061 | 0.222541 ± 0.001568 | 0.141509 ± 0.001402 | 0.227557 ± 0.002578 |

Uniform r16 seeds 1 and 2 **stalled at blank output** (dev-clean WER 1.0, loss about 2.85, no NaN). Their per-seed WERs remain in the aggregate and are excluded from the converged-run mean. LAARA exceeded the rank-4 LoRA budget by **+41.9%, +29.9%, and +42.8%** for seeds 0–2, respectively. Its WER is therefore **not an equal-budget comparison**. PARA needs a separately trained rank-16 parent for each seed, plus pruning; these costs are recorded in the [aggregate notes](results/m7_table.md). The [budget table](results/tables/budget_table.md) and [paired differences](results/tables/paired_differences.md) provide additional detail.

The extra uniform r5 and r6 controls have completed **seed-0 dev learning-rate sweeps only**. Both selected `1e-3`: r5 dev-clean/dev-other WER `0.124260 / 0.204346`, and r6 `0.119297 / 0.201480`. Their seed-1/2 and four-set comparisons are **on hold**, so no test result is reported for them.

### Findings and practical recommendations

- **Use uniform r4 when the 663,552-parameter LoRA budget is a hard limit.** It converged for all three seeds and is the completed budget-reference method.
- **Do not describe the current LAARA result as an equal-budget win.** It has the lowest observed WER across the four splits, but its three runs exceed the LoRA budget by 29.9%–42.8%. A budget-capped LAARA variant needs to be implemented and evaluated before making a same-budget claim.
- **The evaluated PARA configuration did not match the uniform-r4 WER.** Its three-seed dev-clean and dev-other means are 0.200930 and 0.293103 versus 0.130056 and 0.212995 for uniform r4. Its near-matched LoRA parameter count does not make it an accuracy improvement.
- **Treat uniform r16 as an unstable result, not a three-seed baseline.** Only seed 0 converged; seeds 1 and 2 stalled at blank output.
- **Keep r5/r6 exploratory.** Their seed-0 development sweeps are informative, but the four-split, three-seed evaluation is on hold.

These recommendations describe the observed runs only. The pending M8 analysis has not established a causal explanation for the rank-allocation results.

### Figures

![Test WER for the five completed M7 methods](results/figures/fig2_wer_by_method.png)

*Test WER across converged runs. Dots are seeds; error bars show sample standard deviation where n > 1. LAARA is over budget; r16 has only one converged seed. [PDF](results/figures/fig2_wer_by_method.pdf).*

![WER versus LoRA parameter count](results/figures/fig3_wer_vs_params.png)

*Observed WER and LoRA parameter counts. Read LAARA as an over-budget result, not an equal-budget comparison. [PDF](results/figures/fig3_wer_vs_params.pdf).*

![PARA and LAARA mean ranks across encoder layers and module types](results/figures/fig11a_rank_heatmaps.png)

*Observed rank allocation averaged over three seeds, shown on a shared color scale. LAARA has more LoRA parameters than the rank-4 budget. [PDF](results/figures/fig11a_rank_heatmaps.pdf) · [per-layer rank plot](results/figures/fig11b_rank_per_layer.png).*

![Per-layer rank allocation](results/figures/fig11b_rank_per_layer.png)

*Mean rank by encoder layer and module type for PARA and LAARA. [PDF](results/figures/fig11b_rank_per_layer.pdf).*

![Training-loss curves including stalled uniform-r16 seeds](results/figures/fig6_loss_curves.png)

*Logged training loss illustrates the stalled r16 seeds. [PDF](results/figures/fig6_loss_curves.pdf).*

![LAARA LoRA parameter budget over training](results/figures/fig7_laara_rank_history.png)

*Per-seed LAARA LoRA parameter counts during training, compared with the rank-4 budget. The paper-default runs finish above the budget. [PDF](results/figures/fig7_laara_rank_history.pdf).*

## Experiment protocol

- **Data:** LibriSpeech 10-hour fine-tuning subset (2,763 utterances, 9.7468 hours), with the four official development/test splits. Local layout: [data/README.md](data/README.md).
- **Model and targets:** `facebook/hubert-base-ls960` with a new CTC head; six LoRA target modules in each of 12 encoder layers. Greedy decoding, no language model.
- **Training:** 30 epochs, last-epoch adapter, seeds 0/1/2. Select each method's learning rate **on seed 0 only** using the lowest mean of dev-clean and dev-other WER; fix it for later seeds. Test splits are evaluated after selection.
- **Selected rates:** uniform r4 `1e-3`; uniform r16 `1e-3`; PARA's rank-16 parent `3e-4`; LAARA `1e-3`; uniform r3 `1e-3`. The extra r5/r6 seed-0 sweeps also selected `1e-3` but their full comparison is on hold.
- **Budget accounting:** LoRA parameters and the additional 23,070 head parameters are reported separately. PARA is pruned to at most 663,552 LoRA parameters. LAARA uses the paper-default rank rule without a budget cap.

Full settings and artifact paths are in [config.yaml](config.yaml) and [the M7 selection lock](results/m7_selection.json). The [LAARA budget-cap proposal](docs/m7_laara_budget_cap_proposal.md) has not been implemented.

## Setup and commands

Use Python **3.11+**, a CUDA-capable PyTorch installation for training, and the local datasets described in [data/README.md](data/README.md). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the plotting dependencies with `python -m pip install -r requirements-analysis.txt` when regenerating figures. These commands print the existing plan, regenerate the M7 table from saved evaluation JSON and any local run logs, and regenerate shareable figures. Training and model evaluation require the local data and adapters; the runner reuses completed outputs. Training-time notes in a regenerated aggregate require the Git-ignored local logs; the published [M7 table](results/m7_table.md) preserves the observed notes.

```bash
python -m src.experiment --seed 0 --gpu 0 --phase all --dry-run
python -m src.aggregate
python scripts/make_figures.py --only fig2 fig3 fig11
```

For local checks, install `requirements-dev.txt` and run `pytest -q`, `ruff check .`, `ruff format --check .`, and `mypy src`. The repository includes [pytest path configuration](pyproject.toml) so tests run from the project root.

For GPU commands, seed reuse paths, and the M8 data verification sequence, see [the M8 runbook](docs/m8_runbook.md) and [M7 runner](src/experiment.py). The M8 CCA code has **not** yet produced verified scores; no CCA or rank-correlation result is claimed here.

The [research presentation](deliverables/HuBERT_LoRA_Rank_Study_2026-11-12.pptx) summarizes the study and its observed M7 results.

## Repository layout

| Path | Purpose |
| --- | --- |
| `config.yaml` | Experiment, method, data, and M8 analysis settings |
| `src/` | Data/model stages, training, evaluation, methods, aggregation, rank utilities |
| `scripts/` | GPU runner helpers, plotting, and M8 execution scripts |
| `tests/` | Focused budget and M8 contract tests |
| `results/eval/` | Shareable evaluation JSON; no model weights |
| `results/ranks/` | Small final rank snapshots with source-run provenance |
| `results/figures/`, `results/tables/` | Shareable figures and summary tables |
| `results/runs/`, `results/para/` | Local adapters, checkpoints, and large intermediate outputs (Git-ignored) |
| `docs/`, `ref/` | Runbooks, proposal, and research notes |
| `deliverables/` | Presentation slides |

Generated model weights, local data, downloaded external code, training logs, and build caches are excluded by [.gitignore](.gitignore). Keep the observed JSON and figures when publishing so the README's result links resolve.

## References

- [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447)
- [Post-Optimization Adaptive Rank Allocation for LoRA (PARA)](https://arxiv.org/abs/2604.27796)
- [LAARA: Layer-Aware Adaptive Rank Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2607.19391)
- [LibriSpeech corpus](https://www.openslr.org/12)
- [Layerwise analysis code used for the pending M8 CCA study](https://github.com/ankitapasad/layerwise-analysis)
