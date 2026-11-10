# HuBERT CTC LoRA rank experiment

This repository will compare uniform LoRA, PARA, and LAARA on HuBERT-base
fine-tuned for CTC ASR. The experiment configuration is in [`config.yaml`](config.yaml);
dependencies are listed in [`requirements.txt`](requirements.txt).

## Status

M0, M1, and M2 are complete based on user-reported verification. M3 baseline
runs are finished; learning-rate selection is pending M4 evaluation.

## M1 data verification

The loader uses the observed tree `data/librispeech_finetuning/{1h,9h}` and
`data/LibriSpeech/{dev-clean,dev-other,test-clean,test-other}`. Audio and chapter
transcripts are paired by utterance ID from `.flac` and `.trans.txt` files.
Configuration, including the total-audio-seconds batch budget, is in
[`config.yaml`](config.yaml).

Run M1 verification from the repository root:

```bash
python -m src.data --verify
```

The verification checks model/preprocessing imports, reports dataset metadata
and split leakage, prints three length-bucketed training batches, and writes
JSONL manifests plus `data/vocab.json` only when `--write-manifests` is passed.
By default, verification does not overwrite existing outputs. `--debug` limits
batches to 50 train utterances and 20 utterances per evaluation split.

## M2 model inspection

After M1 verification, run the non-training model inspection on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.model --inspect
```

It reports the loaded HuBERT config and layer modules, checks CNN freezing and
PEFT LoRA parameter counts, then runs float32/bfloat16 CTC forwards and debug
batch output-length checks. No optimizer or training loop is invoked.

## M3 baseline training

Run the one-epoch debug check first. It uses 50 training utterances, 30 update
steps, and 20 dev-clean utterances; it checks that loss falls, that label
decoding has WER 0, and that saved/reloaded adapter loss matches:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --method uniform --rank 4 --seed 0 --epochs 1 --lr 3e-4 --out results/runs/debug-r4 --debug
```

The rank-4 learning-rate sweep runs five epochs at each configured learning
rate and records epoch WER curves in `results/sweep.json`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --method uniform --rank 4 --seed 0 --sweep
```

An example standalone rank-16 run (using the configured default learning rate;
this is not a sweep winner selection) is:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train --method uniform --rank 16 --seed 0 --epochs 5 --lr 3e-4 --out results/runs/uniform-r16-seed0
```

## M4 full-split evaluation

Evaluate the last checkpoints of the three rank-4 learning-rate runs on
dev-clean and dev-other. The command refuses test splits unless `--allow-test`
is explicitly passed:

```bash
for lr in 3e-4 1e-3 3e-3; do run="uniform-r4-seed0-lr${lr}"; CUDA_VISIBLE_DEVICES=0 python -m src.evaluate --adapter "results/runs/${run}/last" --splits dev-clean dev-other --out "results/eval/${run}.json" || exit; done
```

Each split is evaluated twice for deterministic repeatability and reports
corpus-level WER, S/D/I counts, and five best/worst utterances. Compare the
saved development results without selecting a winner:

```bash
python -m src.pick_lr
```

## Configuration note

`experiment.seeds` is set to `[0, 1, 2]` for the experiment.

## Planned layout

- `src/`: data, model, training, evaluation, adapter methods, runner, and analysis
- `results/`: run configurations, logs, metrics, and plots
- `checkpoints/`: training checkpoints
- `config.yaml`: experiment settings
- `PROGRESS.md`: module completion checklist
# genAI
