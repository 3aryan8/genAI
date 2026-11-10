[x] M0 Scaffold done - 2026-11-10 - User-reported verification: torch 2.15.0+cu132, CUDA OK, bf16 supported, peft 0.20.0, transformers 5.22.2; 2x RTX 4500 Ada 24GB.
[x] M1 Data done - 2026-11-10 - User-reported verification: train10h 2763 utterances/9.7468 h (1h 286, 9h 2477, no overlap); eval counts dev-clean 2703, dev-other 2864, test-clean 2620, test-other 2939; all 16 kHz mono; vocab 30; no train/eval leakage; text round-trip passed; batch-label/manifest mismatches 0 on all splits; same-seed batch hashes identical. Earlier glued words were a terminal copy artifact (manifest True/False check confirmed).
[x] M2 Model done - 2026-11-10 - User-reported GPU 1 verification: 12 layers, hidden 768/FFN 3072, vocab 30/pad 0, newly initialized lm_head; 72 targets (6/layer); PEFT/formula counts agree (r4 686,622; r16 2,677,278); encoder frozen; fp32/bf16 losses finite and within 0.002%; minimum frame-label gap 97.
[ ] M3 Baseline training - M3 baseline runs finished, selection pending M4. User-reported verification: adapter and lm_head saved (144 LoRA tensors plus lm_head weight/bias); reload loss difference 0; debug fixed-batch loss 8.469 -> 6.255 in 30 steps; 55.06 s/epoch (182 steps, 3.305 steps/s), peak GPU memory ~6.3 GB. The initial rank-4, lr 3e-4, one-epoch debug produced dev-clean WER 1.000 with blank outputs; the later 30-epoch learning-rate runs finished and M4 evaluation/selection is pending.
[ ] M4 Evaluation - M4 code written, verification pending on user machine
[ ] M5 PARA
[ ] M6 LAARA
[ ] M7 Experiment runner
[ ] M8 Analysis
[ ] M9 Reproducibility
