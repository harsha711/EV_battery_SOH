Why is CNN beating LSTM and Transformer?
Context
Reported k-fold val MAE on this dataset (from the summary tables in notebooks/05_lstm_cycle.ipynb and 06_transformer_cycle.ipynb):

Model	val MAE %	val RMSE %
MLP	11.62	16.13
CNN	7.46	12.64
Transformer	10.25	15.23
LSTM	10.88	15.61
The user is surprised: LSTM/Transformer "should" win on a 301-step sequence. Is there a bug, or is this the real story?

Short answer: the numbers are real, not a bug. I read all four model files, the trainer, the dataset/preprocessing, and every relevant cell of the four training notebooks plus the k-fold driver. There is no implementation defect that would invalidate the comparison. But several methodological choices systematically disadvantage LSTM and Transformer relative to CNN, and the framing "sequence models should win on sequences" is itself wrong for this particular task. This plan lays out what is and isn't a bug, ranked by likely impact, plus a focused experiment list to confirm.

What I verified is correct (no bugs here)
Bidirectional LSTM hidden-state extraction — lstm.py:99-103 uses h_n[-2] (last-layer forward) and h_n[-1] (last-layer backward). With num_layers=2, bidirectional=True, h_n has shape (4, B, H) laid out as [L0_fwd, L0_bwd, L1_fwd, L1_bwd]. -2 and -1 are the right indices. Correct.
CNN permute — cnn1d.py:109 permutes (B, 301, 3) → (B, 3, 301) before Conv1d. Correct.
Transformer positional embeddings — transformer.py:84-88,128-130 registers a (1, 301) index buffer and adds pos_embedding(positions) to the projected input. No off-by-one, no broadcasting bug. Correct.
Normalization — stats are computed on the train pool only via BatterySOHDataset.compute_normalization_stats (dataset.py:151-206), then reused for val/test. No leakage.
Battery-level split — dataset.py:294-311 asserts disjoint train/val/test battery sets. No row-level leakage.
Same loss, optimizer, scheduler, seed for all four models in the k-fold driver (00_kfold_baselines.ipynb cell 1, 5). Apples-to-apples within those constraints.
What is genuinely uneven across models (ranked by impact)
1. The label is per-block, not per-step — destroys LSTM/Transformer's main advantage
This is the single biggest insight. From preprocessing.py:174-189:

for block_idx in range(len(ref_indices) - 1):
    soh_label = soh_vals[block_idx + 1]      # one SOH per block
    for j in range(start + 1, end):           # all steps in block
        ...
        y_list.append(soh_label)              # same label for every step
Every random-walk step in a block gets the same SOH label — the SOH measured at the next reference discharge, possibly hundreds of steps later. So one sample is a single 301-point step (~minutes of cycling), but the label reflects degradation accumulated over the whole block.

Consequence: within a single 301-point window there is no temporal signal that maps to the label. The label is determined by which battery and which block the window came from, not by the temporal evolution inside the window. Useful information is the distribution of (V, I, T) over the window — exactly what 20 hand-crafted summary stats (MLP) and global-pooled conv features (CNN) capture. Sequence-order modeling (LSTM/Transformer) is largely wasted capacity here.

This is why MLP at 11.62 is barely worse than LSTM/Transformer at ~10–11: those models are essentially learning the same per-window distributional features but with much harder optimization.

The CNN wins because (a) it captures distributional features via local filters + global avg-pool, AND (b) its inductive bias toward local patterns (knees, plateaus in the V/I traces inside one step) gives it real signal that summary stats miss, without paying the gradient-flow cost of recurrence/attention.

2. Batch size disparity in the k-fold driver
00_kfold_baselines.ipynb cell 1: BATCH_SIZES = {'MLP': 65536, 'CNN': 2048, 'LSTM': 2048, 'Transformer': 512}.

Same LR (1e-3) for all. Smaller batch = noisier gradients per step. Transformer at 512 takes ~4× more steps per epoch than CNN at 2048, with noisier estimates each step, and the same scheduler patience of 5 epochs. This is not catastrophic, but it slightly disadvantages Transformer.

3. Epoch budget
K-fold uses MAX_EPOCHS=40 with PATIENCE=10. The single-split CNN notebook ran 56 epochs to its best (val MAE 4.02) — almost 1.5× the k-fold cap. Transformer/LSTM converge slower than CNN; capping at 40 disadvantages them more than CNN, but probably not by enough to flip the ranking.

4. Hyperparameters not tuned per architecture
All four use LR=1e-3, weight_decay=1e-4, MSELoss, patience=5 for ReduceLROnPlateau, no warmup. This is fine for CNN, defensible for LSTM, but unusual for Transformer — Transformers typically want LR warmup and often benefit from norm_first=True (Pre-LN) when training small models from scratch with no pretraining. Current code uses Post-LN (transformer.py:97).

5. Transformer is small and undertrained
3 layers, d_model=64, 4 heads, ~150K params. Plus learnable positional embeddings starting from random (no inductive bias for "position 5 is near position 6"). With small batches and no warmup, the model spends early epochs learning what sinusoidal PE would give for free.

6. Single-step sequence is short for an LSTM
301 steps with bidirectional 2-layer LSTM is fine in theory, but the per-window signal (point 1) means the LSTM mostly needs aggregation, not memory. It's paying recurrence cost for a job that doesn't need recurrence.

So is the process flawed?
The k-fold methodology and code are sound. The ranking is real for this framing of the problem (predict SOH from one 301-point step). It is not a fair test of "which architecture is best for temporal data" because the labels carry no within-window temporal signal — they're block-level constants. The CNN wins because the task reduces to feature aggregation, which CNNs do efficiently with strong local-pattern bias. LSTM/Transformer are bringing tools that don't match the actual problem.

The cycle-history notebooks (05_lstm_cycle.ipynb, 06_transformer_cycle.ipynb) are the right next step: they reframe the input as (WINDOW=10, N_FEATURES=15) sequences of block-level summary features, where temporal structure across blocks actually carries the label signal. That's where LSTM/Transformer have a fighting chance — and the cycle-history notebooks are explicitly set up to test that.

Recommended diagnostic experiments (in order)
Each is small and isolates one hypothesis. Run only as many as needed to satisfy yourself.

A. Confirm the per-block label hypothesis (cheapest, most informative)
For each (battery, block), compute the variance of CNN-feature-vector predictions vs. the variance of the label. If label variance within a block is ≈ 0 and prediction variance within a block is also small, that confirms the model is learning "which block am I in" rather than within-step dynamics. Expected: high agreement → confirms the framing issue.

Code-wise: take the saved CNN checkpoint, compute predictions for all train samples, group by (battery_id, block_index) from the HDF5 file, and look at per-group std-dev of predictions vs. labels.

B. Run the cycle-history notebooks if not done
05_lstm_cycle.ipynb and 06_transformer_cycle.ipynb already exist and use (WINDOW=10, 15 features) sequences. If LSTM/Transformer beat CNN there (or match it), that is direct evidence that they were just being given the wrong task.

C. Equalize the k-fold compute budget
Edit 00_kfold_baselines.ipynb cell 1: bump MAX_EPOCHS from 40 to ~80, set BATCH_SIZES['Transformer'] = 2048 (matching CNN/LSTM — VRAM is sufficient based on the comment in 04_transformer.ipynb cell 2). Re-run only LSTM and Transformer folds. If CNN still wins by ~3 points, the architecture-vs-task hypothesis stands.

D. Add Transformer training niceties (only if C still shows CNN ≫ Transformer)
Either: (1) set norm_first=True in transformer.py:97 for Pre-LN, OR (2) add a 1000-step linear warmup before ReduceLROnPlateau kicks in. These are small edits to the model/training loop. If neither moves the needle, conclude the task itself doesn't reward attention.

Critical files (read-only references)
project/src/preprocessing.py:174-189 — the per-block label assignment
project/src/models/cnn1d.py — verified correct
project/src/models/lstm.py:99-103 — verified correct
project/src/models/transformer.py:84-130 — verified correct
project/src/dataset.py:151-206 — verified normalization sound
project/notebooks/00_kfold_baselines.ipynb — k-fold driver, batch sizes & epoch budget live here
project/notebooks/05_lstm_cycle.ipynb, 06_transformer_cycle.ipynb — cycle-history reframing
Verification
For each experiment above, the verification is to re-run the affected notebook end-to-end and compare the printed Test MAE / Val MAE to the table at the top of this plan. No new tests to write.

Stopping rule: if experiment A confirms per-block-label dominance, stop — the methodology section of the report should explain that the single-step framing rewards aggregation models (CNN, MLP) over sequence models, and the cycle-history experiments (B) are the appropriate sequence-model evaluation.