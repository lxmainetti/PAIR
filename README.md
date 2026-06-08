# ItemStats

A local, GPU-accelerated pipeline that predicts the pairwise correlation between any two psychometric items directly from their wording. It is built on Qwen-3 text embeddings and a Siamese DNN trained end-to-end on aggregated open-access survey data.

The headline result: given just the *text* of two scale items, the model recovers their empirical inter-item correlation with **Pearson r ≈ 0.85** on out-of-training items from familiar scale families, and **r ≈ 0.74** on entirely unseen scales it never encountered during training (validation data from Arslan and Hommel, 2025).

> **Architecture note (current direction):** earlier versions of the pipeline fed the model a large auxiliary feature block — transformer cross-encoder scores (NLI, reranker similarity, pair sentiment) and per-item sentiment/emotion classifications. Permutation-importance analysis showed these added essentially nothing on top of the raw embeddings: permuting the embedding pathway drops Pearson r by ≈ 0.85, while the entire cross-encoder + sentiment block contributes only ≈ 0.005, almost all of which is recovered by a single feature (`global_sim`). The pipeline is therefore being streamlined to **embeddings + `global_sim` only**, and the cross-encoder and per-item sentiment stages are being retired. See [Dropped features](#-dropped-features) below.


## 📊 Data sources

This project is built on the principles of Open Science. The training pool aggregates **~32 published psychometric scales** drawn from open-access repositories; the full scale-by-scale list with abbreviations and source URLs lives in [`data/scales/sources.md`](data/scales/sources.md). By provenance:

- **`psychTools` R package** (5 scales): Eysenck Personality Inventory (EPI), Big Five Inventory (BFI), Motivational State Questionnaire (MSQ), SAPA Personality Inventory (SPI), and the Athenstaedt Gender-Role Self-Concept scale.
- **OpenPsychometrics.org** (18 scales): HSQ, Taylor Manifest Anxiety, HEXACO, RIASEC, Consideration of Future Consequences, DASS, ECR, Empathizing/Systemizing Quotient, GCBS, KIMS, MACH-IV, MGKT, NPAS, Rosenberg Self-Esteem, RWAS, Sexual Compulsivity, AMBI, and the 16PF.
- **OSF & open research repositories** (8 scales): PID-5, SCL-90-R, Psychological Strain Scales, the Comprehensive Autistic Inventory / ASRS (plus an Adult ADHD self-report), C-PETS, EPTEPS, the Vanity scale, and a Self-Efficacy Fragility scale from an own study deposited to OSF.
- **Journal supplement (DOI)** (1 scale): General Attitudes towards AI Scale (GAAIS).

The two out-of-training splits come from separate studies kept fully disjoint from the training scales: the **Bainbridge** personality mega-study (holdout — item-disjoint, overlapping scale families) and a **SurveyBot** study of entirely new scales (validation).



## 🔄 Pipeline

The pipeline is modular — each stage reads from disk and writes to disk, so any single stage can be re-run in isolation. All stages support three splits via a `split` switch: `""` (training), `"holdout_"` (cross-scale held-out items), and `"validation_"` (entirely separate scales).

1. **Data integration** (`item_integration/data_integration.R`) — Parses the heterogeneous open-access surveys (CSV/TSV/SAV/RDS + codebooks), computes per-scale inter-item correlations, and normalizes everything into a unified item-level correlation table plus a per-item descriptive-stats list. Emits the training / holdout / validation parquet files under `data/raw/`.
2. **Embedding generation** (`feature_engineering/embed_items.R`, backends in `r_functions/get_embeddings.R`) — Generates a semantic embedding for every unique item via a local Ollama model (Qwen-3-Embedding 8B with a task-specific instruction prefix is the best performer) or, optionally, the OpenAI / Google embedding APIs. Writes an item-keyed 4096-dim embedding table per split.
3. **Autoencoder compression & pair table** (`feature_engineering/autoencoder_feature_prep.ipynb`) — Trains a small undercomplete autoencoder that compresses the 4096-dim embedding to a 512-dim bottleneck, derives the `global_sim` feature (cosine similarity between the two items in the compressed space), and assembles the per-pair feature/target table. Saves the autoencoder weights to `models/psychometric_ae_weights.pt`. These autoencoded dimensional products are deprecated and the global similarity will soon be implented to resemble the similarity of all (not-encoded) 4096 embedding dimensions of Qwen3-8b-embedding.
4. **Modelling**
   - **Siamese DNN — headline model** (`modelling/model_training.ipynb`) — A shared encoder applied to each item embedding, a symmetric 4-way interaction (`concat[h1, h2, h1·h2, |h1−h2|]`), the `global_sim` scalar, and a head MLP. Trained in Fisher-z space with a Huber loss, AdamW, and early stopping on outer-validation RMSE. Reads the **raw 4096-dim embeddings directly** so the encoder can learn task-relevant geometry rather than being constrained by the autoencoder's reconstruction loss. Best checkpoint saved to `models/dnn_siamese_cor.pt`.
   - **Evaluation** (`modelling/model_validation.ipynb`) — Standalone scoring on the holdout and validation splits: rebuilds the item-disjoint split and train-fitted preprocessors, recomputes `global_sim` through the trained autoencoder for new items, and reports r / R² / RMSE / MAE plus calibration plots.


## 🧪 Dropped features

Two feature-engineering stages were built, benchmarked, and then removed because they did not improve the Siamese model:

- **Cross-encoder features** (`feature_engineering/crossencoder.ipynb`) — per-pair NLI (entail / contradict / neutral), reranker similarity, pair sentiment, and derived interactions (`thematic_intensity`, `logical_friction`, `sentiment_balance`).
- **Per-item sentiment/emotion** (`feature_engineering/semantic_analysis_items.ipynb`) — 3-class sentiment and 7-class emotion scores for every item.

Permutation importance on the trained model showed the raw embeddings dominate (≈ 0.85 drop in Pearson r when permuted), while the whole cross-encoder + sentiment block accounts for ≈ 0.005 — and `global_sim` alone recovers most of that. Removing both stages leaves correlation essentially unchanged and raises RMSE only marginally, at a large saving in compute (the cross-encoder runs a transformer over *every pair*). The notebooks are retained for reference but are no longer part of the active pipeline.

> **Migration note:** completing the removal means repointing `embed_items.R` and `autoencoder_feature_prep.ipynb` at the raw `item_correlations` / `item_list` tables instead of the current `*_with_cross` / `*_sentiment` variants, and trimming the cross-encoder + sentiment entries (and the renamed modelling notebooks) from the `tasks` list in `coordinator.ipynb`.


## 💻 Hardware & performance

Designed for local, GPU-enabled processing for data privacy and iteration speed:

- **Device:** optimized for NVIDIA CUDA GPUs (developed on an RTX 2080 Ti); falls back to CPU.
- **Efficiency:** float32 throughout, with the embedding table held as a frozen on-device lookup (~51 MB for 3,127 items × 4,096 dim) so each training batch only moves `(idx1, idx2, global_sim, target)` rows rather than the full embeddings.
- **Hyperparameter tuning:** standalone Optuna script in `code/modelling/optuna_hpt/dnn_siamese_holdout_tuning.py` — multi-worker SQLite study (`dnn_siamese_holdout_v*.db`), TPESampler + MedianPruner, optimizing Pearson r on the held-out Bainbridge pair file. The current best configuration (`dnn_siamese_holdout_v2_best.json`): encoder `(512)`, head `(256)`, dropout 0.163, lr 3.25e-4, weight decay 2.6e-4, batch 2048, Huber β 0.063, no aux skip-connection. 

> **Note on hyperparameter tuning:** The hyperparameter tuning was conducted with auxilliary features that were abandoned since. Therefore another optuna tuning run is warranted.


## 📈 Current performance

Siamese DNN (embeddings + `global_sim`), single trained checkpoint (`models/dnn_siamese_cor.pt`):

| Split | Description | N pairs | Pearson r | R² | RMSE | MAE |
|---|---|---|---|---|---|---|
| Outer val | Item-disjoint slice of the training data | 19,313 | **≈ 0.85** | 0.70 | 0.097 | 0.073 |
| Holdout | Bainbridge — item-disjoint, same scale families | 87,153 | **≈ 0.85** | 0.65 | 0.096 | 0.073 |
| Validation | SurveyBot3000 validation data — entirely new scales | 34,191 | **≈ 0.74** | 0.54 | 0.139 | 0.108 |

On the validation split predictions stay strongly correlated with the human values (r ≈ 0.74) but are compressed toward zero — the best-fit slope is ≈ 0.55, a regression-to-the-mean effect that under-shoots extreme correlations and shows up as the slightly higher RMSE.


## 🗺️ Repository layout

```
code/
├── coordinator.ipynb                  # runs every stage end-to-end via papermill
├── item_integration/
│   └── data_integration.R             # ~32 scales -> unified train/holdout/validation parquet
├── feature_engineering/
│   ├── embed_items.R                  # local Ollama or hosted-API embeddings
│   ├── r_functions/get_embeddings.R   # Ollama / OpenAI / Gemini embedding backends
│   ├── autoencoder_feature_prep.ipynb # 4096 -> 512 bottleneck, global_sim + pair table
│   ├── crossencoder.ipynb             # (retired) per-pair NLI / similarity / sentiment
│   └── semantic_analysis_items.ipynb  # (retired) per-item sentiment + 7-class emotion
└── modelling/
    ├── model_training.ipynb           # headline model: Siamese DNN, end-to-end
    ├── model_validation.ipynb         # standalone holdout + validation evaluation
    ├── mean_sd_modelling.ipynb        # exploratory: item-mean prediction
    └── optuna_hpt/                    # standalone Optuna hyperparameter search

data/
├── scales/        raw per-scale data + codebooks; sources.md lists every scale
├── raw/           integrated correlation tables + per-split embedding parquets
├── processed/     intermediate joined tables
└── clustered_embeddings/   autoencoder bottleneck + per-pair feature tables

models/            psychometric_ae_weights.pt, dnn_siamese_cor.pt
```

Earlier XGBoost and residual-MLP baselines on the autoencoded features reached comparable r (≈ 0.85); they were abandoned because they performed significantly worse on unseen data.


## 🛣️ Further plans & possible ideas

- Item-level descriptive-stats model (e.g. IRT-difficulty).
- Inter-Scale correlation prediction.
- Classification model for jingle/jangle detection.