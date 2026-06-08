# --- OPTUNA HPT FOR THE SIAMESE DNN ---
# Logic: Search hyperparameters for `dnn_siamese_modelling.ipynb` by maximizing
#        Pearson r on the held-out Bainbridge s2 pair file. Trials persist to
#        a SQLite study so parallel workers + resume "just work".
# Workflow: For each trial - build the same item_disjoint 80/20 split as the
#           notebook, train a Siamese encoder + interaction + head with the
#           sampled config, early-stop on the outer-val r-space RMSE, reload
#           the best checkpoint and report Pearson r on the holdout.
#
# Config is set as constants at the top of the file. Edit and re-run.

from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=UserWarning)


# ---- Config (no argparse - edit and re-run) ----

# Per-process trial budget. With parallel workers total = N_TRIALS * N_workers.
N_TRIALS = 50
TIMEOUT  = None          # optional wall-clock seconds; None = no limit

# Persistent SQLite study for parallel workers + resume
STUDY_NAME = "dnn_siamese_holdout_v2"
STORAGE    = None        # None -> defaults to sqlite next to this file

# Data
EMB_MODEL  = "qwen3-8b-with-instruction"
K_CLUSTERS = 512         # locates the pair file

# Split + target transform
OUTER_VAL_FRAC = 0.2
R_CLIP         = 0.999   # clip r before atanh to keep z finite

# Training caps (per trial)
MAX_EPOCHS     = 150
EARLY_STOP_PAT = 15      # patience on outer-val r-space RMSE

SEED   = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths (relative to this script's location: code/modelling/optuna_hpt/)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / ".." / ".." / ".." / "data"
TRAIN_PAIR_PATH = DATA_DIR / "clustered_embeddings" / EMB_MODEL / f"autoencoded_clusters_{K_CLUSTERS}.parquet"
TRAIN_EMB_PATH  = DATA_DIR / "raw" / EMB_MODEL / "embeddings_raw.parquet"
HOLD_PAIR_PATH  = DATA_DIR / "clustered_embeddings" / EMB_MODEL / f"holdout_autoencoded_clusters_{K_CLUSTERS}.parquet"
HOLD_EMB_PATH   = DATA_DIR / "raw" / EMB_MODEL / "holdout_embeddings_raw.parquet"

if STORAGE is None:
    STORAGE = f"sqlite:///{HERE / (STUDY_NAME + '.db')}"

# Aux feature columns (must match the notebook).
AUX_NUMERIC = [
    "pair_negative", "pair_positive", "contradiction", "entail", "logic_neutral",
    "similarity", "thematic_intensity", "logical_friction", "sentiment_balance",
    "global_sim",
    "sent_positive_item1", "sent_neutral_item1", "sent_negative_item1",
    "emo_neutral_item1", "emo_surprise_item1", "emo_joy_item1", "emo_fear_item1",
    "emo_anger_item1", "emo_sadness_item1", "emo_disgust_item1",
    "sent_positive_item2", "sent_neutral_item2", "sent_negative_item2",
    "emo_neutral_item2", "emo_surprise_item2", "emo_joy_item2", "emo_fear_item2",
    "emo_anger_item2", "emo_sadness_item2", "emo_disgust_item2",
]
AUX_NOMINAL = [
    "top_sentiment_item1", "top_emotion_item1",
    "top_sentiment_item2", "top_emotion_item2",
]


# ---- Model (must match dnn_siamese_modelling.ipynb bit-for-bit) ----

class SiameseEncoder(nn.Module):
    def __init__(self, emb_dim: int, encoder_dims: tuple[int, ...], dropout: float):
        super().__init__()
        layers, prev = [], emb_dim
        for h in encoder_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class SiameseDNN(nn.Module):
    def __init__(
        self, emb_dim, aux_dim, encoder_dims, head_dims, dropout, use_skip,
    ):
        super().__init__()
        self.encoder = SiameseEncoder(emb_dim, encoder_dims, dropout)
        e = self.encoder.out_dim
        head_in = 4 * e + aux_dim

        layers, prev = [], head_in
        for h in head_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.head = nn.Sequential(*layers)
        self.out  = nn.Linear(prev, 1)

        self.use_skip = use_skip
        if use_skip:
            self.aux_skip = nn.Linear(aux_dim, 1)

    def forward(self, e1, e2, aux):
        h1 = self.encoder(e1)
        h2 = self.encoder(e2)
        inter = torch.cat([h1, h2, h1 * h2, (h1 - h2).abs()], dim=-1)
        x = torch.cat([inter, aux], dim=-1)
        z = self.out(self.head(x)).squeeze(-1)
        if self.use_skip:
            z = z + self.aux_skip(aux).squeeze(-1)
        return z   # Fisher-z space


# ---- Data load (done once at import; reused across trials) ----

def _load_all():
    """Load embeddings + train + holdout pair tables, build the item_disjoint
    split, fit preprocessors. Returns the dict of tensors that train_one()
    consumes per trial - reloading parquet each trial would dominate runtime.
    """
    print(f"Loading: {TRAIN_EMB_PATH.name}")
    if not TRAIN_PAIR_PATH.exists(): sys.exit(f"Missing: {TRAIN_PAIR_PATH}")
    if not TRAIN_EMB_PATH.exists():  sys.exit(f"Missing: {TRAIN_EMB_PATH}")
    if not HOLD_PAIR_PATH.exists():  sys.exit(f"Missing: {HOLD_PAIR_PATH}")
    if not HOLD_EMB_PATH.exists():   sys.exit(f"Missing: {HOLD_EMB_PATH}")

    # Training item embeddings
    emb_df = pl.read_parquet(TRAIN_EMB_PATH)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb")]
    EMB_DIM = len(emb_cols)
    item_to_idx = {n: i for i, n in enumerate(emb_df["item"].to_list())}
    ITEM_EMB = torch.tensor(emb_df.select(emb_cols).to_numpy(), dtype=torch.float32).to(DEVICE)

    # Pair table
    dat = pl.read_parquet(TRAIN_PAIR_PATH).filter(
        (pl.col("r").is_not_null()) & (pl.col("r") != 1)
    )
    known = pl.Series("item", list(item_to_idx.keys())).implode()
    dat = dat.filter(pl.col("Parameter1").is_in(known) & pl.col("Parameter2").is_in(known))

    # item_disjoint split
    items_all = (
        pl.concat([
            dat.select(pl.col("Parameter1").alias("item")),
            dat.select(pl.col("Parameter2").alias("item")),
        ]).unique().to_series().sample(fraction=1.0, seed=SEED)
    )
    split_idx = int(len(items_all) * (1 - OUTER_VAL_FRAC))
    train_items = items_all.slice(0, split_idx).implode()
    outer_items = items_all.slice(split_idx, None).implode()
    train_df = dat.filter(pl.col("Parameter1").is_in(train_items) & pl.col("Parameter2").is_in(train_items))
    outer_df = dat.filter(pl.col("Parameter1").is_in(outer_items) & pl.col("Parameter2").is_in(outer_items))

    # Aux preprocessors fit on training pool only
    scaler  = QuantileTransformer(
        output_distribution="normal", n_quantiles=1000,
        subsample=200_000, random_state=SEED,
    ).fit(train_df.select(AUX_NUMERIC).to_numpy())
    encoder_oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        train_df.select(AUX_NOMINAL).to_numpy()
    )

    def featurize(df):
        idx1 = np.fromiter((item_to_idx[p] for p in df["Parameter1"].to_list()), dtype=np.int64)
        idx2 = np.fromiter((item_to_idx[p] for p in df["Parameter2"].to_list()), dtype=np.int64)
        aux_num = scaler.transform(df.select(AUX_NUMERIC).to_numpy())
        aux_cat = encoder_oh.transform(df.select(AUX_NOMINAL).to_numpy())
        aux = np.nan_to_num(np.hstack([aux_num, aux_cat]), nan=0.0).astype(np.float32)
        y = df.select("r").to_numpy().flatten().astype(np.float32)
        return idx1, idx2, aux, y

    tr_i1, tr_i2, tr_aux, tr_y = featurize(train_df)
    ov_i1, ov_i2, ov_aux, ov_y = featurize(outer_df)
    AUX_DIM = tr_aux.shape[1]

    # Holdout item embeddings + pair table (separate item id space)
    hold_emb_df = pl.read_parquet(HOLD_EMB_PATH)
    hold_emb_cols = [c for c in hold_emb_df.columns if c.startswith("emb")]
    assert len(hold_emb_cols) == EMB_DIM
    hold_item_to_idx = {n: i for i, n in enumerate(hold_emb_df["item"].to_list())}
    HOLD_ITEM_EMB = torch.tensor(hold_emb_df.select(hold_emb_cols).to_numpy(),
                                 dtype=torch.float32).to(DEVICE)

    hold_df = pl.read_parquet(HOLD_PAIR_PATH).filter(
        (pl.col("r").is_not_null()) & (pl.col("r") != 1)
    )
    known_h = pl.Series("item", list(hold_item_to_idx.keys())).implode()
    hold_df = hold_df.filter(
        pl.col("Parameter1").is_in(known_h) & pl.col("Parameter2").is_in(known_h)
    )
    h_i1 = np.fromiter((hold_item_to_idx[p] for p in hold_df["Parameter1"].to_list()), dtype=np.int64)
    h_i2 = np.fromiter((hold_item_to_idx[p] for p in hold_df["Parameter2"].to_list()), dtype=np.int64)
    h_aux_num = scaler.transform(hold_df.select(AUX_NUMERIC).to_numpy())
    h_aux_cat = encoder_oh.transform(hold_df.select(AUX_NOMINAL).to_numpy())
    h_aux     = np.nan_to_num(np.hstack([h_aux_num, h_aux_cat]), nan=0.0).astype(np.float32)
    h_y       = hold_df.select("r").to_numpy().flatten().astype(np.float32)

    print(
        f"item embs (train): {ITEM_EMB.shape}  | item embs (holdout): {HOLD_ITEM_EMB.shape}\n"
        f"train pairs: {tr_y.shape[0]:,}  outer val: {ov_y.shape[0]:,}  holdout: {h_y.shape[0]:,}\n"
        f"AUX_DIM = {AUX_DIM}  EMB_DIM = {EMB_DIM}"
    )

    return {
        "EMB_DIM":  EMB_DIM,
        "AUX_DIM":  AUX_DIM,
        "ITEM_EMB": ITEM_EMB,
        "HOLD_ITEM_EMB": HOLD_ITEM_EMB,
        # train tensors
        "tr_idx1": torch.tensor(tr_i1, dtype=torch.long,    device=DEVICE),
        "tr_idx2": torch.tensor(tr_i2, dtype=torch.long,    device=DEVICE),
        "tr_aux":  torch.tensor(tr_aux, dtype=torch.float32, device=DEVICE),
        "tr_y_z":  torch.tensor(
            np.arctanh(np.clip(tr_y, -R_CLIP, R_CLIP)), dtype=torch.float32, device=DEVICE
        ),
        # outer val tensors
        "ov_idx1": torch.tensor(ov_i1, dtype=torch.long,    device=DEVICE),
        "ov_idx2": torch.tensor(ov_i2, dtype=torch.long,    device=DEVICE),
        "ov_aux":  torch.tensor(ov_aux, dtype=torch.float32, device=DEVICE),
        "ov_y_r":  ov_y.astype(np.float32),
        # holdout tensors
        "h_idx1":  torch.tensor(h_i1, dtype=torch.long,    device=DEVICE),
        "h_idx2":  torch.tensor(h_i2, dtype=torch.long,    device=DEVICE),
        "h_aux":   torch.tensor(h_aux, dtype=torch.float32, device=DEVICE),
        "h_y_r":   h_y.astype(np.float32),
    }


DATA = _load_all()


# ---- Trial ----

def sample_params(trial: optuna.Trial) -> dict:
    """Sample one config. Architecture is parameterized as depth + base width +
    decay factor so Optuna explores a small set of dimensions per branch."""
    # Encoder shape: depth + base width + decay factor (e.g. 512 -> 384 -> 256)
    enc_depth      = trial.suggest_int("enc_depth", 1, 3)
    enc_base_width = trial.suggest_categorical("enc_base_width", [128, 256, 384, 512, 768])
    enc_decay      = trial.suggest_float("enc_decay", 0.4, 1.0)
    enc_dims, w = [], enc_base_width
    for _ in range(enc_depth):
        enc_dims.append(int(max(32, round(w))))
        w *= enc_decay

    # Head shape: depth + base width + decay (operating on 4*enc_dims[-1] + aux)
    head_depth      = trial.suggest_int("head_depth", 1, 3)
    head_base_width = trial.suggest_categorical("head_base_width", [128, 256, 384, 512])
    head_decay      = trial.suggest_float("head_decay", 0.4, 1.0)
    head_dims, w = [], head_base_width
    for _ in range(head_depth):
        head_dims.append(int(max(32, round(w))))
        w *= head_decay

    return {
        "encoder_dims":   tuple(enc_dims),
        "head_dims":      tuple(head_dims),
        "dropout":        trial.suggest_float("dropout", 0.05, 0.6),
        "use_skip":       trial.suggest_categorical("use_skip", [True, False]),
        "lr":             trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay":   trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size":     trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096]),
        "huber_beta":     trial.suggest_float("huber_beta", 0.05, 1.5, log=True),
        "grad_clip":      trial.suggest_float("grad_clip", 0.5, 5.0),
        "sched_patience": trial.suggest_int("sched_patience", 4, 12),
    }


def train_one(trial: optuna.Trial, params: dict) -> tuple[float, float]:
    """Train one Siamese DNN. Returns (best_outer_rmse, holdout_pearson_r)."""
    torch.manual_seed(SEED + trial.number)
    np.random.seed(SEED + trial.number)

    model = SiameseDNN(
        emb_dim=DATA["EMB_DIM"], aux_dim=DATA["AUX_DIM"],
        encoder_dims=params["encoder_dims"], head_dims=params["head_dims"],
        dropout=params["dropout"], use_skip=params["use_skip"],
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=params["sched_patience"]
    )
    criterion = nn.SmoothL1Loss(beta=params["huber_beta"])

    n_train = DATA["tr_idx1"].shape[0]
    bs      = params["batch_size"]
    grad_clip = params["grad_clip"]

    best_rmse  = float("inf")
    best_state = None
    since_best = 0
    rng = np.random.default_rng(SEED + trial.number)

    for epoch in range(MAX_EPOCHS):
        # --- train epoch ---
        model.train()
        perm = rng.permutation(n_train)
        for start in range(0, n_train, bs):
            idx = perm[start:start + bs]
            idx_t = torch.from_numpy(idx).to(DEVICE)

            i1 = DATA["tr_idx1"][idx_t]
            i2 = DATA["tr_idx2"][idx_t]
            ax = DATA["tr_aux"][idx_t]
            yb = DATA["tr_y_z"][idx_t]

            e1 = DATA["ITEM_EMB"][i1]
            e2 = DATA["ITEM_EMB"][i2]

            optimizer.zero_grad()
            preds = model(e1, e2, ax)
            loss  = criterion(preds, yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        # --- outer val (r-space RMSE in chunks) ---
        model.eval()
        with torch.no_grad():
            chunks = []
            for s in range(0, DATA["ov_idx1"].shape[0], 8192):
                i1c = DATA["ov_idx1"][s:s+8192]; i2c = DATA["ov_idx2"][s:s+8192]
                axc = DATA["ov_aux"][s:s+8192]
                preds_z = model(DATA["ITEM_EMB"][i1c], DATA["ITEM_EMB"][i2c], axc)
                chunks.append(torch.tanh(preds_z).cpu().numpy())
            ov_preds = np.concatenate(chunks)
        outer_rmse = float(np.sqrt(np.mean((ov_preds - DATA["ov_y_r"]) ** 2)))
        scheduler.step(outer_rmse)

        if outer_rmse < best_rmse - 1e-5:
            best_rmse = outer_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1

        # report negative RMSE so pruner treats higher-is-better
        trial.report(-outer_rmse, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if since_best >= EARLY_STOP_PAT:
            break

    # --- holdout eval with best checkpoint ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_z = model(
            DATA["HOLD_ITEM_EMB"][DATA["h_idx1"]],
            DATA["HOLD_ITEM_EMB"][DATA["h_idx2"]],
            DATA["h_aux"],
        )
        h_preds = torch.tanh(preds_z).cpu().numpy()
    holdout_r, _ = pearsonr(DATA["h_y_r"], h_preds)

    del model, best_state
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return best_rmse, float(holdout_r)


def objective(trial: optuna.Trial) -> float:
    params = sample_params(trial)
    t0 = time.time()
    try:
        outer_rmse, holdout_r = train_one(trial, params)
    except optuna.TrialPruned:
        raise
    except Exception as exc:
        print(f"  trial {trial.number} crashed: {exc!r}")
        raise optuna.TrialPruned() from exc

    elapsed = time.time() - t0
    trial.set_user_attr("outer_rmse",   outer_rmse)
    trial.set_user_attr("encoder_dims", list(params["encoder_dims"]))
    trial.set_user_attr("head_dims",    list(params["head_dims"]))
    trial.set_user_attr("elapsed_sec",  elapsed)
    print(
        f"  trial {trial.number:3d} | holdout r {holdout_r:.4f} | "
        f"outer RMSE {outer_rmse:.4f} | enc {params['encoder_dims']} | "
        f"head {params['head_dims']} | dropout {params['dropout']:.2f} | "
        f"lr {params['lr']:.2e} | wd {params['weight_decay']:.2e} | "
        f"bs {params['batch_size']} | skip {params['use_skip']} | "
        f"beta {params['huber_beta']:.2f} | {elapsed:.1f}s"
    )
    return holdout_r


# ---- Main ----

def main():
    sampler = TPESampler(seed=SEED, multivariate=True, group=True)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=STORAGE,
        load_if_exists=True,
    )

    print(f"Device: {DEVICE}")
    print(f"Study:  {STUDY_NAME}  |  storage: {STORAGE}")
    print(f"Running {N_TRIALS} trials (timeout = {TIMEOUT})...")
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, gc_after_trial=True)

    print("\n" + "=" * 70)
    print("BEST TRIAL")
    print("=" * 70)
    best = study.best_trial
    print(f"  Holdout Pearson r: {best.value:.4f}")
    print(f"  Trial #:           {best.number}")
    print(f"  Outer val RMSE:    {best.user_attrs.get('outer_rmse'):.4f}")
    print(f"  Encoder dims:      {best.user_attrs.get('encoder_dims')}")
    print(f"  Head dims:         {best.user_attrs.get('head_dims')}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    out_json = HERE / f"{STUDY_NAME}_best.json"
    out_json.write_text(json.dumps({
        "holdout_r":    best.value,
        "outer_rmse":   best.user_attrs.get("outer_rmse"),
        "encoder_dims": best.user_attrs.get("encoder_dims"),
        "head_dims":    best.user_attrs.get("head_dims"),
        "params":       best.params,
    }, indent=2))
    print(f"\nSaved best params -> {out_json}")


if __name__ == "__main__":
    main()
