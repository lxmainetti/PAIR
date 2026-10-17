"""Siamese DNN training — raw embeddings, end-to-end.

Script version of model_training.ipynb. Predicts pairwise item correlations
from raw item embeddings plus a single cosine-similarity aux feature
(`global_sim`). Trained end-to-end so the shared encoder can learn
task-relevant structure directly, rather than through a separate
reconstruction-loss-constrained compression step.

Architecture lives in siamese_model.py (shared with model_validation.py) so
this file and the eval script can never silently drift apart on model shape.

All config below defaults to whatever model_training.ipynb currently has
hardcoded; override via CLI flags to train a different backbone / config
without touching the file. Paths are resolved relative to this script's own
location, not the caller's working directory, so it can be invoked from
anywhere:

    python model_training.py --emb-model Qwen-Qwen3-Embedding-8B
    python model_training.py --emb-model intfloat-e5-mistral-7b-instruct --max-epochs 50

The notebook (model_training.ipynb) is untouched and still works standalone;
this script is a drop-in equivalent for batch/HPC use.
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: always save figures, never try to pop a window
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer

SCRIPT_DIR = Path(__file__).resolve().parent          # code/modelling
REPO_ROOT = SCRIPT_DIR.parent.parent                   # project root


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the PAIR Siamese DNN on raw item embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Paths / backbone ----
    p.add_argument("--emb-model", default="Qwen-Qwen3-Embedding-8B",
                    help="Embedding backbone name (matches the folder under data/raw/ and models/).")
    p.add_argument("--data-root", default=str(REPO_ROOT / "data" / "raw"),
                    help="Root of the raw parquet data (item_correlations.parquet, embeddings, etc.).")
    p.add_argument("--models-root", default=str(REPO_ROOT / "models"),
                    help="Root under which the checkpoint + scaler are written, in a per-backbone subfolder.")

    # ---- Split ----
    p.add_argument("--outer-val-frac", type=float, default=0.1,
                    help="Fraction of items (not pairs) held out as an item-disjoint internal validation set.")
    p.add_argument("--r-clip", type=float, default=0.999,
                    help="Clip |r| to this value before the Fisher-z (arctanh) transform, to keep z finite.")

    # ---- Architecture (see siamese_model.py) ----
    p.add_argument("--encoder-dims", default="384",
                    help="Comma-separated encoder layer widths, e.g. '384' or '512,256'.")
    p.add_argument("--head-dims", default="256,130",
                    help="Comma-separated head layer widths, e.g. '256,130'.")
    p.add_argument("--dropout", type=float, default=0.226)
    p.add_argument("--use-skip", action="store_true", default=False,
                    help="Add a linear aux->output skip connection.")

    # ---- Optimization (Optuna v3 config) ----
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--weight-decay", type=float, default=0.0023)
    p.add_argument("--huber-beta", type=float, default=0.127)
    p.add_argument("--grad-clip", type=float, default=1.863,
                    help="Max gradient norm; set <= 0 to disable clipping.")
    p.add_argument("--sched-patience", type=int, default=4,
                    help="ReduceLROnPlateau patience (epochs).")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20,
                    help="Early-stopping patience (epochs without outer-val RMSE improvement).")

    p.add_argument("--seed", type=int, default=67,
                    help="Master RNG seed. Seeds Python, NumPy and torch (CPU + CUDA), the "
                         "item-disjoint split and the aux quantile transformer.")
    p.add_argument("--nondeterministic", action="store_true", default=False,
                    help="Allow cuDNN to pick nondeterministic kernels. Faster on some GPUs, "
                         "but run-to-run results will no longer be bit-identical.")
    p.add_argument("--no-plots", action="store_true", default=True,
                    help="Skip generating/saving diagnostic + holdout plots.")
    p.add_argument("--quiet-epochs", action="store_true", default=False,
                    help="Only print at early-stopping / final summary, not every 5 epochs.")
    p.add_argument("--verbose", action="store_true", default=False,
                    help="Print setup detail (tensor shapes, module tree, file paths). Off by "
                         "default: the run prints a one-line config header, the epoch trace, "
                         "and a metrics table.")

    return p


def metrics_row(label: str, n: int, r: float, r2: float, rmse: float, mae: float) -> str:
    """One fixed-width row of the end-of-run metrics table (see metrics_header)."""
    return f"{label:<12s}{n:>10,}  {r:>7.4f}  {r2:>7.4f}  {rmse:>7.4f}  {mae:>7.4f}"


def metrics_header() -> str:
    return (f"{'':<12s}{'n pairs':>10}  {'r':>7}  {'R2':>7}  {'RMSE':>7}  {'MAE':>7}\n"
            + "-" * 56)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the training run touches, and pin cuDNN to deterministic kernels.

    torch.manual_seed() alone is not enough for reproducible runs: it leaves the
    stdlib `random` module unseeded, and it does not stop cuDNN from autotuning its
    way to a different (equally valid, differently rounded) convolution/GEMM kernel
    on a later run. Seeding all four RNGs and fixing the cuDNN backend makes a run
    reproducible on the same machine and GPU.

    Note that this pins the *stochastic* parts of training. It does not make results
    comparable across different GPUs, CUDA versions, or torch builds, where kernel
    selection and floating-point reduction order differ regardless of seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # no-op when CUDA is unavailable
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = deterministic
    # benchmark=True lets cuDNN re-autotune per input shape, which is a second,
    # seed-independent source of run-to-run drift -- off whenever we want determinism.
    torch.backends.cudnn.benchmark = not deterministic


def parse_dims(s: str) -> tuple:
    s = s.strip()
    if not s:
        return ()
    return tuple(int(x) for x in s.split(","))


# --------------------------------------------------------------------------- #
# Shared helpers (mirrors add_global_sim in model_training.ipynb / model_validation.ipynb)
# --------------------------------------------------------------------------- #

def add_global_sim(df, emb, name_to_idx, chunk=50_000):
    known = pl.Series("item", list(name_to_idx.keys())).implode()
    df = df.filter(pl.col("Parameter1").is_in(known) & pl.col("Parameter2").is_in(known))
    i1 = np.fromiter((name_to_idx[p] for p in df["Parameter1"].to_list()), np.int64)
    i2 = np.fromiter((name_to_idx[p] for p in df["Parameter2"].to_list()), np.int64)

    En = torch.nn.functional.normalize(emb, dim=1)
    out = np.empty(len(i1), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(i1), chunk):
            sl = slice(s, s + chunk)
            a = En[torch.as_tensor(i1[sl], device=En.device)]
            b = En[torch.as_tensor(i2[sl], device=En.device)]
            out[sl] = (a * b).sum(dim=1).cpu().numpy()
    return df.with_columns(pl.Series("global_sim", out))


def main(argv=None):
    args = build_argparser().parse_args(argv)

    encoder_dims = parse_dims(args.encoder_dims)
    head_dims = parse_dims(args.head_dims)

    data_root = Path(args.data_root)
    models_root = Path(args.models_root)
    model_dir = models_root / args.emb_model
    model_dir.mkdir(parents=True, exist_ok=True)

    train_pair_path = data_root / "item_correlations.parquet"
    train_emb_path = data_root / args.emb_model / "embeddings_raw.parquet"
    hold_pair_path = data_root / "holdout_item_correlations.parquet"
    hold_emb_path = data_root / args.emb_model / "holdout_embeddings_raw.parquet"
    ckpt_path = model_dir / "dnn_siamese_cor.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, deterministic=not args.nondeterministic)

    def vprint(*a, **k):
        """Setup detail: shown only with --verbose, so a sweep's log stays readable."""
        if args.verbose:
            print(*a, **k)

    # ---- 1. Load item embeddings + pair table; build index ----
    emb_df = pl.read_parquet(train_emb_path)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb")]
    emb_dim = len(emb_cols)
    vprint(f"Item embeddings: {emb_df.shape}  emb_dim={emb_dim}")

    item_to_idx = {name: i for i, name in enumerate(emb_df["item"].to_list())}
    item_emb = torch.tensor(emb_df.select(emb_cols).to_numpy(), dtype=torch.float32).to(device)
    vprint(f"ITEM_EMB on device: {tuple(item_emb.shape)} "
           f"({item_emb.element_size() * item_emb.nelement() / 1e6:.1f} MB)")

    dat = pl.read_parquet(train_pair_path).filter(
        (pl.col("r").is_not_null()) & (pl.col("r") != 1)
    )
    dat = add_global_sim(dat, item_emb, item_to_idx)
    vprint(f"Pair rows (raw): {dat.height:,}")

    known = pl.Series("item", list(item_to_idx.keys())).implode()
    dat = dat.filter(
        pl.col("Parameter1").is_in(known) & pl.col("Parameter2").is_in(known)
    ).select(cs.contains("Para") | cs.contains("prod"), pl.col(["global_sim", "r"]))
    vprint(f"Pair rows (with embeddings): {dat.height:,}")

    aux_numeric = ["global_sim"]

    # ---- 2. item_disjoint split ----
    # sort() before sample(): polars' unique() is multithreaded and does NOT guarantee
    # row order, so the same seed was permuting a differently-ordered item list on every
    # run and silently producing a different item-disjoint split each time. Sorting
    # pins the input order, which is what makes `seed` actually reproduce a split.
    items_all = (
        pl.concat([
            dat.select(pl.col("Parameter1").alias("item")),
            dat.select(pl.col("Parameter2").alias("item")),
        ]).unique(maintain_order=True).to_series().sort().sample(fraction=1.0, seed=args.seed)
    )
    split_idx = int(len(items_all) * (1 - args.outer_val_frac))
    train_items = items_all.slice(0, split_idx).implode()
    outer_items = items_all.slice(split_idx, None).implode()

    train_df = dat.filter(pl.col("Parameter1").is_in(train_items) & pl.col("Parameter2").is_in(train_items))
    outer_df = dat.filter(pl.col("Parameter1").is_in(outer_items) & pl.col("Parameter2").is_in(outer_items))
    dropped = dat.height - train_df.height - outer_df.height
    vprint(f"Split: item-disjoint | train pairs {train_df.height:,} | "
           f"outer val pairs {outer_df.height:,} | dropped mixed pairs: {dropped:,}")

    # ---- 3. Fit aux preprocessor on the training pool only ----
    scaler = QuantileTransformer(
        output_distribution="normal", n_quantiles=1000, subsample=200_000, random_state=args.seed,
    ).fit(train_df.select(pl.col("global_sim")).to_numpy())
    aux_dim = scaler.transform(train_df.select(pl.col("global_sim")).head(1).to_numpy()).shape[1]
    vprint(f"AUX_DIM (numeric) = {aux_dim}")

    def featurize_pairs(df):
        idx1 = np.fromiter((item_to_idx[p] for p in df["Parameter1"].to_list()), dtype=np.int64)
        idx2 = np.fromiter((item_to_idx[p] for p in df["Parameter2"].to_list()), dtype=np.int64)
        aux = scaler.transform(df.select(pl.col("global_sim")).to_numpy())
        y = df.select("r").to_numpy().flatten().astype(np.float32)
        return idx1, idx2, aux, y

    train_idx1, train_idx2, train_aux, train_y = featurize_pairs(train_df)
    outer_idx1, outer_idx2, outer_aux, outer_y = featurize_pairs(outer_df)
    vprint(f"train tensors: idx1 {train_idx1.shape} aux {train_aux.shape} y {train_y.shape}")
    vprint(f"outer tensors: idx1 {outer_idx1.shape} aux {outer_aux.shape} y {outer_y.shape}")

    # ---- 4. Move pair tensors to device ----
    train_idx1_t = torch.tensor(train_idx1, dtype=torch.long, device=device)
    train_idx2_t = torch.tensor(train_idx2, dtype=torch.long, device=device)
    train_aux_t = torch.tensor(train_aux, dtype=torch.float32, device=device)
    train_y_z = torch.tensor(
        np.arctanh(np.clip(train_y, -args.r_clip, args.r_clip)), dtype=torch.float32, device=device
    )

    outer_idx1_t = torch.tensor(outer_idx1, dtype=torch.long, device=device)
    outer_idx2_t = torch.tensor(outer_idx2, dtype=torch.long, device=device)
    outer_aux_t = torch.tensor(outer_aux, dtype=torch.float32, device=device)
    outer_y_r = outer_y.astype(np.float32)

    # ---- 5. Model ----
    from siamese_model import SiameseDNN  # noqa: E402  (import after sys config, matches notebook)

    model = SiameseDNN(
        emb_dim=emb_dim, aux_dim=aux_dim,
        encoder_dims=encoder_dims, head_dims=head_dims,
        dropout=args.dropout, use_skip=args.use_skip,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vprint(model)
    print(f"{args.emb_model} | {device} | seed {args.seed} | {emb_dim}d in -> "
          f"{n_params/1e6:.2f}M params")
    print(f"encoder {encoder_dims} | head {head_dims} | dropout {args.dropout} | "
          f"lr {args.lr:g} | batch {args.batch_size}")
    print(f"pairs: {train_df.height:,} train | {outer_df.height:,} outer-val "
          f"(item-disjoint, {dropped:,} mixed pairs dropped)\n")

    # ---- 6. Training loop ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.sched_patience
    )
    criterion = nn.SmoothL1Loss(beta=args.huber_beta)

    history = {"train_loss": [], "outer_val_rmse": []}
    best_rmse = float("inf")
    since_best = 0
    rng = np.random.default_rng(args.seed)
    n_train = train_idx1_t.shape[0]

    for epoch in range(args.max_epochs):
        model.train()
        perm = rng.permutation(n_train)
        running = 0.0
        for start in range(0, n_train, args.batch_size):
            idx_t = torch.from_numpy(perm[start:start + args.batch_size]).to(device)
            e1 = item_emb[train_idx1_t[idx_t]]
            e2 = item_emb[train_idx2_t[idx_t]]
            aux = train_aux_t[idx_t]
            yb = train_y_z[idx_t]

            optimizer.zero_grad()
            loss = criterion(model(e1, e2, aux), yb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            running += loss.item() * idx_t.size(0)
        train_loss = running / n_train

        model.eval()
        with torch.no_grad():
            chunks = []
            for s in range(0, outer_idx1_t.shape[0], 8192):
                i1c = outer_idx1_t[s:s + 8192]
                i2c = outer_idx2_t[s:s + 8192]
                ac = outer_aux_t[s:s + 8192]
                chunks.append(torch.tanh(model(item_emb[i1c], item_emb[i2c], ac)).cpu().numpy())
            outer_preds_r = np.concatenate(chunks)
        outer_rmse = float(np.sqrt(np.mean((outer_preds_r - outer_y_r) ** 2)))

        scheduler.step(outer_rmse)
        history["train_loss"].append(train_loss)
        history["outer_val_rmse"].append(outer_rmse)

        improved = outer_rmse < best_rmse - 1e-5
        if improved:
            best_rmse = outer_rmse
            since_best = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            since_best += 1

        if (not args.quiet_epochs and epoch % 5 == 0) or since_best >= args.patience:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1:3d} | train Huber(z) {train_loss:.5f} | "
                  f"outer RMSE {outer_rmse:.5f} | best {best_rmse:.5f} | lr {lr_now:.2e}")
        if since_best >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}.")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    vprint(f"Loaded best checkpoint (outer val RMSE = {best_rmse:.5f}) from {ckpt_path}")

    # ---- 7. Export the fitted scaler + metadata ----
    import joblib

    scaler_path = model_dir / "quantile_transformer.joblib"
    scaler_meta_path = model_dir / "quantile_transformer_meta.json"
    joblib.dump(scaler, scaler_path)

    scaler_meta = {
        "emb_model": args.emb_model,
        "aux_numeric": aux_numeric,
        "aux_dim": int(aux_dim),
        "global_sim": "raw cosine similarity over all embedding dims",
        "output_distribution": scaler.get_params()["output_distribution"],
        "n_quantiles": scaler.get_params()["n_quantiles"],
        "subsample": scaler.get_params()["subsample"],
        "random_state": args.seed,
        "n_features_in": int(scaler.n_features_in_),
        "outer_val_frac": args.outer_val_frac,
        "r_clip": args.r_clip,
        "sklearn_version": sklearn.__version__,
        "checkpoint": str(ckpt_path),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(scaler_meta_path, "w") as f:
        json.dump(scaler_meta, f, indent=2)
    vprint(f"Saved scaler -> {scaler_path}")
    vprint(f"Saved meta   -> {scaler_meta_path}")

    # ---- 8. Evaluation on outer val ----
    with torch.no_grad():
        chunks = []
        for s in range(0, outer_idx1_t.shape[0], 8192):
            i1c = outer_idx1_t[s:s + 8192]
            i2c = outer_idx2_t[s:s + 8192]
            ac = outer_aux_t[s:s + 8192]
            chunks.append(torch.tanh(model(item_emb[i1c], item_emb[i2c], ac)).cpu().numpy())
    y_pred = np.concatenate(chunks)
    y_true = outer_y

    corr, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print("\n" + metrics_header())
    print(metrics_row("outer-val", len(y_true), corr, r2, rmse, mae))

    if not args.no_plots:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        ax_l = axes[0]
        ax_l.plot(history["train_loss"], label="train Huber loss (z-space)", color="tab:blue")
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("Train Huber (z)", color="tab:blue")
        ax_l.tick_params(axis="y", labelcolor="tab:blue")
        ax_l.set_yscale("log")
        ax_r = ax_l.twinx()
        ax_r.plot(history["outer_val_rmse"], label="OUTER val RMSE (r-space)", color="tab:orange")
        ax_r.set_ylabel("Outer RMSE (r)", color="tab:orange")
        ax_r.tick_params(axis="y", labelcolor="tab:orange")
        lines_l, lab_l = ax_l.get_legend_handles_labels()
        lines_r, lab_r = ax_r.get_legend_handles_labels()
        ax_l.legend(lines_l + lines_r, lab_l + lab_r, loc="upper right")
        ax_l.set_title("Siamese DNN — training curve (item_disjoint)")
        ax_l.grid(True, alpha=0.3)

        sns.regplot(x=y_true, y=y_pred, ax=axes[1],
                    scatter_kws={"alpha": 0.3, "color": "black"}, line_kws={"color": "red"})
        axes[1].plot([-1, 1], [-1, 1], color="blue", linestyle="--")
        axes[1].set_xlim(-1, 1)
        axes[1].set_ylim(-1, 1)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_title(f"Outer Val: Actual vs Predicted (r = {corr:.3f})")
        axes[1].set_xlabel("Human Correlation (Pearson r)")
        axes[1].set_ylabel("Model Prediction")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = model_dir / "training_curve.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved training/outer-val plot -> {plot_path}")

    # ---- 9. Holdout evaluation ----
    hold_df = pl.read_parquet(hold_pair_path).filter(
        (pl.col("r").is_not_null()) & (pl.col("r") != 1)
    )
    vprint(f"Holdout pair rows on disk: {hold_df.height:,}")

    hold_emb_df = pl.read_parquet(hold_emb_path)
    hold_emb_cols = [c for c in hold_emb_df.columns if c.startswith("emb")]
    assert len(hold_emb_cols) == emb_dim, "Holdout embedding dim doesn't match training dim"

    combined_item_to_idx = dict(item_to_idx)
    new_local_rows, new_names = [], []
    for local_row, name in enumerate(hold_emb_df["item"].to_list()):
        if name not in combined_item_to_idx:
            combined_item_to_idx[name] = len(item_to_idx) + len(new_local_rows)
            new_local_rows.append(local_row)
            new_names.append(name)

    if new_local_rows:
        hold_emb_np = hold_emb_df.select(hold_emb_cols).to_numpy()[new_local_rows]
        hold_emb_new = torch.tensor(hold_emb_np, dtype=torch.float32).to(device)
        combined_emb = torch.cat([item_emb, hold_emb_new], dim=0)
    else:
        combined_emb = item_emb
    vprint(f"Combined embedding lookup: {len(item_to_idx):,} train + {len(new_names):,} new holdout = "
          f"{combined_emb.shape[0]:,} items")

    known_h = pl.Series("item", list(combined_item_to_idx.keys())).implode()
    pre_n = hold_df.height
    hold_df = hold_df.filter(pl.col("Parameter1").is_in(known_h) & pl.col("Parameter2").is_in(known_h))
    if pre_n != hold_df.height:
        print(f"  WARNING: {pre_n - hold_df.height:,} of {pre_n:,} holdout pairs dropped -- "
              f"items missing from the embedding file. Metrics cover {hold_df.height:,} pairs.")
    else:
        vprint(f"After item-lookup filter: {hold_df.height:,} pairs (dropped 0)")

    hold_df = add_global_sim(hold_df, combined_emb, combined_item_to_idx)
    h_idx1 = np.fromiter((combined_item_to_idx[p] for p in hold_df["Parameter1"].to_list()), np.int64)
    h_idx2 = np.fromiter((combined_item_to_idx[p] for p in hold_df["Parameter2"].to_list()), np.int64)

    aux_h = scaler.transform(hold_df.select(aux_numeric).to_numpy())
    y_h = hold_df.select("r").to_numpy().flatten().astype(np.float32)

    h_idx1_t = torch.tensor(h_idx1, dtype=torch.long, device=device)
    h_idx2_t = torch.tensor(h_idx2, dtype=torch.long, device=device)
    aux_h_t = torch.tensor(aux_h, dtype=torch.float32, device=device)

    model.eval()
    chunks = []
    with torch.no_grad():
        for s in range(0, h_idx1_t.shape[0], 8192):
            e1 = combined_emb[h_idx1_t[s:s + 8192]]
            e2 = combined_emb[h_idx2_t[s:s + 8192]]
            ax_ = aux_h_t[s:s + 8192]
            chunks.append(torch.tanh(model(e1, e2, ax_)).cpu().numpy())
    h_preds = np.concatenate(chunks)

    corr_h, _ = pearsonr(y_h, h_preds)
    rmse_h = np.sqrt(mean_squared_error(y_h, h_preds))
    r2_h = r2_score(y_h, h_preds)
    mae_h = mean_absolute_error(y_h, h_preds)

    print(metrics_row("holdout", len(y_h), corr_h, r2_h, rmse_h, mae_h))

    from scipy import stats
    slope, intercept, _, _, se = stats.linregress(y_h, h_preds)
    print(f"\ncalibration slope (holdout): {slope:.4f} +/- {se:.4f} (SE)")

    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.regplot(x=y_h, y=h_preds, ax=ax, scatter_kws={"alpha": 0.3, "color": "black"},
                    line_kws={"color": "red"}, ci=99)
        ax.plot([-1, 1], [-1, 1], color="blue", linestyle="--")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Holdout: Actual vs Predicted (r = {corr_h:.3f})  n = {len(y_h):,}")
        ax.set_xlabel("Human Correlation (Pearson r)")
        ax.set_ylabel("Model Prediction")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = model_dir / "holdout_scatter.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved holdout scatter -> {plot_path}")

    print(f"\ncheckpoint -> {ckpt_path}")
    return {
        "outer_val": {"n": len(y_true), "r": corr, "r2": r2, "rmse": rmse, "mae": mae},
        "holdout": {"n": len(y_h), "r": corr_h, "r2": r2_h, "rmse": rmse_h, "mae": mae_h,
                    "slope": slope, "intercept": intercept},
    }


if __name__ == "__main__":
    main()
