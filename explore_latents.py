from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ProbeMetrics:
    test_auc: float
    shuffle_test_auc: float
    n_train: int
    n_val: int
    n_test: int
    label_counts: dict[str, dict[str, int]]
    settings: dict[str, object]


def _label_counts(y: np.ndarray) -> dict[str, int]:
    uniq, cnt = np.unique(y, return_counts=True)
    return {str(int(u)): int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}


def load_latents(path: Path) -> dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def plot_umap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    out_path: Path,
    seed: int,
    pca_dim: int | None,
    n_neighbors: int,
    min_dist: float,
) -> None:
    import umap  # type: ignore[import-not-found]  # pylint: disable=import-error

    Xtr, Xva, Xte = X_train, X_val, X_test
    if pca_dim is not None:
        pca = PCA(n_components=min(int(pca_dim), Xtr.shape[1]), random_state=int(seed))
        Xtr = pca.fit_transform(Xtr)
        Xva = pca.transform(Xva)
        Xte = pca.transform(Xte)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        random_state=int(seed),
        metric="euclidean",
        low_memory=True,
    )
    Etr = reducer.fit_transform(Xtr)
    Eva = reducer.transform(Xva)
    Ete = reducer.transform(Xte)

    def scatter(ax, E, y, title, alpha):
        for lab, color in [(0, "tab:blue"), (1, "tab:orange")]:
            mask = (y == lab)
            ax.scatter(
                E[mask, 0],
                E[mask, 1],
                s=14,
                alpha=alpha,
                c=color,
                label=f"label={lab}",
                linewidths=0,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    scatter(ax1, Etr, y_train, "UMAP (train)", alpha=0.65)
    scatter(ax2, Eva, y_val, "UMAP (val)", alpha=0.8)
    scatter(ax3, Ete, y_test, "UMAP (test)", alpha=0.9)
    ax3.legend(loc="best", frameon=False)

    fig.suptitle(
        f"CVAE latents (mu): UMAP(2D), seed={seed}, pca_dim={pca_dim}, "
        f"n_neighbors={n_neighbors}, min_dist={min_dist}",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def linear_probe_auc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    shuffle: bool,
) -> float:
    rng = np.random.default_rng(int(seed))
    ytr = y_train.copy()
    if shuffle:
        rng.shuffle(ytr)

    clf = LogisticRegression(
        max_iter=2000,
        random_state=int(seed),
        class_weight="balanced",
    )
    clf.fit(X_train, ytr)
    prob = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, prob))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=str, default="checkpoints/cvae_latents.npz")
    ap.add_argument("--out-plot", type=str, default="img/umap_mu.png")
    ap.add_argument("--out-metrics", type=str, default="data/latent/probe_metrics.json")
    ap.add_argument(
        "--embedding",
        type=str,
        default="mu_y0",
        choices=["mu", "mu_y0", "mu_y1"],
        help=(
            "Which latent mean embedding to analyze. "
            "`mu` is label-conditioned (may leak label). "
            "`mu_y0`/`mu_y1` encode all samples with a fixed label and are preferred."
        ),
    )
    ap.add_argument("--seed", type=int, default=440)
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    args = ap.parse_args()

    lat_path = Path(args.latents)
    out_plot = Path(args.out_plot)
    out_metrics = Path(args.out_metrics)

    lat = load_latents(lat_path)
    emb = str(args.embedding)
    required = [
        f"{emb}_train",
        f"{emb}_val",
        f"{emb}_test",
        "labels_train",
        "labels_val",
        "labels_test",
    ]
    missing = [k for k in required if k not in lat]
    if missing:
        raise KeyError(f"missing keys in {lat_path}: {missing}")

    X_train = np.asarray(lat[f"{emb}_train"], dtype=np.float32)
    X_val = np.asarray(lat[f"{emb}_val"], dtype=np.float32)
    X_test = np.asarray(lat[f"{emb}_test"], dtype=np.float32)
    y_train = np.asarray(lat["labels_train"], dtype=np.int64)
    y_val = np.asarray(lat["labels_val"], dtype=np.int64)
    y_test = np.asarray(lat["labels_test"], dtype=np.int64)

    print(f"loaded {lat_path}")
    print(f"embedding: {emb}")
    print(
        "shapes:",
        f"{emb}_train",
        X_train.shape,
        f"{emb}_val",
        X_val.shape,
        f"{emb}_test",
        X_test.shape,
    )
    print(
        "label_counts:",
        "train",
        _label_counts(y_train),
        "val",
        _label_counts(y_val),
        "test",
        _label_counts(y_test),
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    plot_umap(
        X_train_s,
        y_train,
        X_val_s,
        y_val,
        X_test_s,
        y_test,
        out_path=out_plot,
        seed=args.seed,
        pca_dim=args.pca_dim,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    print(f"saved {out_plot}")

    test_auc = linear_probe_auc(
        X_train_s, y_train, X_test_s, y_test, seed=args.seed, shuffle=False
    )
    shuffle_auc = linear_probe_auc(
        X_train_s, y_train, X_test_s, y_test, seed=args.seed, shuffle=True
    )

    metrics = ProbeMetrics(
        test_auc=test_auc,
        shuffle_test_auc=shuffle_auc,
        n_train=int(X_train.shape[0]),
        n_val=int(X_val.shape[0]),
        n_test=int(X_test.shape[0]),
        label_counts={
            "train": _label_counts(y_train),
            "val": _label_counts(y_val),
            "test": _label_counts(y_test),
        },
        settings={
            "embedding": emb,
            "seed": int(args.seed),
            "pca_dim": int(args.pca_dim),
            "umap_neighbors": int(args.umap_neighbors),
            "umap_min_dist": float(args.umap_min_dist),
            "latents": str(lat_path),
            "out_plot": str(out_plot),
            "out_metrics": str(out_metrics),
        },
    )
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(
        json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"test ROC-AUC: {test_auc:.4f} | shuffle ROC-AUC: {shuffle_auc:.4f}")
    print(f"saved {out_metrics}")


if __name__ == "__main__":
    main()

