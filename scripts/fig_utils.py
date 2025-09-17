import os, json, math, time, gzip, hashlib, subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def git_short_hash():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def provenance_footer(ax, *paths):
    sha = git_short_hash()
    paths = [str(p) for p in paths if p]
    footer = "Data: " + "; ".join(paths) + f" | Commit: {sha}"
    ax.text(0.0, -0.20, footer, transform=ax.transAxes, fontsize=8, va="top", ha="left", alpha=0.8)


def load_json_or_none(p):
    p = Path(p)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_import_bson():
    try:
        import bson
        return bson
    except Exception:
        return None


def load_bson_or_npz(p):
    p = Path(p)
    if not p.exists():
        return None
    if p.suffix == ".npz":
        return dict(np.load(p, allow_pickle=True))
    b = safe_import_bson()
    if b:
        with open(p, "rb") as f:
            try:
                return b.BSON(f.read()).decode()
            except Exception:
                return b.loads(f.read())
    return None


def ensure_results_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def set_matplotlib_defaults():
    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def bland_altman(y_true, y_pred):
    diffs = np.asarray(y_pred) - np.asarray(y_true)
    means = 0.5 * (np.asarray(y_pred) + np.asarray(y_true))
    bias = diffs.mean()
    sd = diffs.std(ddof=1)
    loa_low, loa_high = bias - 1.96 * sd, bias + 1.96 * sd
    return means, diffs, bias, loa_low, loa_high


def empirical_coverage(y, pred_quantiles, qs):
    y = np.asarray(y)
    cov = []
    for q in qs:
        lo, hi = (1 - q) / 2, (1 + q) / 2
        if isinstance(pred_quantiles, dict):
            L = pred_quantiles.get(lo)
            U = pred_quantiles.get(hi)
            if L is None or U is None:
                cov.append(np.nan)
                continue
            L = np.asarray(L)
            U = np.asarray(U)
        else:
            samples = np.asarray(pred_quantiles)
            L = np.quantile(samples, lo, axis=0)
            U = np.quantile(samples, hi, axis=0)
        cov.append(np.mean((y >= L) & (y <= U)))
    return np.array(cov)
