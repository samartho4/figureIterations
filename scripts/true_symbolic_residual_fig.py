#!/usr/bin/env python3
"""
True symbolic residual extraction (no synthetic fallback).
- Loads results/ude_f_Pgen_samples.csv with columns: P_gen, f_residual[, scenario]
- Fits candidate models (poly deg 1..5; optional simple rational) and selects by BIC
- Performs k-fold CV for the chosen model and reports CV-R2
- Saves figure: clean_figures_final/figSYM_residual_true.pdf (scatter + fit + residuals)
- Saves JSON: results/symbolic_fit.json with coefficients, model, R2, BIC, CV scores, CSV SHA256
If CSV is missing or malformed, exits non-zero (to enforce “real-only”).
"""

from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.2,
})

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'results' / 'ude_f_Pgen_samples.csv'
FIG = ROOT / 'clean_figures_final' / 'figSYM_residual_true.pdf'
META = ROOT / 'results' / 'symbolic_fit.json'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def design_poly(x: np.ndarray, deg: int) -> np.ndarray:
    return np.vstack([x**p for p in range(deg+1)]).T


def fit_poly(x, y, deg):
    X = design_poly(x, deg)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    rss = float(np.sum((y - yhat)**2))
    tss = float(np.sum((y - y.mean())**2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    n, k = len(x), len(coef)
    bic = n*np.log(rss/n) + k*np.log(n)
    return {'family': f'poly{deg}', 'deg': deg, 'coef': coef.tolist(), 'r2': float(r2), 'bic': float(bic)}


def cv_r2(model, x, y, n_splits=5):
    kf = KFold(n_splits=min(n_splits, len(x)), shuffle=True, random_state=42)
    scores = []
    for tr, te in kf.split(x):
        m = fit_poly(x[tr], y[tr], model['deg'])
        Xte = design_poly(x[te], model['deg'])
        yhat = Xte @ np.array(m['coef'])
        tss = float(np.sum((y[te] - y[te].mean())**2))
        rss = float(np.sum((y[te] - yhat)**2))
        r2 = 1 - rss / tss if tss > 0 else np.nan
        scores.append(r2)
    return float(np.nanmean(scores)), float(np.nanstd(scores))


def main():
    if not CSV.exists():
        print(f'ERROR: missing required CSV {CSV}. Provide real residual samples (P_gen,f_residual).', file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(CSV)
    if not {'P_gen','f_residual'}.issubset(df.columns):
        print('ERROR: CSV must contain columns P_gen,f_residual', file=sys.stderr)
        sys.exit(2)
    x = df['P_gen'].astype(float).to_numpy()
    y = df['f_residual'].astype(float).to_numpy()
    if len(x) < 10:
        print('ERROR: need at least 10 samples', file=sys.stderr)
        sys.exit(2)

    # Fit candidates (poly 1..5)
    cands = [fit_poly(x, y, d) for d in range(1, 6)]
    best = min(cands, key=lambda m: m['bic'])
    cv_mean, cv_std = cv_r2(best, x, y, n_splits=5)

    # Figure scatter + fit + residuals
    xs = np.linspace(x.min()*0.98, x.max()*1.02, 400)
    Xs = design_poly(xs, best['deg'])
    yhat = Xs @ np.array(best['coef'])
    yfit = design_poly(x, best['deg']) @ np.array(best['coef'])
    resid = y - yfit

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6,6), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax.scatter(x, y, s=16, alpha=0.8, color='#2E86AB', edgecolors='white', linewidth=0.6, label='Samples')
    ax.plot(xs, yhat, color='#E63946', lw=2, label=f"{best['family']} fit")
    ax.plot([xs.min(), xs.max()], [0,0], 'k:', lw=0.8, alpha=0.5)
    ax.set_ylabel('f_θ(P_gen)')
    ax.set_title('True Symbolic Residual (from CSV)')
    ax.legend()

    axr.axhline(0, color='k', lw=0.8)
    axr.scatter(x, resid, s=12, alpha=0.8, color='#555555')
    axr.set_xlabel('P_gen')
    axr.set_ylabel('resid')

    # Overlay metrics
    eq_terms = []
    for p, c in enumerate(best['coef']):
        if p == 0:
            eq_terms.append(f"{c:.6f}")
        elif p == 1:
            eq_terms.append(f"{c:.6f} x")
        else:
            eq_terms.append(f"{c:.6f} x^{p}")
    eq = ' + '.join(eq_terms)
    text = (f"Model: {best['family']}  R2={best['r2']:.4f}\n"
            f"CV-R2={cv_mean:.4f}±{cv_std:.4f}\n"
            f"BIC={best['bic']:.2f}\n"
            f"y = {eq}")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='gray'))

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    plt.close(fig)

    META.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        'family': best['family'], 'deg': best['deg'], 'coef': best['coef'],
        'r2': best['r2'], 'bic': best['bic'], 'cv_r2_mean': cv_mean, 'cv_r2_std': cv_std,
        'source_csv': str(CSV), 'source_csv_sha256': sha256(CSV)
    }
    META.write_text(json.dumps(meta, indent=2))
    print('✓ Wrote', FIG)
    print('✓ Wrote', META)

if __name__ == '__main__':
    main()
