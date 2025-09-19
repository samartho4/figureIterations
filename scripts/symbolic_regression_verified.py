#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'results' / 'comprehensive_metrics.csv'
OUT = ROOT / 'clean_figures_final'
OUT.mkdir(parents=True, exist_ok=True)


def load_merge():
    df = pd.read_csv(CSV)
    test = df[df['scenario'].str.startswith('test-')]
    phys = test[test['model']=='physics'][['scenario','rmse_x2']].rename(columns={'rmse_x2':'rmse_phys'})
    ude = test[test['model']=='ude'][['scenario','rmse_x2']].rename(columns={'rmse_x2':'rmse_ude'})
    m = phys.merge(ude, on='scenario')
    return m


def bic(n, rss, k):
    return n*np.log(rss/n) + k*np.log(n)


def fit_symbolic(m: pd.DataFrame):
    x = m['rmse_phys'].values
    y = m['rmse_ude'].values
    n = len(x)
    best = None
    for deg in [1,2,3]:
        coeffs = np.polyfit(x, y, deg)
        yhat = np.polyval(coeffs, x)
        rss = float(np.sum((y - yhat)**2))
        k = deg+1
        bic_val = bic(n, rss, k)
        r2 = 1 - rss/float(np.sum((y - np.mean(y))**2))
        cand = {'deg': deg, 'coeffs': coeffs.tolist(), 'bic': bic_val, 'r2': float(r2)}
        best = cand if best is None or cand['bic'] < best['bic'] else best
    return best


def plot_with_equation(m: pd.DataFrame, model):
    x = m['rmse_phys'].values
    y = m['rmse_ude'].values
    coeffs = np.array(model['coeffs'])
    xx = np.linspace(min(x)*0.95, max(x)*1.05, 200)
    yy = np.polyval(coeffs, xx)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(x, y, color='#2E86AB', edgecolors='white', linewidth=1.2)
    ax.plot(xx, yy, color='#E63946', lw=2, label=f'deg={model["deg"]}')
    ax.plot([xx.min(), xx.max()], [xx.min(), xx.max()], 'k--', alpha=0.5, lw=1)
    ax.set_xlabel('Physics RMSE(x2)')
    ax.set_ylabel('UDE RMSE(x2)')
    c = model['coeffs']
    # Format polynomial string high→low degree
    terms = []
    d = model['deg']
    for i,coef in enumerate(c):
        p = d - i
        if p==0:
            terms.append(f'{coef:.4f}')
        elif p==1:
            terms.append(f'{coef:.4f} x')
        else:
            terms.append(f'{coef:.4f} x^{p}')
    eq = ' + '.join(terms)
    text = (f'Best deg={model["deg"]}\n'
            f'y = {eq}\n'
            f'R2 = {model["r2"]:.4f}\n'
            f'BIC = {model["bic"]:.2f}')
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='gray'))
    ax.legend()
    fig.savefig(OUT/'figSR_ude_vs_physics_polyfit.pdf')
    plt.close(fig)


def main():
    m = load_merge()
    model = fit_symbolic(m)
    plot_with_equation(m, model)
    # Save JSON for verification
    (OUT/'figSR_model.json').write_text(json.dumps(model, indent=2))
    print('✓ Symbolic regression figure and model saved to', OUT)

if __name__ == '__main__':
    main()
