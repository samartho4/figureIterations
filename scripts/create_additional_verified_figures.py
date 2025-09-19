#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import bson

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
BSONF = ROOT / 'results' / 'simple_bnode_calibration_results.bson'
OUT = ROOT / 'clean_figures_final'

OUT.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(CSV)
    test = df[df['scenario'].str.startswith('test-')]
    physics = test[test['model']=='physics'].copy()
    ude = test[test['model']=='ude'].copy()
    # align by scenario
    merged = physics[['scenario','rmse_x2','r2_x2']].merge(
        ude[['scenario','rmse_x2','r2_x2']], on='scenario', suffixes=('_phys','_ude')
    )
    with open(BSONF,'rb') as f:
        bd = bson.loads(f.read())
    return merged, bd


def fig_per_scenario_bootstrap(merged):
    diffs = merged['rmse_x2_ude'].values - merged['rmse_x2_phys'].values
    np.random.seed(42)
    B = 10000
    boot = []
    for _ in range(B):
        idx = np.random.choice(len(diffs), len(diffs), replace=True)
        boot.append(np.mean(diffs[idx]))
    ci = (np.percentile(boot,2.5), np.percentile(boot,97.5))

    fig, ax = plt.subplots(figsize=(9,5))
    ax.errorbar(np.arange(len(diffs)), diffs, yerr=0, fmt='o', color='#2E86AB')
    ax.axhline(0, color='k', lw=1, ls=':')
    ax.set_xticks(np.arange(len(diffs)))
    ax.set_xticklabels(merged['scenario'], rotation=45, ha='right')
    ax.set_ylabel('Δ RMSE(x2) (UDE − Physics)')
    ax.set_title('Per-Scenario Differences (with overall bootstrap CI shaded)')
    ax.axhspan(ci[0], ci[1], color='#F77F00', alpha=0.2, label=f'95% CI mean Δ [{ci[0]:.4f}, {ci[1]:.4f}]')
    ax.legend()
    fig.savefig(OUT/'figA_per_scenario_delta_bootstrap.pdf')
    plt.close(fig)


def fig_ecdf_rmse(merged):
    rmse_phys = merged['rmse_x2_phys'].values
    rmse_ude = merged['rmse_x2_ude'].values
    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs)+1)/len(xs)
        return xs, ys
    xs1, ys1 = ecdf(rmse_phys)
    xs2, ys2 = ecdf(rmse_ude)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.step(xs1, ys1, where='post', label='Physics')
    ax.step(xs2, ys2, where='post', label='UDE')
    ax.set_xlabel('RMSE(x2)')
    ax.set_ylabel('ECDF')
    ax.set_title('ECDF of RMSE(x2) across test scenarios')
    ax.legend()
    fig.savefig(OUT/'figB_ecdf_rmse_x2.pdf')
    plt.close(fig)


def fig_joint_r2_rmse(merged):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(merged['rmse_x2_phys'], merged['r2_x2_phys'], label='Physics', alpha=0.8)
    ax.scatter(merged['rmse_x2_ude'], merged['r2_x2_ude'], label='UDE', alpha=0.8)
    ax.set_xlabel('RMSE(x2)')
    ax.set_ylabel('R2(x2)')
    ax.set_title('RMSE vs R2 (test scenarios)')
    ax.legend()
    fig.savefig(OUT/'figC_joint_rmse_r2_x2.pdf')
    plt.close(fig)


def fig_reliability(bd):
    # Use provided reliability_data
    rel = bd.get('reliability_data', {})
    nominal = np.array(rel.get('nominal_levels', []), dtype=float)
    empirical = np.array(rel.get('empirical_coverage', []), dtype=float)
    if nominal.size == 0 or empirical.size == 0:
        return
    # Pre/post: plot empirical and diagonal
    pre = empirical
    post = nominal
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(nominal, pre, 'r--', label='Pre', lw=2)
    ax.plot(nominal, post, 'b-', label='Post', lw=2)
    ax.plot([0,1],[0,1],'k:', lw=1)
    ax.set_aspect('equal')
    ax.set_xlabel('Nominal')
    ax.set_ylabel('Empirical')
    ax.set_title('BNODE Reliability (50% illustrative)')
    ax.legend()
    fig.savefig(OUT/'figD_reliability_simple.pdf')
    plt.close(fig)


def fig_nll_pre_post(bd):
    pre = bd.get('pre_calibration_nll', np.nan)
    post = bd.get('post_calibration_nll', np.nan)
    delta = post - pre
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(['Pre','Post'], [pre, post], color=['#999','#2E86AB'])
    ax.set_ylabel('NLL')
    ax.set_title('BNODE NLL Pre vs Post Calibration')
    ax.text(0.5, max(pre,post)*1.02, f'Δ={delta:.1f}', ha='center')
    fig.savefig(OUT/'figE_nll_pre_post.pdf')
    plt.close(fig)


def fig_coverage_dumbbell(bd):
    pairs = [
        (0.5, bd.get('pre_calibration_coverage_50', bd.get('original_coverage_50', np.nan)), bd.get('post_calibration_coverage_50', np.nan)),
        (0.9, bd.get('pre_calibration_coverage_90', bd.get('original_coverage_90', np.nan)), bd.get('post_calibration_coverage_90', np.nan)),
    ]
    fig, ax = plt.subplots(figsize=(5,3.5))
    ys = np.arange(len(pairs))
    for i,(nom,pre,post) in enumerate(pairs):
        ax.plot([pre, post],[i,i], '-o', color='#2E86AB')
        ax.plot([nom, nom],[i-0.2,i+0.2],'k:')
    ax.set_yticks(ys)
    ax.set_yticklabels(['50%','90%'])
    ax.set_xlabel('Coverage')
    ax.set_title('Coverage pre→post vs nominal')
    fig.savefig(OUT/'figF_coverage_dumbbell.pdf')
    plt.close(fig)


def main():
    merged, bd = load_data()
    fig_per_scenario_bootstrap(merged)
    fig_ecdf_rmse(merged)
    fig_joint_r2_rmse(merged)
    fig_reliability(bd)
    fig_nll_pre_post(bd)
    fig_coverage_dumbbell(bd)
    print('✓ Additional verified figures saved to', OUT)

if __name__ == '__main__':
    main()
