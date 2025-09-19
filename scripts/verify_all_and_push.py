#!/usr/bin/env python3
import json, hashlib, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'results' / 'comprehensive_metrics.csv'
FIG_DIR = ROOT / 'clean_figures_final'
TARGETS = [
    ('fig1_scatter_rmse_x2_ude_vs_physics_improved.pdf', None),
    ('fig2_hist_delta_rmse_x2_ude_minus_physics_improved.pdf', None),
    ('fig3_bland_altman_rmse_x2_improved.pdf', None),
    ('fig4_paired_lines_rmse_x2_by_model_improved.pdf', None),
]

PROVENANCE = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def recompute_stats():
    df = pd.read_csv(CSV)
    test = df[df['scenario'].str.startswith('test-')]
    physics = test[test['model'] == 'physics']['rmse_x2'].values
    ude = test[test['model'] == 'ude']['rmse_x2'].values
    delta = ude - physics
    mean_delta = float(np.mean(delta))
    np.random.seed(42)
    bs = [np.mean(np.random.choice(delta, len(delta), replace=True)) for _ in range(10000)]
    ci_lower = float(np.percentile(bs, 2.5))
    ci_upper = float(np.percentile(bs, 97.5))
    wilcoxon_p = float(stats.wilcoxon(delta).pvalue)
    cohens_d = float(mean_delta / np.std(delta))
    std_delta = float(np.std(delta, ddof=1))
    loa_lower = float(mean_delta - 1.96 * std_delta)
    loa_upper = float(mean_delta + 1.96 * std_delta)
    return {
        'n_pairs': int(len(delta)),
        'mean_delta': mean_delta,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'wilcoxon_p': wilcoxon_p,
        'cohens_d': cohens_d,
        'loa_lower': loa_lower,
        'loa_upper': loa_upper,
    }


def regenerate_figures():
    # Regenerate publication-ready figs (1-3)
    subprocess.check_call([sys.executable, str(ROOT / 'scripts' / 'create_publication_ready_figures.py')])
    # Regenerate paired plot (4)
    subprocess.check_call([sys.executable, str(ROOT / 'scripts' / 'create_remaining_figures.py')])


def check_artifacts(stats):
    # Minimal checks: files exist and are non-empty; store hashes.
    failures = []
    for name, _ in TARGETS:
        p = FIG_DIR / name
        if not p.exists() or p.stat().st_size == 0:
            failures.append(f'missing_or_empty:{name}')
        else:
            PROVENANCE[name] = {'sha256': sha256(p), 'bytes': p.stat().st_size}
    return failures


def write_provenance(stats):
    report = {
        'csv_path': str(CSV),
        'csv_sha256': sha256(CSV),
        'stats': stats,
        'figures': PROVENANCE,
        'script_commits': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
    }
    out = ROOT / 'provenance_report.json'
    out.write_text(json.dumps(report, indent=2))
    return out


def main():
    try:
        stats = recompute_stats()
        regenerate_figures()
        failures = check_artifacts(stats)
        if failures:
            print('FAIL: artifact check failed:', failures)
            sys.exit(2)
        rep = write_provenance(stats)
        print('OK: provenance at', rep)
    except Exception as e:
        print('FAIL:', e)
        sys.exit(1)

if __name__ == '__main__':
    main()
