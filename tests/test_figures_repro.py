import pathlib

def _read_all_scripts():
    root = pathlib.Path('scripts')
    return "\n".join(p.read_text(encoding='utf-8', errors='ignore') for p in root.glob('*.py'))

def test_no_synthetic_results():
    txt = _read_all_scripts()
    assert 'np.random' not in txt and 'randn' not in txt, 'Synthetic np.random/randn present in plotting code.'
    assert '0.541' not in txt and '0.849' not in txt, 'Hard-coded coverage points found.'
    assert 'ax.plot(0.5' not in txt, 'Hard-coded 50% point found.'

def test_has_bland_altman():
    vf = pathlib.Path('scripts')/ 'verify_and_improve_figures.py'
    txt = vf.read_text(encoding='utf-8', errors='ignore')
    assert 'bland_altman(' in txt and 'Bland' in txt, 'Bland–Altman LoA not implemented via helper.'
