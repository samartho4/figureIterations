#!/usr/bin/env python3
import os
from pathlib import Path
from scripts.fig_manifest import FIGS

# These two modules already exist in your repo per your notes:
# - scripts/create_remaining_improved_figures.py  (makes new panels)
# - scripts/verify_and_improve_figures.py         (QA pass: footer, dpi, square axes, colors)
try:
    import scripts.create_remaining_improved_figures as create
except Exception as e:
    create = None
    print("[warn] create_remaining_improved_figures not importable:", e)
try:
    import scripts.verify_and_improve_figures as qa
except Exception as e:
    qa = None
    print("[warn] verify_and_improve_figures not importable:", e)

OUT = Path("clean_figures_final")
OUT.mkdir(parents=True, exist_ok=True)


def _save_both(extless_path: str, fig):
    png = OUT / f"{extless_path}.png"
    pdf = OUT / f"{extless_path}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    print("[ok]", png, "and", pdf)


def main():
    # 1) Generate / regenerate raw figures (no randomness in result-bearing plots).
    generated = []
    if create and hasattr(create, "generate_all"):
        # preferred: a single entry point in your module
        created = create.generate_all(OUT)
        generated.extend(created or [])
    else:
        # fallback: call individual creators if exposed, otherwise skip silently
        for key in FIGS.keys():
            fn = getattr(create, f"make_{key}", None) if create else None
            if fn is None:
                print(f"[skip] no generator for {key}")
                continue
            fig = fn()  # must return a Matplotlib Figure
            _save_both(key, fig)
            generated.append(key)

    # 2) QA / improvements pass (footer, square reliability, colorblind palette, units, N).
    if qa and hasattr(qa, "run_all"):
        qa.run_all(OUT)  # your existing verifier/improver; idempotent
    elif qa:
        # Try conventional helpers if available
        for fname in sorted(OUT.glob("*_improved.png")):
            try:
                if hasattr(qa, "ensure_provenance_footer"):
                    qa.ensure_provenance_footer(fname)
                if "reliability" in fname.name and hasattr(qa, "ensure_square_axes"):
                    qa.ensure_square_axes(fname)
                if hasattr(qa, "ensure_colorblind_safe"):
                    qa.ensure_colorblind_safe(fname)
                if hasattr(qa, "ensure_units_and_N"):
                    qa.ensure_units_and_N(fname)
            except Exception as e:
                print("[warn] QA step failed for", fname, ">", e)

    # 3) Final report
    have = sorted([p.name for p in OUT.glob("*_improved.png")])
    print("\n[summary] improved PNGs:", len(have))
    for h in have:
        print("  -", h)


if __name__ == "__main__":
    main()
