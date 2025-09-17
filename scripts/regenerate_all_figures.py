#!/usr/bin/env python3
import subprocess, sys
cmds = [
  [sys.executable, "scripts/verify_and_improve_figures.py"],
  [sys.executable, "scripts/create_remaining_improved_figures.py"],
]
for c in cmds:
  print(">>>", " ".join(c), flush=True)
  subprocess.check_call(c)
print("All figures regenerated.")
