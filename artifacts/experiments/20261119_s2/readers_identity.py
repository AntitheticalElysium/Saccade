"""One process: the committed s2b.py (HEAD) against the edited one on a 16-start route smoke: free and record_real must
be bitwise equal; the two reader arms must keep the first generated frame and differ from both."""
import gc, importlib, importlib.util, os, subprocess, sys
import numpy as np, torch
assert "S2_CKPT" not in os.environ
ROOT = "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA"
old_src = subprocess.run(["git", "-C", ROOT, "show", "HEAD:artifacts/experiments/20261119_s2/s2b.py"], capture_output=True, text=True, check=True).stdout
path = os.path.join(os.path.dirname(__file__), "s2b_old.py"); open(path, "w").write(old_src)
new = importlib.import_module("artifacts.experiments.20261119_s2.s2b")
spec = importlib.util.spec_from_file_location("s2b_old", path); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
old.routes(16, ("free", "record_real"), "chk_old"); gc.collect(); torch.cuda.empty_cache()
new.routes(16, ("free", "record_real", "dread_real", "mread_real"), "chk_new"); gc.collect()
H = new.HERE
a, b = np.load(H / "chk_old_smoke.npz"), np.load(H / "chk_new_smoke.npz")
for k in a.files:
    print(k, "equal" if np.array_equal(a[k], b[k]) else "DIFFERS")
for m in ("dread_real", "mread_real"):
    g = b[f"{m}_m_grid"]
    print(m, "first frame equal to free:", np.array_equal(g[:, :, :2], b["free_m_grid"][:, :, :2]),
          "| differs from free:", not np.array_equal(g, b["free_m_grid"]), "| differs from record_real:", not np.array_equal(g, b["record_real_m_grid"]))
for f in ("chk_old_smoke.npz", "chk_new_smoke.npz"):
    os.remove(H / f)
