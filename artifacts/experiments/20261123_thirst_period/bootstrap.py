"""Episode-bootstrap 95% intervals for the bar readings (P(drawn drop) at the due lag; p = 25 at lag 21), reader-correct
events only, from run/eval_{0,2000}_units.json.  python artifacts/experiments/20261123_thirst_period/bootstrap.py"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
res = {}
for u in (0, 2000):
    rec = json.loads((HERE / "run" / f"eval_{u}_units.json").read_text())
    rng = np.random.default_rng(0)
    out = {}
    for p, lag in ((16, 16), (25, 25), (25, 21), (21, 21), (13, 13), (29, 29)):
        rs = [r for r in rec if r["p"] == p and r["lag"] == lag and r["real_read"] == r["true"]]
        eps = sorted({r["e"] for r in rs})
        by = {e: [np.mean([d == r["prev"] - 1 for d in r["drawn"]]) for r in rs if r["e"] == e] for e in eps}
        pt = np.mean([x for e in eps for x in by[e]])
        bs = [np.mean([x for e in rng.choice(eps, len(eps)) for x in by[e]]) for _ in range(2000)]
        out[f"p{p}_lag{lag}"] = {"p_drawn_drop": round(float(pt), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3),
                                 round(float(np.percentile(bs, 97.5)), 3)], "events": len(rs), "episodes": len(eps)}
    res[str(u)] = out
txt = json.dumps(res, indent=1); print(txt); (HERE / "bootstrap.json").write_text(txt + "\n")
