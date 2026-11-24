"""Where the location component's time goes, per frame of one stream (engineering check, 11-24; no result of the
test). Times each part of an estimator step and the read on one panel episode, with the sense withheld on the
training schedule so recognition has real work.

    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261124_learned_location.profile [episode] [frames]
"""
import importlib, json, sys, time
from collections import defaultdict

import numpy as np
import torch

LOC = importlib.import_module("artifacts.experiments.20261124_learned_location.location")
W = importlib.import_module("artifacts.experiments.20261124_learned_location.world")


def main(ep=0, frames=300):
    parts = LOC.load()
    r = torch.load(LOC.LM.DATA / "tokens_panel.pt")[ep]
    T = min(r["T"], frames)
    est = W.MemEstimator(parts, r["T"])
    mask = LOC.outage_mask(r["T"], np.random.default_rng(0))
    d = LOC.sensed(r)
    tm = defaultdict(float)
    orig = est.propose

    def timed_propose(t, p, v):
        t0 = time.perf_counter(); out = orig(t, p, v); tm["recognition (propose)"] += time.perf_counter() - t0
        return out
    est.propose = timed_propose
    with torch.no_grad():
        for t in range(T + 1):
            t0 = time.perf_counter()
            if t == 0:
                x, var = torch.zeros(2, device=LOC.DEV), LOC.SENSED_VAR
            else:
                x, var = est.predict(t, est.tok[t - 1].float(), est.light[t - 1], r["prev_action"][t], None if mask[t] else d[t])
            tm["predict (motion model)"] += time.perf_counter() - t0
            t0 = time.perf_counter(); W.read_feats(est, t, x, parts); tm["read (lookup + features)"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            est.step(t, r["p"][t].float(), r["v"][t].float(), x, var, not mask[t])
            tm["step total"] += time.perf_counter() - t0
            t0 = time.perf_counter(); est.write_extra(t, r["p"][t].float(), r["v"][t].float(), r["g"][t].float(), r["fix"][t])
            tm["write extras"] += time.perf_counter() - t0
    tm["step: write + gate (step minus recognition)"] = tm["step total"] - tm["recognition (propose)"]
    out = {"device": str(LOC.DEV), "frames": T + 1, "ms_per_frame": {k: 1000 * v / (T + 1) for k, v in tm.items()}}
    print(json.dumps(out, indent=1))
    (LOC.HERE / f"profile_{LOC.DEV.type}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
