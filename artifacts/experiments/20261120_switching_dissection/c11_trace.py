"""C11b (PLAN.md): P1.5's chooser looks in training against their paired uniform windows, by update (CPU).

    python -m artifacts.experiments.20261120_switching_dissection.c11_trace
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import importlib, json
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D
from saccade.eval import s1 as S, gates as G

T = importlib.import_module("artifacts.experiments.20261104_p1f2_review.trace")
HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "20261116_p15/run"
KEYS = ("any_entity", "zombies", "cows", "changed_hud", "entering_map")


def main():
    ck = torch.load(RUN / "ckpt_10.pt", map_location="cpu", weights_only=False)
    raw = np.memmap(RUN / "gaze.bin", dtype=C.TRACE, mode="r", shape=(ck["fixation_controller"]["trace_rows"],))
    eps = D.load_split(ck["cfg"])["train"]; off = D.frame_offsets(eps)
    labels = dict(np.load(S.LABELS / "s0_train400_labels.npz")); keys = {(e.shard, e.slot): i for i, e in enumerate(eps)}
    mapping = np.full(len(eps), -1, int)
    for le, key in enumerate(zip(labels["ep_shard"], labels["ep_slot"])):
        if tuple(key) in keys:
            mapping[keys[tuple(key)]] = le
    kept, positions = [], []
    for start in range(0, len(raw), 262144):
        q = raw[start:start + 262144]; ei = np.searchsorted(off, q["target"], side="right") - 1
        take = (q["target"] >= 0) & (mapping[ei] >= 0)
        kept.append(np.array(q[take])); positions.append(np.flatnonzero(take) + start)
    q = np.concatenate(kept); position = np.concatenate(positions); ei = np.searchsorted(off, q["target"], side="right") - 1
    ep = mapping[ei]; tt = q["target"] - off[ei]; loff = np.r_[0, np.cumsum(labels["ep_T"] + 1)]
    assert ((tt >= 1) & (tt <= labels["ep_T"][ep])).all()
    T.A.L = labels; T.A.E = G.episode_grids(labels); T.A.OFF = loff; T.A.AOFF = np.r_[0, np.cumsum(labels["ep_T"])]
    eall = np.concatenate([np.full(int(n), e) for e, n in enumerate(labels["ep_T"])])
    tall = np.concatenate([np.arange(1, int(n) + 1) for n in labels["ep_T"]]); si = loff[eall] + tall
    sem = T.training_semantics(labels, eall, tall, si, loff)
    lookup = np.full(int(loff[-1]), -1, int); lookup[si] = np.arange(len(si)); rows = lookup[loff[ep] + tt]
    assert (rows >= 0).all()
    # rows per update: 16 streams x 64 targets in the first half (12,500 updates), 16 x 256 after
    first = 12_500 * 16 * 64
    update = np.where(position < first, position // (16 * 64), 12_500 + (position - first) // (16 * 256))
    edges = list(range(0, 2_501, 250)) + [5_000, 7_500, 10_000, 12_500, 13_125, 13_750, 14_375, 15_000, 15_625]
    win = C.WIN.numpy(); w = q["window"].astype(int); u = q["uniform"].astype(int)
    res = {"rows": int(len(q)), "bins": []}
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (update >= a) & (update < b)
        out = {"updates": [a, b], "rows": int(sel.sum())}
        for name in KEYS:
            mk = sem["masks"][name][rows[sel]]; present = mk.any(-1)
            hl = mk[np.arange(len(mk))[:, None], win[w[sel]]].any(-1); hu = mk[np.arange(len(mk))[:, None], win[u[sel]]].any(-1)
            out[name] = {"chooser": float(hl[present].mean()), "uniform": float(hu[present].mean()), "n": int(present.sum())}
        res["bins"].append(out)
        print(f"{a:6d}-{b:6d} n {out['rows']:7d}  " + "  ".join(f"{k[:6]} {out[k]['chooser']:.3f}/{out[k]['uniform']:.3f}" for k in KEYS), flush=True)
    (HERE / "c11_trace.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
