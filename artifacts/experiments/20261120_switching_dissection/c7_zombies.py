"""C7 (PLAN.md): does P1.5's belief know where zombies are around the player? Linear probes from the corrected belief
b_t (the actor's input) and the predicted belief bp_t (the denoiser's memory), against the S1 reader's mob
probabilities on frame t and t - 1; all DEV episodes, 120 frames each, 80 / 20 by episode.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c7_zombies [smoke]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade.eval import s1 as S

C5 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c5_calibration")
HERE = Path(__file__).resolve().parent
PLAYER = 31                                                                 # row 3, column 4 of the 7 x 9 view
NEAR = [r * 9 + c for r in range(1, 6) for c in range(2, 7) if r * 9 + c != PLAYER]
ADJ = [r * 9 + c for r in range(2, 5) for c in range(3, 6) if r * 9 + c != PLAYER]


def main(smoke=False):
    model, cfg = C5.AI.load(C5.CK)
    model.requires_grad_(False).eval()
    R = C5.S2.load_readers(model, cfg)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    _, _, lookup, loff, cover = C5.C1.semantics(L)
    n = 40 if smoke else None
    enc = S.encode(model, S.first_episodes(L, n), cfg)                     # the first n episodes keep their offsets
    rng = np.random.default_rng(20261121)                                   # C3's order
    eps = [e for e in rng.permutation(len(L["ep_T"])) if L["ep_T"][e] >= 10 and e < len(enc)]
    feats = {}
    for e in eps:
        T = int(L["ep_T"][e])
        keep = np.sort(np.random.default_rng(int(e)).choice(np.arange(1, T + 1), min(120, T), replace=False))
        r = enc[e]
        pr = lambda ts: C5.kind_probs(R, r["p"][ts].float().to(C5.DEV), r["v"][ts].float().to(C5.DEV))[:, :, 1:].flatten(1).cpu()
        z = cover["zombies"][lookup[loff[e] + keep]][:, :63]
        y = np.concatenate([z[:, ADJ].any(1, keepdims=True), z[:, NEAR].any(1, keepdims=True), z.any(1, keepdims=True), z[:, NEAR]], 1)
        feats[e] = (r["b"][keep].float(), r["bp"][keep].float(), pr(keep), pr(keep - 1), y)
    cut = int(0.8 * len(eps)); fit, held = eps[:cut], eps[cut:]
    cat = lambda es, k: torch.cat([feats[e][k] for e in es]) if k < 4 else np.concatenate([feats[e][k] for e in es])
    ytr, yte = cat(fit, 4), cat(held, 4)
    res = {"fit_episodes": len(fit), "held_episodes": len(held), "held_frames": int(len(yte)),
           "base_rate": {"adjacent": float(yte[:, 0].mean()), "within_2": float(yte[:, 1].mean()), "on_screen": float(yte[:, 2].mean()),
                         "per_cell_within_2": float(yte[:, 3:].mean())}, "ap": {}}
    for name, k in (("belief_b_t", 0), ("predicted_belief_bp_t", 1), ("reader_frame_t", 2), ("reader_frame_t_minus_1", 3)):
        _, p = C5.fit_multi(cat(fit, k), ytr, cat(held, k))
        ap = lambda i: C5.C3P.average_precision(yte[:, i].astype(float), p[:, i])
        res["ap"][name] = {"adjacent": ap(0), "within_2": ap(1), "on_screen": ap(2),
                           "per_cell_within_2": C5.C3P.average_precision(yte[:, 3:].reshape(-1).astype(float), p[:, 3:].reshape(-1))}
        print(name, json.dumps(res["ap"][name]), flush=True)
    for name, k in (("reader_frame_t_unfitted", 2), ("reader_frame_t_minus_1_unfitted", 3)):   # P(zombie) itself
        z = cat(held, k).reshape(len(yte), 63, 4)[..., 0].numpy()
        sc = [z[:, ADJ].max(1), z[:, NEAR].max(1), z.max(1)]
        res["ap"][name] = {**{t: C5.C3P.average_precision(yte[:, i].astype(float), sc[i]) for i, t in enumerate(("adjacent", "within_2", "on_screen"))},
                           "per_cell_within_2": C5.C3P.average_precision(yte[:, 3:].reshape(-1).astype(float), z[:, NEAR].reshape(-1))}
    print(json.dumps(res, indent=1), flush=True)
    (HERE / f"c7_zombies{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(len(sys.argv) > 1 and sys.argv[1] == "smoke")
