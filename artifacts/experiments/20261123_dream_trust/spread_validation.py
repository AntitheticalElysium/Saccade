"""Does the world's own spread know when its dreamed action ordering is wrong? PLAN.md (2026-11-23).

    .venv/bin/python -m artifacts.experiments.20261123_dream_trust.spread_validation
"""
import importlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S2 = HERE.parent / "20261119_s2"
R = importlib.import_module("artifacts.experiments.20261119_s2.first_action_returns")


def sign(x):
    return np.where(np.abs(x) <= R.EPS, 0, np.sign(x))


def auroc(score, label):
    """P(score of a positive > score of a negative), ties half."""
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):                                                 # average ranks over ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot(fn, episode, rng, reps=2000):
    eps = np.unique(episode); idx = {e: np.flatnonzero(episode == e) for e in eps}
    vals = []
    for _ in range(reps):
        ii = np.concatenate([idx[e] for e in rng.choice(eps, len(eps))])
        v = fn(ii)
        if v is not None:
            vals.append(v)
    return np.quantile(vals, [.025, .975]).tolist() if vals else None


def main():
    rng = np.random.default_rng(20261123)
    with np.load(S2 / "bank_competent.npz") as bank:
        first = np.flatnonzero(bank["kind"] == 1); second = first + 1
        assert (bank["plans"][first, 0] == 5).all() and (bank["plans"][second, 0] == 0).all()
        np.testing.assert_array_equal(bank["rows"][first], bank["rows"][second])
        episode = bank["rows"][first, 0]
    with np.load(S2 / "model_competent.npz") as m, np.load(S2 / "oracle_competent.npz") as o:
        dream = R.returns(m["m_reward"], m["m_done"]); actual = R.returns(o["lab_reward"], o["lab_done"])
    a = sign(actual[first, :8].mean(1) - actual[second, :8].mean(1))
    b = sign(actual[first, 8:].mean(1) - actual[second, 8:].mean(1))
    out = {}
    for h in (1, 4, 15):
        t = h - 1
        x, y = dream[first, :, t], dream[second, :, t]                       # [pairs, 8] dreamed returns per action
        sx, sy = actual[first, :, t], actual[second, :, t]                   # [pairs, 16]
        gap = x.mean(1) - y.mean(1)
        mean_sign = sign(gap)
        use = (a[:, t] == b[:, t]) & (a[:, t] != 0) & (mean_sign != 0)
        wrong = (mean_sign != b[:, t])[use]
        d_spread = np.sqrt((x.var(1, ddof=1) + y.var(1, ddof=1)) / 2)[use]
        se = np.sqrt(x.var(1, ddof=1) / 8 + y.var(1, ddof=1) / 8)
        z = np.abs(gap) / np.maximum(se, 1e-9)
        conf_inv = (1 / np.maximum(z, 1e-9))[use]
        s_spread = np.sqrt((sx.var(1, ddof=1) + sy.var(1, ddof=1)) / 2)[use]
        ep = episode[use]
        res = {"consensus_non_tie": int(use.sum()), "wrong": int(wrong.sum()), "episodes": int(len(np.unique(ep)))}
        if wrong.sum() and (~wrong).sum():
            for name, sc in (("dream_spread", d_spread), ("dream_low_confidence", conf_inv), ("simulator_spread", s_spread)):
                res[name] = {"auroc": auroc(sc, wrong), "ci95": boot(lambda ii: auroc(sc[ii], wrong[ii]), ep, rng)}
            q = np.quantile(s_spread, [1 / 3, 2 / 3]); terc = np.digitize(s_spread, q)
            within = []
            for k in range(3):
                mk = terc == k
                within.append({"n": int(mk.sum()), "wrong": int(wrong[mk].sum()), "auroc": auroc(d_spread[mk], wrong[mk])})
            vals = [w["auroc"] for w in within if w["auroc"] is not None]
            res["dream_spread_within_sim_terciles"] = {"terciles": within, "mean": float(np.mean(vals)) if vals else None}
            zw = z[use][wrong]
            res["wrong_confident_share_z_ge_2"] = float((zw >= 2).mean())
            res["right_confident_share_z_ge_2"] = float((z[use][~wrong] >= 2).mean())
            res["zero_spread_share"] = {"wrong": float((d_spread[wrong] == 0).mean()), "right": float((d_spread[~wrong] == 0).mean())}
        out[str(h)] = res
        print(h, json.dumps(res), flush=True)
    (HERE / "spread_validation.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
