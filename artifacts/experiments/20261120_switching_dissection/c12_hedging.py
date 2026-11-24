"""C12 (PLAN.md): is the day coin a hedge over where a mob goes next? C5's draws (c5_units.npz) by the mob's free
4-neighbours at t0 under Craftax-Classic's move rule (in view, not solid, not lava / out of bounds, no entity, not the
player's cell). CPU.

    python -m artifacts.experiments.20261120_switching_dissection.c12_hedging
"""
import importlib, json
from pathlib import Path

import numpy as np

from saccade.eval import gates as G, s1 as S

DL = importlib.import_module("artifacts.experiments.20261120_switching_dissection.deletions")
HERE = Path(__file__).resolve().parent
SOLID = {3, 4, 5, 8, 9, 10, 11, 12, 15, 16}                               # craftax_classic constants.SOLID_BLOCKS
BLOCKED = SOLID | {0, 1, 14}                                              # invalid, out of bounds, lava
KINDS = ("none", "zombies", "cows", "skeletons", "arrows")


def main():
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    E = G.episode_grids(L)
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)        # C5's units, exactly (c5_calibration.main)
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    z = np.load(DL.EV.S2 / "model.npz"); mg, mp = z["m_grid"], z["m_pos"].astype(int)
    units = []
    for u in U:
        w, kind, i, j = u[:4]
        e, t0 = int(rows[w, 0]), int(rows[w, 3])
        if 9 <= t0 + 1 <= L["ep_T"][e] and DL.deleted(mg, mp, u):
            units.append((e, t0 + 1, int(kind), int(i), int(j), bool(u[7])))
    c5 = np.load(HERE / "c5_units.npz")
    order = sorted(range(len(units)), key=lambda q: units[q][0])          # c5's loop: episodes in order, units within
    kept_units = [units[q] for q in order]
    keep_rows = [(e, t, k, i, j, n) for (e, t, k, i, j, n) in kept_units if any(0 <= r < 7 and 0 <= c < 9
                 for r in range(i - (L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1])[0] - 1, i - (L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1])[0] + 2)
                 for c in range(j - (L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1])[1] - 1, j - (L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1])[1] + 2))]
    assert len(keep_rows) == len(c5["e"]) and (c5["e"] == [x[0] for x in keep_rows]).all() and (c5["t"] == [x[1] for x in keep_rows]).all() \
        and (c5["kinds"] == [x[2] for x in keep_rows]).all(), "units differ from C5's"
    free = []
    for e, t, k, i, j, n in keep_rows:
        v, g = E[e]["view"][t - 1], E[e]["grid"][t - 1]
        assert g[i, j] == k, "the unit's mob is not at its cell at t0"
        cnt = 0
        for r, c in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            ok = 0 <= r < 7 and 0 <= c < 9 and int(v[r, c]) not in BLOCKED and g[r, c] == 0 and (r, c) != (3, 4)
            cnt += ok
        free.append(cnt)
    free = np.array(free); dele = 1 - c5["kept"].mean(1); kinds = c5["kinds"]; night = c5["night"]
    res = {"units": int(len(free)), "table": {}}
    for kn in (2, 1, 3):
        for tag, sel in (("day", ~night), ("night", night)):
            for f in range(5):
                m = (kinds == kn) & sel & (free == f)
                if m.sum():
                    res["table"][f"{KINDS[kn]}|{tag}|free={f}"] = {"units": int(m.sum()), "deletion": float(dele[m].mean()),
                                                                   "se": float(dele[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else None}
    m = (kinds == 2) & ~night
    hi, lo = dele[m & (free == 4)], dele[m & (free <= 1)]
    rng = np.random.default_rng(0)
    boot = [rng.choice(hi, len(hi)).mean() / max(rng.choice(lo, len(lo)).mean(), 1e-9) for _ in range(5000)]
    res["day_cows_ratio_free4_over_free_le1"] = {"ratio": float(hi.mean() / max(lo.mean(), 1e-9)), "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                                                 "free4": [float(hi.mean()), int(len(hi))], "free_le1": [float(lo.mean()), int(len(lo))]}
    res["day_cows_corr_free_deletion"] = float(np.corrcoef(free[m], dele[m])[0, 1])
    (HERE / "c12_hedging.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res["table"].items():
        print(f"{k:28s} n {v['units']:4d}  deletion {v['deletion']:.3f}" + (f" +- {v['se']:.3f}" if v["se"] is not None else ""))
    print(json.dumps({k: v for k, v in res.items() if k != "table"}, indent=1))


if __name__ == "__main__":
    main()
