"""Check 15's inventory rows read a true token change of 0.000 in P1.5 (stone / coal / iron +1, sapling +/-1) against
12-29 in F6.1. Measurement (wrong cells) or encoder (P1.5's tokens blind to those digits)? For each world, the mean
squared change of each HUD cell's coarse token (cells 63-80, check 15's pass 1 on DEV fold 4) on each inventory
count change, and the cell that changes most. Perception only.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_p15.inventory_tokens
"""
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
IF = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
WORLDS = {"f61": HERE.parent / "20261114_p1f61/run/ckpt_10.pt", "p15": HERE / "run/ckpt_10.pt"}

if __name__ == "__main__":
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    eps = S.episodes(L)
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    fold = list(range(4, len(eps), 5))
    inv = L["s_inv"].astype(int)
    res = {"episodes": len(fold)}
    for name, ck in WORLDS.items():
        model, cfg = IF.load(str(ck))
        sums, counts = {}, {}
        with torch.no_grad():
            for e in fold:
                ft = torch.as_tensor(D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"].numpy(), device=S.DEV)
                p = torch.cat([C.coarse(model.perception, z)[0] for z in ft.split(512)]).float()
                d = (p[1:, 63:] - p[:-1, 63:]).square().sum(-1).cpu().numpy()            # [T, 18] cells 63-80
                di = np.sign(np.diff(inv[off[e]:off[e + 1]], axis=0))                       # [T, 12]
                for i in range(inv.shape[1]):
                    for s, sg in (("up", 1), ("down", -1)):
                        m = di[:, i] == sg
                        if m.any():
                            k = f"inv{i}_{s}"
                            sums[k] = sums.get(k, 0) + d[m].sum(0)
                            counts[k] = counts.get(k, 0) + int(m.sum())
                quiet = (di == 0).all(1)
                sums["quiet"] = sums.get("quiet", 0) + d[quiet].sum(0)
                counts["quiet"] = counts.get("quiet", 0) + int(quiet.sum())
        res[name] = {}
        for k in sorted(sums):
            mean = sums[k] / counts[k]
            res[name][k] = {"events": counts[k], "cell": 63 + int(mean.argmax()), "change": float(mean.max()),
                            "by_cell": [round(float(x), 3) for x in mean]}
        print(name, {k: (v["events"], v["cell"], round(v["change"], 2)) for k, v in res[name].items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    (HERE / "inventory_tokens.json").write_text(json.dumps(res, indent=1) + "\n")
