"""Where do the fourth localizer's natural relocalisations go? On the panel's first episodes (simulator positions score
only), at every jump (Chebyshev > 1 from the last pose): the true step, the jump's size, how many of the 63 cells have
a present entry at the winning pose vs the true pose, both poses' summed memory scores, and day / dark.

    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_located_memory.diagnosis_v4.jumps [n_eps]
"""
import importlib, json, sys

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v4.memory_v4")


@torch.no_grad()
def main(n_eps=3):
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / "localizer_v4.pt", map_location="cpu")); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[:n_eps]
    rows = []
    for ei, r in enumerate(data):
        mp = LM.Map(1, torch.device("cpu")); pos = r["pos"]; tc = LM.truth_cells(pos)
        for t in range(r["T"] + 1):
            one = lambda k: r[k][t][None]
            first = torch.tensor([t == 0]); valid = torch.tensor([True]); tt = torch.tensor([t])
            before = mp.pose.clone(); ent = mp.view()[0]
            LM.advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"), one("prev_action"),
                       tt, tt, first, valid, truth=tc[t][None])
            if t > 0 and (mp.pose[0] - before[0]).abs().max() > 1:
                L = mp.last_L[0]
                true_pose = before[0] + (pos[t] - pos[t - 1])
                def present(pz):
                    rr = pz[0] + LM.KR; cc = pz[1] + LM.KC
                    ok = (rr >= 0) & (rr < LM.G) & (cc >= 0) & (cc < LM.G)
                    e = ent[0, rr.clamp(0, LM.G - 1), cc.clamp(0, LM.G - 1)]
                    return int((ok & (e[..., LM.CTIME] >= 0)).sum())
                rows.append({"ep": ei, "t": t, "dark": float(r["light"][t]) < 0.5,
                             "true_step": (pos[t] - pos[t - 1]).tolist(), "jump": (mp.pose[0] - before[0]).tolist(),
                             "present_win": present(mp.pose[0]), "present_true": present(true_pose),
                             "L_win": round(float(L[mp.pose[0, 0], mp.pose[0, 1]]), 1),
                             "L_true": round(float(L[true_pose[0], true_pose[1]]), 1)})
    a = lambda k: np.array([x[k] for x in rows], dtype=float)
    out = {"jumps": len(rows), "dark_share": float(a("dark").mean()) if rows else None,
           "present_win_median": float(np.median(a("present_win"))) if rows else None,
           "present_true_median": float(np.median(a("present_true"))) if rows else None,
           "win_has_no_entries_share": float((a("present_win") == 0).mean()) if rows else None,
           "L_win_median": float(np.median(a("L_win"))) if rows else None, "L_true_median": float(np.median(a("L_true"))) if rows else None,
           "examples": rows[:12]}
    txt = json.dumps(out, indent=1); print(txt)
    (LM.HERE / "diagnosis_v4" / f"jumps_localizer_v4_{n_eps}eps.json").write_text(txt + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
