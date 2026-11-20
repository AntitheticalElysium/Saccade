"""C11 (PLAN.md): a refresh schedule for the world-eye's score cache, from a map of checkpoint i's own error read at every
later checkpoint j (deployed: j's chooser C with i's S), beside j's fresh eye and chooser, on C2's DEV frames.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c11_schedule [smoke]
"""
import importlib, itertools, json, sys

import numpy as np
import torch

from saccade import chooser as C
from saccade.eval import s1 as S

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
C1 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c1")
C2 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c2_eye")
C9 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c9_eye_cache")
WIN = C.WIN.numpy()
CK = tuple(range(1, 11))
KEYS = ("any_entity", "zombies", "cows", "changed_hud", "entering_map")


def main(smoke=False):
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    _, light, lookup, loff, cover = C1.semantics(L)
    dl = S.episodes(L)
    eps = np.sort(np.random.default_rng(20261120).choice(len(L["ep_T"]), 150, replace=False))   # C2's frames, exactly
    rng = np.random.default_rng(7)
    picks = {int(e): np.sort(rng.choice(np.arange(9, int(L["ep_T"][e]) + 1), min(20, int(L["ep_T"][e]) - 8), replace=False))
             for e in eps if L["ep_T"][e] >= 10}
    if smoke:
        picks = dict(list(picks.items())[:3])
    keys = [(e, int(t)) for e in picks for t in picks[e]]
    rws = np.array([lookup[loff[e] + t] for e, t in keys])
    Sw, Cw, Hw = {}, {}, {}                                                 # per checkpoint: [n, 49] world S, chooser C and S
    for k in CK:
        model, cfg = AI.load(C9.RUN / f"ckpt_{k:02d}.pt")
        model.requires_grad_(False).eval()
        pol = dict(cfg["fixation"]["chooser"])
        head = C.Head(pol["history"]).to(C9.DEV)
        head.load_state_dict(torch.load(C9.RUN / f"ckpt_{k:02d}.pt", weights_only=False, map_location="cpu")["fixation_controller"]["head"]); head.eval()
        s_, c_, h_ = [], [], []
        for e, ts in picks.items():
            ts = [int(t) for t in ts]
            st = C2.episode(model, cfg, dl, e)
            x = C2.estimates(model, cfg, st, e, ts).float().cpu()
            s_.append(((st[1].p[0, ts].float().cpu() - x) ** 2).sum(-1).numpy()[:, WIN].sum(-1))
            c, s, _ = C9.chooser_scores(model, cfg, head, dl, e)
            c_.append(c[[t - 8 for t in ts]].numpy()); h_.append(s[[t - 8 for t in ts]].numpy())
        Sw[k], Cw[k], Hw[k] = np.concatenate(s_), np.concatenate(c_), np.concatenate(h_)
        del model; torch.cuda.empty_cache()
        print("checkpoint", k, flush=True)
    np.savez_compressed(C9.HERE / f"c11_scores{'_smoke' if smoke else ''}.npz", **{f"{a}{k}": d[k] for a, d in (("S", Sw), ("C", Cw), ("H", Hw)) for k in CK})
    pc = lambda c, s: C.policy(torch.as_tensor(c), torch.as_tensor(s), pol).numpy()

    def cov(prob):
        out = {}
        for name in KEYS:
            mk = cover[name][rws]; present = mk.any(-1); hit = mk[:, WIN].any(-1)
            out[name] = float((prob * hit).sum(-1)[present].mean())
        return out
    fresh = {j: cov(pc(Cw[j], Sw[j])) for j in CK}
    chooser = {j: cov(pc(Cw[j], Hw[j])) for j in CK}
    grid = {(i, j): cov(pc(Cw[j], Sw[i])) for j in CK for i in CK if i < j}
    res = {"frames": len(keys), "fresh": {str(j): v for j, v in fresh.items()}, "chooser": {str(j): v for j, v in chooser.items()},
           "map": {f"{i}->{j}": v for (i, j), v in grid.items()}, "schedules": {}}

    def run(refresh):                                                       # per pass p = 1..10: the eye in use at its end
        out = []
        for p in CK:
            src = [r for r in refresh if r < p]
            out.append(grid[(max(src), p)] if src else chooser[p])
        return out
    sched = {"chooser_throughout": [chooser[p] for p in CK], "fresh_throughout": [fresh[p] for p in CK]}
    for n in (1, 2):
        for rr in itertools.combinations(range(1, 10), n):
            sched["refresh_after_" + "_".join(map(str, rr))] = run(rr)
    for name, per in sched.items():
        res["schedules"][name] = {"mean": {k: float(np.mean([x[k] for x in per])) for k in KEYS}, "per_pass": per}
    base_z = res["schedules"]["chooser_throughout"]["mean"]["zombies"]
    two = [(n, v) for n, v in res["schedules"].items() if n.startswith("refresh_after_") and n.count("_") == 3 and v["mean"]["zombies"] >= base_z - 0.01]
    res["chosen_two_refresh"] = max(two, key=lambda x: x[1]["mean"]["any_entity"])[0] if two else None
    (C9.HERE / f"c11_schedule{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1) + "\n")
    print("map, entity coverage (rows: cache from i; cols: read at j), then fresh / chooser at j", flush=True)
    print("      " + "".join(f"{j:7d}" for j in CK))
    for i in CK[:-1]:
        print(f"{i:5d} " + "".join(f"{grid[(i, j)]['any_entity']:7.3f}" if i < j else " " * 7 for j in CK))
    print("fresh " + "".join(f"{fresh[j]['any_entity']:7.3f}" for j in CK))
    print("chsr  " + "".join(f"{chooser[j]['any_entity']:7.3f}" for j in CK))
    top = sorted(res["schedules"].items(), key=lambda x: -x[1]["mean"]["any_entity"])
    for n, v in top[:12] + [x for x in top if x[0] in ("chooser_throughout", "refresh_after_1", "refresh_after_1_5")]:
        m = v["mean"]; print(f"{n:24s} ent {m['any_entity']:.4f} zom {m['zombies']:.4f} cow {m['cows']:.4f} hud {m['changed_hud']:.4f}", flush=True)
    print("chosen", res["chosen_two_refresh"], flush=True)


if __name__ == "__main__":
    main(len(sys.argv) > 1 and sys.argv[1] == "smoke")
