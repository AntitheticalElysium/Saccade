"""C9, early training (PLAN.md, added after C9's early check): a score cache one pass stale at checkpoints 2-5 (checkpoint
j - 1's estimate against its own tokens, read at checkpoint j against j's fresh eye), contingency off and deployed with
checkpoint j's own chooser head; and the chooser at j. C2's DEV frames.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c9_early
"""
import importlib, json

import numpy as np
import torch

from saccade import chooser as C
from saccade.eval import s1 as S

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
C1 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c1")
C2 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c2_eye")
C9 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c9_eye_cache")
WIN = C.WIN.numpy()


def main():
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    _, light, lookup, loff, cover = C1.semantics(L)
    dl = S.episodes(L)
    eps = np.sort(np.random.default_rng(20261120).choice(len(L["ep_T"]), 150, replace=False))
    rng = np.random.default_rng(7)
    picks = {int(e): np.sort(rng.choice(np.arange(9, int(L["ep_T"][e]) + 1), min(20, int(L["ep_T"][e]) - 8), replace=False))
             for e in eps if L["ep_T"][e] >= 10}
    keys = [(e, int(t)) for e in picks for t in picks[e]]
    rws = np.array([lookup[loff[e] + t] for e, t in keys])
    S_own, Cw, Sch = {}, {}, {}
    for k in (1, 2, 3, 4, 5):
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
            s_.append(((st[1].p[0, ts].float().cpu() - x) ** 2).sum(-1).numpy())
            c, s, _ = C9.chooser_scores(model, cfg, head, dl, e)
            c_.append(c[[t - 8 for t in ts]].numpy()); h_.append(s[[t - 8 for t in ts]].numpy())
        S_own[k], Cw[k], Sch[k] = np.concatenate(s_)[:, WIN].sum(-1), np.concatenate(c_), np.concatenate(h_)
        del model; torch.cuda.empty_cache()
        print("checkpoint", k, flush=True)
    night = light[rws] < 0.5

    def cov(prob):
        out = {}
        for name in ("any_entity", "zombies", "cows", "changed_hud", "entering_map"):
            mk = cover[name][rws]; present = mk.any(-1); hit = mk[:, WIN].any(-1)
            out[name] = float((prob * hit).sum(-1)[present].mean())
        return out
    res = {}
    for j in (2, 3, 4, 5):
        pc = lambda c, s: C.policy(torch.as_tensor(c), torch.as_tensor(s), pol).numpy()
        r = {}
        for form, c in (("off", np.zeros_like(Cw[j])), ("deployed", Cw[j])):
            fresh, stale = pc(c, S_own[j]), pc(c, S_own[j - 1])
            r[form] = {"tv": float(0.5 * np.abs(fresh - stale).sum(-1).mean()), "agree": float((fresh.argmax(-1) == stale.argmax(-1)).mean()),
                       "fresh": cov(fresh), "stale_1": cov(stale)}
        r["chooser"] = cov(pc(Cw[j], Sch[j]))
        res[f"ckpt_{j:02d}"] = r
        print(j, json.dumps(r), flush=True)
    (C9.HERE / "c9_early.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
