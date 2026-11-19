"""Diagnostic (11-19, after the model stage's prefix assert fired at RMS 0.036 > 0.03 on one start): the prefix belief
rebuilt per start against encode's, for every bank start, beside the prefix length; and an FP32 rebuild of the
worst starts, to see which bf16 computation the FP32 one sides with. No rollout."""
import os
os.environ.setdefault("MAMBA_DETERMINISTIC", "1")
import importlib, json
import numpy as np, torch
from saccade import model as M, train as TR
from saccade.eval import s1 as S
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
DEV = S.DEV


@torch.no_grad()
def prefix(model, cfg, en, t, amp=True):
    ctx = TR.autocast(DEV, cfg) if amp else torch.autocast("cuda", enabled=False)
    with ctx:
        pp = M.Percept(*(en[k][:t + 1].to(DEV).float()[None] for k in ("p", "g", "v")), en["fix"][:t + 1].to(DEV)[None])
        reads, _, _ = S.history(model, pp.p)
        y, _ = model.belief(model.belief.tokens(S.embed(model, pp, reads), en["reward"][:t + 1].to(DEV)[None],
                            en["terminated"][:t + 1].to(DEV)[None], en["action"][:t + 1].to(DEV)[None]), model.belief.initial(1, DEV))
    return y[0, -1].float()


def main():
    model, cfg = S2.load()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    rows, _ = S2.bank()
    eps = np.unique(rows[:, 0]); sub = {e: i for i, e in enumerate(eps)}
    enc = S.encode(model, S2.sublabels(L, eps), cfg)
    err = np.array([float((prefix(model, cfg, enc[sub[e]], int(t)) - enc[sub[e]]["bp"][int(t) + 1].to(DEV).float()).square().mean().sqrt())
                    for e, _, _, t, _, _ in rows])
    t0 = rows[:, 3]
    out = {"max": float(err.max()), "mean": float(err.mean()), "over_0.03": int((err > 0.03).sum()),
           "quantiles": np.quantile(err, [.5, .9, .99, .999]).round(5).tolist(),
           "by_t0_bin": {f"{lo}-{hi}": [float(err[(t0 >= lo) & (t0 < hi)].mean()), float(err[(t0 >= lo) & (t0 < hi)].max())] if ((t0 >= lo) & (t0 < hi)).any() else None
                         for lo, hi in ((0, 64), (64, 256), (256, 512), (512, 1024), (1024, 3000))}}
    worst = np.argsort(-err)[:5]
    out["worst"] = []
    for w in worst:
        e, t = int(rows[w, 0]), int(rows[w, 3]); en = enc[sub[e]]
        f32 = prefix(model, cfg, en, t, amp=False)
        out["worst"].append({"start": int(w), "t0": t, "bf16_rebuild_vs_encode": float(err[w]),
                             "fp32_vs_rebuild": float((f32 - prefix(model, cfg, en, t)).square().mean().sqrt()),
                             "fp32_vs_encode": float((f32 - en["bp"][t + 1].to(DEV).float()).square().mean().sqrt())})
    (S2.HERE / "prefix_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
