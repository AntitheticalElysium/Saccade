"""C3's readability probe (RESULTS.md, C3): can a linear read of P1.5's record through frame t - 1, at a HUD cell,
anticipate whether that cell's digit flips at frame t?
- **Units:** C3's DEV episodes and fit / held-out split, 120 frames each, P1.5's coarse tokens.
- **Cells and target:** food, drink and energy (cells 64 / 65 / 66); the label's value changes from t - 1 to t.
- **Feature sets, per cell:** (a) the record (1,024); (b) the chooser's own view, the last 8 coarse tokens (128);
  (c) both.
- **Fit:** logistic regression, L2, the same optimiser and epochs for every set.
- **Reading:** held-out average precision against the base rate.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c3_probe
"""
import importlib, json
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, train as TR
from saccade.eval import s1 as S

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
HERE = Path(__file__).resolve().parent
CK = HERE.parent / "20261116_p15/run/ckpt_10.pt"
DEV = torch.device("cuda")
CELLS = {"food": (64, 1), "drink": (65, 2), "energy": (66, 3)}              # periphery cell, s_body column


def average_precision(y, s):
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


@torch.no_grad()
def episode(model, cfg, dl, L, loff, e):
    T = int(dl[e].T)
    chunk = D.gather([dl[e]], [[(0, 1, t, 4, 4) for t in range(T + 1)]])
    with TR.autocast(DEV, cfg):
        p, _ = C.coarse(model.perception, torch.as_tensor(chunk["frames"], device=DEV).flatten(0, 1))
    p = p.reshape(1, T + 1, 81, 16).float()
    keep = np.sort(np.random.default_rng(int(e)).choice(np.arange(8, T + 1), min(120, T - 7), replace=False))
    state, rec = model.record.zero(1, DEV), []
    for a in range(0, T + 1, 512):
        first = torch.zeros(1, min(T + 1, a + 512) - a, dtype=torch.bool, device=DEV); first[:, 0] = a == 0
        ri, si = model.record.scan(p[:, a:a + 512], first, state); state = si[:, -1]
        rec.append(ri[0][:, [c for c, _ in CELLS.values()]].half().cpu()); del ri, si
    rec = torch.cat(rec)[keep]                                              # [n, 3, 1024]: the record through t - 1
    hist = torch.stack([p[0, keep - k][:, [c for c, _ in CELLS.values()]] for k in range(1, 9)], 2).flatten(2).half().cpu()   # [n, 3, 128]
    body = L["s_body"][loff[e]:loff[e] + T + 1]
    y = np.stack([body[keep, col] != body[keep - 1, col] for _, col in CELLS.values()], 1)   # [n, 3]
    return rec, hist, y


def fit_probe(Xtr, ytr, Xte, epochs=30, lr=1e-2, wd=1e-4):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-4
    Xtr, Xte = ((Xtr - mu) / sd).to(DEV), ((Xte - mu) / sd).to(DEV)
    w = torch.zeros(Xtr.shape[1], device=DEV, requires_grad=True); b = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    yt = torch.as_tensor(ytr, device=DEV, dtype=torch.float32)
    pos = float(yt.mean()); pw = torch.tensor((1 - pos) / max(pos, 1e-6), device=DEV)
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(epochs):
        for ix in torch.randperm(len(Xtr), generator=g).split(4096):
            ix = ix.to(DEV)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr[ix] @ w + b, yt[ix], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (Xte @ w + b).cpu().numpy()


def main():
    model, cfg = AI.load(CK)
    model.requires_grad_(False).eval()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    dl = S.episodes(L)
    rng = np.random.default_rng(20261121)                                   # C3's split
    eps = [e for e in rng.permutation(len(L["ep_T"])) if L["ep_T"][e] >= 10][:400]
    fit, held = eps[:320], eps[320:]
    data = {e: episode(model, cfg, dl, L, loff, e) for e in eps}
    cat = lambda es, i: torch.cat([data[e][i] for e in es]) if i < 2 else np.concatenate([data[e][i] for e in es])
    res = {}
    for ci, name in enumerate(CELLS):
        ytr, yte = cat(fit, 2)[:, ci], cat(held, 2)[:, ci]
        sets = {"record": (cat(fit, 0)[:, ci].float(), cat(held, 0)[:, ci].float()),
                "last_8_frames": (cat(fit, 1)[:, ci].float(), cat(held, 1)[:, ci].float())}
        sets["both"] = (torch.cat([sets["record"][0], sets["last_8_frames"][0]], 1), torch.cat([sets["record"][1], sets["last_8_frames"][1]], 1))
        res[name] = {"base_rate": float(yte.mean()), "flips_held_out": int(yte.sum()), "flips_fit": int(ytr.sum())}
        for k, (a, b) in sets.items():
            res[name][k] = average_precision(yte.astype(float), fit_probe(a, ytr, b))
        print(name, json.dumps(res[name]), flush=True)
    (HERE / "c3_probe.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
