"""C3 of the eye's checks (PLAN.md): the cheap memory read.

P1.5's final chooser head is fine-tuned offline on DEV frames (P1.5's frozen coarse tokens) in two matched arms:
- **record:** the same head plus a per-cell read of P1.5's record through the previous frame (Linear 1,024 -> 64,
  zero-initialised, added to each cell's embedding, as Mamba's `record_in` reads it);
- **plain:** the head alone.

Both arms share the seed, batches and schedule (the chooser's own recipe: AdamW 1e-3 cosine to 1e-4, batch 256, clip
1). Reading, on held-out DEV episodes: next-token error and surprise by region, and policy coverage. Regions are C1's
groups and cover masks.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c3 [UPDATES]
"""
import importlib, json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from saccade import chooser as C, data as D, train as TR
from saccade.eval import s1 as S

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
C1 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c1")
HERE = Path(__file__).resolve().parent
CK = HERE.parent / "20261116_p15/run/ckpt_10.pt"
DEV = torch.device("cuda")
WIN = C.WIN.numpy()


class RecordHead(nn.Module):
    """chooser.Head with an optional per-cell record read added to the cell embeddings."""

    def __init__(self, base, record_dim=None):
        super().__init__()
        self.base = base
        self.record = None if record_dim is None else nn.Linear(record_dim, 64)
        if self.record is not None:
            nn.init.zeros_(self.record.weight), nn.init.zeros_(self.record.bias)

    def forward(self, inputs, rec=None):
        p, ctx, act = inputs
        b = self.base
        x = b.local(p.transpose(1, 2).flatten(2))
        x = x + b.context(ctx)[:, None] + b.action(act)[:, None] + b.position
        if self.record is not None:
            x = x + self.record(rec.float())
        return p[:, -1] + b.output(b.blocks(x))


@torch.no_grad()
def episode_data(model, cfg, dl, e):
    """Coarse tokens / chooser inputs / targets for frames 8..T and the record through each previous frame (bf16)."""
    T = int(dl[e].T)
    chunk = D.gather([dl[e]], [[(0, 1, t, 4, 4) for t in range(T + 1)]])
    fr = torch.as_tensor(chunk["frames"], device=DEV)
    with TR.autocast(DEV, cfg):
        p, v = C.coarse(model.perception, fr.flatten(0, 1))
    p, v = p.reshape(1, T + 1, 81, 16).float(), v.reshape(1, T + 1, 8).float()
    xx, y = C.inputs(p, v, chunk, 8, 8)
    first = torch.zeros(1, T + 1, dtype=torch.bool, device=DEV); first[:, 0] = True
    rin, _ = model.record.scan(p, first, model.record.zero(1, DEV))              # the record built from coarse tokens
    return xx, y, rin[0, 8:].to(torch.bfloat16)


def main(updates=3000):
    model, cfg = AI.load(CK)
    model.requires_grad_(False).eval()
    pol = dict(cfg["fixation"]["chooser"])
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    groups, light, lookup, loff, cover = C1.semantics(L)
    dl = S.episodes(L)
    rng = np.random.default_rng(20261121)
    eps = [e for e in rng.permutation(len(L["ep_T"])) if L["ep_T"][e] >= 10][:400]
    fit, held = eps[:320], eps[320:]
    data = {e: episode_data(model, cfg, dl, e) for e in eps}
    print("episodes", len(fit), "fit /", len(held), "held out", flush=True)
    state = torch.load(CK, weights_only=False, map_location="cpu")["fixation_controller"]["head"]
    arms = {}
    for name, rd in (("plain", None), ("record", model.record.dim)):
        torch.manual_seed(611601)
        base = C.Head(pol["history"]); base.load_state_dict(state)
        arms[name] = RecordHead(base, rd).to(DEV)
    opts = {n: torch.optim.AdamW(h.parameters(), lr=pol["learning_rate"], weight_decay=pol["weight_decay"]) for n, h in arms.items()}
    sizes = np.array([len(data[e][1]) for e in fit]); prob_e = sizes / sizes.sum()
    g = np.random.default_rng(611602)
    log = []
    for u in range(updates):
        lr = pol["minimum_learning_rate"] + 0.5 * (pol["learning_rate"] - pol["minimum_learning_rate"]) * (1 + math.cos(math.pi * u / updates))
        es = g.choice(len(fit), 8, p=prob_e)
        batch = []
        for ei in es:
            xx, y, rin = data[fit[ei]]
            ix = torch.as_tensor(g.integers(0, len(y), 32), device=DEV)
            batch.append((tuple(z[ix] for z in xx), y[ix], rin[ix]))
        xx = tuple(torch.cat([b[0][i] for b in batch]) for i in range(3)); y = torch.cat([b[1] for b in batch]); rin = torch.cat([b[2] for b in batch])
        row = {"update": u}
        for n, h in arms.items():
            for gr in opts[n].param_groups:
                gr["lr"] = lr
            loss = (h(xx, rin) - y).square().mean()
            opts[n].zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(h.parameters(), pol["grad_clip"]); opts[n].step()
            row[n] = float(loss)
        if u % 500 == 0 or u == updates - 1:
            print(row, flush=True); log.append(row)
    # held-out reading
    res = {"updates": updates, "log": log}
    for n, h in arms.items():
        h.eval()
        acc = {gname: [0., 0., 0.] for gname in C1.GROUPS}                     # surprise, squared error, cells
        cov = {f"{k}|{tag}": [0., 0., 0.] for k in cover for tag in ("all", "day", "night")}
        tot = 0.
        with torch.no_grad():
            for e in held:
                xx, y, rin = data[e]
                taken = h(xx, rin); noop = h((xx[0], xx[1], torch.zeros_like(xx[2])), rin)
                s = (y - taken).square().sum(-1)
                prob = C.policy((taken - noop).square().sum(-1)[:, C.WIN.to(DEV)].sum(-1), s[:, C.WIN.to(DEV)].sum(-1), pol).cpu().numpy()
                s = s.cpu().numpy(); tot += float(s.sum())
                rows = lookup[loff[e] + np.arange(8, len(y) + 8)]
                for gname in C1.GROUPS:
                    mk = groups[gname][rows]; acc[gname][0] += float((s * mk).sum()); acc[gname][2] += float(mk.sum())
                night = light[rows] < 0.5
                for k, mask in cover.items():
                    mk = mask[rows]; present = mk.any(-1); hit = mk[:, WIN].any(-1)
                    lw, uw = (prob * hit).sum(-1), hit.mean(-1)
                    for tag, sel in (("all", present), ("day", present & ~night), ("night", present & night)):
                        cov[f"{k}|{tag}"][0] += float(lw[sel].sum()); cov[f"{k}|{tag}"][1] += float(uw[sel].sum()); cov[f"{k}|{tag}"][2] += float(sel.sum())
        res[n] = {"surprise_share": {gname: v[0] / tot for gname, v in acc.items()},
                  "surprise_per_cell": {gname: v[0] / max(v[2], 1) for gname, v in acc.items()},
                  "coverage": {k: [v[0] / max(v[2], 1), v[1] / max(v[2], 1), v[2]] for k, v in cov.items()}}
    print("held-out surprise per cell (share) by region")
    for gname in C1.GROUPS:
        print(f"  {gname:13s}", " | ".join(f"{n}: {res[n]['surprise_per_cell'][gname]:.4f} ({res[n]['surprise_share'][gname]:.3f})" for n in arms))
    print("held-out coverage when present: learned (uniform)")
    for k in res["plain"]["coverage"]:
        print(f"  {k:20s}", " | ".join(f"{n}: {res[n]['coverage'][k][0]:.3f} ({res[n]['coverage'][k][1]:.3f})" for n in arms))
    (HERE / "c3.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
