"""Where does a look help the next decision? PLAN.md (2026-11-23).

The demonstrator's action at frame t from (predicted belief, periphery, lighting, record read, previous action) plus
a crop block (glimpse tokens at a window, the window, a condition flag: real / blurred / none). The blurred crop is
the same window rendered at the periphery's acuity (each cell pooled to 3x3, upsampled), through the same fovea
encoder: real minus blurred is the value of acuity. Scored on DEV at all 49 windows, beside the old chooser's P(window).

    JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_gaze_decision_value.decision_value [--smoke]
"""
import argparse
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import gates as G, s1 as S

HERE = Path(__file__).resolve().parent
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
AUD = importlib.import_module("artifacts.experiments.20261119_s2.dream_outcome_audit")
DEV = S2.DEV
SEED = 20261122                                                   # the dream-commitment split
FIRST = 8                                                         # decisions with a full context
DIRS = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])       # constants.DIRECTIONS[:5] (s2b)
WIN = C.WIN.numpy()                                               # [49, 9] cell indices per window, row-major
HUD_WIN = (C.CENTRES[:, 0] >= 6).numpy()                          # windows covering a HUD row (7-8)


def frames_of(ep):
    return torch.as_tensor(D.gather([ep], [[(0, 1, t, 4, 4) for t in range(ep.T + 1)]])["frames"][0])


def blurred(frames, acuity):
    """Each 7x7 cell pooled to the periphery's acuity (rounded, as Perception does) and upsampled back (nearest)."""
    f = frames.reshape(-1, 9, 7, 9, 7, 3).permute(0, 1, 3, 5, 2, 4).reshape(-1, 3, 7, 7).float()
    f = F.interpolate(torch.round(F.adaptive_avg_pool2d(f, acuity)), size=(7, 7), mode="nearest")
    return f.reshape(-1, 9, 9, 3, 7, 7).permute(0, 1, 4, 2, 5, 3).reshape(-1, 63, 63, 3).round().clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def glimpses(model, cfg, frames, fix):
    """uint8 frames [n, 63, 63, 3], window centres [n, 2] -> glimpse tokens [n, 9, Cg] (fp16, on the GPU)."""
    out = []
    for a in range(0, len(frames), 1024):
        with TR.autocast(DEV, cfg):
            P = model.perception(frames[a:a + 1024].to(DEV)[None], fix[a:a + 1024].to(DEV)[None])
        out.append(P.g[0].half())
    return torch.cat(out)


def coarse_block(en, ts):
    t = torch.as_tensor(ts)
    return torch.cat((en["bp"][t].half(), en["p"][t].flatten(1).half(), en["v"][t].half(), en["reads"][t].half(),
                      F.one_hot(en["prev_action"][t], 17).half()), -1)


def fine_block(g, w):
    return torch.cat((g.flatten(1).float(), F.one_hot(torch.as_tensor(w, device=g.device), 49).float()), -1)


@torch.no_grad()
def chooser_probs(head, pol, en, ts):
    """The checkpoint's fixation controller on real frames: P(window) [n, 49] at each decision frame (s2b.choose)."""
    H, ts = pol["history"], torch.as_tensor(ts)
    idx = ts[:, None] + torch.arange(-H, 0)[None]
    ok = (idx >= 0).float().to(DEV)
    idx = idx.clamp(min=0)
    get = lambda k: en[k][idx].to(DEV)
    ctx = torch.cat([get("v").float() * ok[..., None],
                     F.one_hot((get("prev_action") + 1).clamp(0, 17), 18).float() * ok[..., None],
                     (M.SYMLOG(get("reward").float()) * ok)[..., None], (get("terminated").float() * ok)[..., None],
                     ok[..., None]], -1).flatten(1)
    ph = get("p").float() * ok[..., None, None]
    pa = en["prev_action"][ts].to(DEV)
    win = torch.as_tensor(WIN, device=DEV)
    taken, noop = head((ph, ctx, pa)), head((ph, ctx, torch.zeros_like(pa)))
    c_ = (taken - noop).square().sum(-1)[:, win].sum(-1)
    s_ = (en["p"][ts].to(DEV).float() - taken).square().sum(-1)[:, win].sum(-1)
    return C.policy(c_, s_, pol).cpu()


def decisions(en, L, so, ao, e):
    T = len(en["reward"]) - 1
    ts = np.arange(FIRST, T)
    keep = ~L["t_exogenous"][ao[e] + ts].astype(bool)
    return ts[keep]


class Probe(nn.Module):
    def __init__(self, dc, df):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dc + df + 3, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 17))

    def forward(self, c, f, flag):
        return self.net(torch.cat((c, f, F.one_hot(flag, 3).float()), -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("highest")
    model, cfg = S2.AI.load(S2.P15)
    model.requires_grad_(False).eval()
    pol = dict(cfg["fixation"]["chooser"])
    head = C.Head(pol["history"]).to(DEV)
    head.load_state_dict(torch.load(S2.P15, weights_only=False, map_location="cpu")["fixation_controller"]["head"])
    head.eval().requires_grad_(False)
    acuity = model.perception.acuity
    rng = np.random.default_rng(SEED + 23)
    data = {}
    for split, n, seed in (("train400", 6 if args.smoke else 128, SEED), ("dev", 6 if args.smoke else 160, SEED + 1)):
        enc, ids = AUD.encode_sample(model, cfg, split, n, seed)
        L = dict(np.load(S.LABELS / f"s0_{split}_labels.npz"))
        sub = S2.sublabels(L, np.asarray(ids))
        data[split] = (enc, ids, L, sub, S.episodes(sub), G.episode_grids(sub))
        print("encoded", split, len(enc), flush=True)

    # ---- training rows: per decision, two real crops, one blurred, one none ----
    enc, ids, L, sub, eps, _ = data["train400"]
    so, ao = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    Cb, Fb, flags, dec, acts = [], [], [], [], []
    base = 0
    for i, e in enumerate(ids):
        ts = decisions(enc[i], L, so, ao, e)
        if not len(ts):
            continue
        fr = frames_of(eps[i])[ts]
        Cb.append(coarse_block(enc[i], ts))
        acts.append(enc[i]["action"][torch.as_tensor(ts)])
        for flag in (0, 0, 1):
            w = rng.integers(49, size=len(ts))
            g = glimpses(model, cfg, fr if flag == 0 else blurred(fr, acuity), C.CENTRES[w])
            Fb.append(fine_block(g, w).half()); flags.append(torch.full((len(ts),), flag)); dec.append(base + torch.arange(len(ts)))
        Fb.append(torch.zeros_like(Fb[-1])); flags.append(torch.full((len(ts),), 2)); dec.append(base + torch.arange(len(ts)))
        base += len(ts)
    Cb, Fb, flags, dec, acts = torch.cat(Cb).to(DEV), torch.cat(Fb).to(DEV), torch.cat(flags).to(DEV), torch.cat(dec).to(DEV), torch.cat(acts).to(DEV)
    cm, cs = Cb.float().mean(0), Cb.float().std(0).clamp_min(1e-3)
    seen = flags < 2
    fm, fs = Fb[seen].float().mean(0), Fb[seen].float().std(0).clamp_min(1e-3)
    normc = lambda x: (x.float() - cm) / cs
    normf = lambda x, fl: torch.where((fl < 2)[:, None], (x.float() - fm) / fs, torch.zeros_like(x, dtype=torch.float32))
    print("training rows", len(Fb), "decisions", len(Cb), flush=True)
    torch.manual_seed(SEED)
    probe = Probe(Cb.shape[1], Fb.shape[1]).to(DEV)
    opt = torch.optim.AdamW(probe.parameters(), lr=7e-4)
    gen = torch.Generator(device=DEV).manual_seed(SEED + 1)
    for epoch in range(2 if args.smoke else 16):
        tot = 0.0
        for ix in torch.randperm(len(Fb), generator=gen, device=DEV).split(1024):
            loss = F.cross_entropy(probe(normc(Cb[dec[ix]]), normf(Fb[ix], flags[ix]), flags[ix]), acts[dec[ix]])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss) * len(ix)
        print(f"epoch {epoch} train CE {tot / len(Fb):.4f}", flush=True)
    probe.eval()

    # ---- DEV: every window, real and blurred, and no crop ----
    enc, ids, L, sub, eps, grids = data["dev"]
    so, ao = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    out = {k: [] for k in ("episode", "lp_none", "lp_real", "lp_blur", "acc_none", "acc_real", "acc_blur", "chooser",
                           "night", "creature", "cwin", "fwin")}
    with torch.no_grad():
        for i, e in enumerate(ids):
            ts = decisions(enc[i], L, so, ao, e)
            if not len(ts):
                continue
            n = len(ts)
            fr = frames_of(eps[i])[ts]
            frb = blurred(fr, acuity)
            c = normc(coarse_block(enc[i], ts).to(DEV))
            a = enc[i]["action"][torch.as_tensor(ts)].to(DEV)
            none = probe(c, torch.zeros(n, Fb.shape[1], device=DEV), torch.full((n,), 2, device=DEV)).log_softmax(-1)
            res = {}
            for name, src, flag in (("real", fr, 0), ("blur", frb, 1)):
                lp, acc = [], []
                for w in range(49):
                    g = glimpses(model, cfg, src, C.CENTRES[w].expand(n, 2))
                    fl = torch.full((n,), flag, device=DEV)
                    o = probe(c, normf(fine_block(g, np.full(n, w)), fl), fl).log_softmax(-1)
                    lp.append(o.gather(1, a[:, None])[:, 0]); acc.append(o.argmax(-1) == a)
                res[name] = (torch.stack(lp, 1).cpu(), torch.stack(acc, 1).cpu())
            out["episode"].append(np.full(n, i)); out["lp_none"].append(none.gather(1, a[:, None])[:, 0].cpu().numpy())
            out["acc_none"].append((none.argmax(-1) == a).cpu().numpy())
            for name in ("real", "blur"):
                out[f"lp_{name}"].append(res[name][0].numpy()); out[f"acc_{name}"].append(res[name][1].numpy())
            out["chooser"].append(chooser_probs(head, pol, enc[i], ts).numpy())
            out["night"].append(L["s_light"][so[e] + ts] < 0.5)
            gr = grids[i]["grid"][ts]                                   # [n, 7, 9] kinds
            occupied = np.concatenate([gr.reshape(n, 63) > 0, np.zeros((n, 18), bool)], 1)   # HUD cells never a creature
            out["creature"].append(occupied.any(1))
            out["cwin"].append(occupied[:, WIN].any(-1))
            d = DIRS[L["s_dir"][so[e] + ts].astype(int)]
            faced = (3 + d[:, 0]) * 9 + (4 + d[:, 1])
            out["fwin"].append((WIN[None] == faced[:, None, None]).any(-1))
            if i % 20 == 0:
                print("dev episode", i, flush=True)
    z = {k: np.concatenate(v) for k, v in out.items()}
    np.savez_compressed(HERE / ("scores_smoke.npz" if args.smoke else "scores.npz"), **z)
    (HERE / ("summary_smoke.json" if args.smoke else "summary.json")).write_text(json.dumps(summarise(z), indent=1) + "\n")
    print(json.dumps(summarise(z), indent=1), flush=True)


def summarise(z, reps=1000):
    ep = z["episode"]
    vr, vb = z["lp_real"] - z["lp_none"][:, None], z["lp_blur"] - z["lp_none"][:, None]
    acu = z["lp_real"] - z["lp_blur"]
    per = {"no_crop_ce": -z["lp_none"],
           "uniform_real_minus_none": vr.mean(1), "uniform_blur_minus_none": vb.mean(1), "uniform_acuity": acu.mean(1),
           "best_real_minus_none": vr.max(1), "best_blur_minus_none": vb.max(1),
           "chooser_real_minus_none": (z["chooser"] * vr).sum(1), "chooser_acuity": (z["chooser"] * acu).sum(1),
           "acc_none": z["acc_none"].astype(float), "acc_uniform_real": z["acc_real"].mean(1),
           "acc_uniform_blur": z["acc_blur"].mean(1)}
    rng = np.random.default_rng(0)
    u = np.unique(ep)
    idx = {k: np.flatnonzero(ep == k) for k in u}

    def boot(v, mask=None):
        m = np.ones(len(v), bool) if mask is None else mask
        point = float(v[m].mean()) if m.any() else None
        bs = []
        for _ in range(reps):
            ii = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
            ii = ii[m[ii]]
            if len(ii):
                bs.append(v[ii].mean())
        return {"mean": point, "ci95": np.quantile(bs, [0.025, 0.975]).tolist() if bs else None, "n": int(m.sum())}

    res = {"decisions": int(len(ep)), "episodes": int(len(u)), "overall": {k: boot(v) for k, v in per.items()}}
    for name, m in (("night", z["night"]), ("day", ~z["night"]), ("creature_in_view", z["creature"]),
                    ("no_creature", ~z["creature"])):
        res[name] = {k: boot(per[k], m) for k in ("uniform_acuity", "best_real_minus_none", "best_blur_minus_none",
                                                   "chooser_acuity", "uniform_real_minus_none")}
    cats = {"creature_window": z["cwin"], "faced_window": z["fwin"], "hud_window": np.broadcast_to(HUD_WIN, acu.shape)}
    res["by_window"] = {}
    for name, m in cats.items():
        res["by_window"][name] = {"inside": float(acu[m].mean()) if m.any() else None,
                                  "outside": float(acu[~m].mean()) if (~m).any() else None,
                                  "share_of_pairs": float(m.mean())}
    best = acu.argmax(1)
    rows = np.arange(len(best))
    res["acuity_best_window"] = {k: {"share": float(m[rows, best].mean()), "chance": float(m.mean())} for k, m in cats.items()}
    res["chooser_mass_on"] = {k: float((z["chooser"] * m).sum(1).mean()) for k, m in cats.items()}
    return res


if __name__ == "__main__":
    main()
