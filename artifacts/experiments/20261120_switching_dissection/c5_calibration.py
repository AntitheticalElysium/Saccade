"""C5 (PLAN.md): the sampler's calibration at mob cells, and does Mamba's predicted belief know where the mobs are.

- **Units:** C4's S2 start mobs (frame t = t0 + 1, the real past), 8 draws each of the default sampler
  (`c4_sample.samples`, which also returns each ladder step's clean estimate). A draw keeps the mob if the reader
  finds its kind in the 3 x 3 neighbourhood of its cell after the real camera move (C4's rule).
- **Readings:** calibration against the reader on the real frame; concentration over units against a binomial; the
  reader's P(kind) along the ladder, deleted vs kept; the deleted cell's form.
- **Mamba probe:** linear, per map cell, entity presence at t from the predicted belief `bp` (what the denoiser reads
  through `memory_out`), from the reader's mob probabilities on the previous real frame (fitted, and unfitted as a
  copy), and from both; every DEV episode without units, 80 / 20 by episode. Then the units' deletion rate by the belief probe's P(entity in the neighbourhood).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c5_calibration [smoke] [ckpt N]

C6 (PLAN.md): `ckpt N` runs the draws and readings at P1.5's checkpoint N with its own S1 kind reader (no probe).
"""
import importlib, json, sys
from math import comb
from pathlib import Path

import numpy as np
import torch

from saccade.eval import s1 as S
from saccade.eval.probe import KEEP

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
C1 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c1")
C2 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c2_eye")
C3P = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c3_probe")
C4 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c4_sample")
DL = importlib.import_module("artifacts.experiments.20261120_switching_dissection.deletions")
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
HERE = Path(__file__).resolve().parent
CK = HERE.parent / "20261116_p15/run/ckpt_10.pt"
DEV = torch.device("cuda")
DRAWS = 8
KINDS = ("none", "zombies", "cows", "skeletons", "arrows")


@torch.no_grad()
def kind_probs(R, p, v):
    """The S1 reader's class probabilities per map cell: p [N, 81, 16], v [N, 8] -> [N, 63, classes] (S2.read's mask)."""
    x = torch.cat([p[:, :63], v[:, None].expand(-1, 63, -1)], -1).reshape(-1, p.shape[-1] + v.shape[-1]).half()
    pr = R["kind"](x)["kind"].float().softmax(-1).reshape(len(p), 63, -1)
    pr[:, ~torch.as_tensor(KEEP, device=pr.device)] = 0
    return pr


def reader(model, cfg, smoke):
    """S1's kind reader for this checkpoint's encoder, by S1's own protocol (train400 cells, token_probe seed 0)."""
    Ltr = S.first_episodes(dict(np.load(S.LABELS / "s0_train400_labels.npz")), 40 if smoke else None)
    Ldv = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 4)           # token_probe's argmax only
    tr, dv = S.cell_table(S.encode(model, Ltr, cfg), Ltr), S.cell_table(S.encode(model, Ldv, cfg), Ldv)
    return S.token_probe(tr, dv, 0)[0]


def fit_multi(Xtr, Ytr, Xte, epochs=30, lr=1e-2, wd=1e-4):
    """c3_probe.fit_probe with one logistic output per map cell (its own positive weight)."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-4
    Xtr, Xte = ((Xtr - mu) / sd).to(DEV), ((Xte - mu) / sd).to(DEV)
    yt = torch.as_tensor(Ytr, device=DEV, dtype=torch.float32)
    w = torch.zeros(Xtr.shape[1], yt.shape[1], device=DEV, requires_grad=True); b = torch.zeros(yt.shape[1], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    pos = yt.mean(0); pw = (1 - pos) / pos.clamp(min=1e-4)
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(epochs):
        for ix in torch.randperm(len(Xtr), generator=g).split(4096):
            ix = ix.to(DEV)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr[ix] @ w + b, yt[ix], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        w, b = w.detach(), b.detach()
        return lambda X: torch.sigmoid(((X - mu) / sd).to(DEV) @ w + b).cpu().numpy(), torch.sigmoid(Xte @ w + b).cpu().numpy()


def auc(pos, neg):
    s = np.r_[pos, neg]; r = s.argsort().argsort() + 1.
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / max(len(pos) * len(neg), 1))


def main(smoke=False, ck=10):
    model, cfg = AI.load(CK.parent / f"ckpt_{ck:02d}.pt")
    model.requires_grad_(False).eval()
    R = S2.load_readers(model, cfg) if ck == 10 else {"kind": reader(model, cfg, smoke)}
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    _, _, lookup, loff, cover = C1.semantics(L)
    dl = S.episodes(L)
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)                # C4's units, exactly
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    z = np.load(DL.EV.S2 / "model.npz"); mg, mp = z["m_grid"], z["m_pos"].astype(int)
    units = []
    for u in U:
        w, kind, i, j = u[:4]
        e, t0 = int(rows[w, 0]), int(rows[w, 3])
        if 9 <= t0 + 1 <= L["ep_T"][e]:
            r = DL.deleted(mg, mp, u)
            if r:
                units.append((w, e, t0 + 1, int(kind), int(i), int(j), np.mean([x[0] for x in r]), bool(u[7])))
    all_unit_eps = {x[1] for x in units}
    units = units[:60] if smoke else units
    # ---- the draws -------------------------------------------------------------------------------------------------
    rec, bp_u = [], []
    for e in sorted({x[1] for x in units}):
        st = C2.episode(model, cfg, dl, e)
        us = [x for x in units if x[1] == e]
        ts = sorted({x[2] for x in us})
        real = kind_probs(R, st[1].p[0, ts].float(), st[1].v[0, ts].float()).cpu()
        prev = kind_probs(R, st[1].p[0, [t - 1 for t in ts]].float(), st[1].v[0, [t - 1 for t in ts]].float()).cpu()
        dr = []                                                             # [draws, steps + final, W, 63, classes]
        for d in range(DRAWS):
            tr = []
            pd, vd = C4.samples(model, cfg, st, e, ts, d, tr)
            dr.append(torch.stack([kind_probs(R, a, v).cpu() for a, v in tr + [(pd, vd)]]))
        dr = torch.stack(dr)
        for w, _, t, kind, i, j, rate, night in us:
            qi = ts.index(t)
            dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
            r0, c0 = i - dpos[0], j - dpos[1]
            cells = [r * 9 + c for r in range(r0 - 1, r0 + 2) for c in range(c0 - 1, c0 + 2) if 0 <= r < 7 and 0 <= c < 9]
            if not cells:
                continue
            own = r0 * 9 + c0 if 0 <= r0 < 7 and 0 <= c0 < 9 else cells[len(cells) // 2]
            rec.append({"e": e, "t": t, "kind": kind, "night": night, "s2_rate": rate,
                        "real_P": float(real[qi, cells, kind].max()), "real_kept": bool((real[qi, cells].argmax(-1) == kind).any()),
                        "real_own_P": float(real[qi, own, kind]),
                        "P": dr[:, :, qi, cells, kind].amax(-1).numpy(),               # [draws, steps + final]
                        "kept": (dr[:, -1, qi][:, cells].argmax(-1) == kind).any(-1).numpy(),
                        "own_class": dr[:, -1, qi, own].argmax(-1).numpy()})
            bp_u.append((st[3][0, t].float().cpu(), prev[qi, :, 1:].flatten(), cells))
        print(e, len(rec), flush=True)
    kept = np.stack([x["kept"] for x in rec]); P = np.stack([x["P"] for x in rec])  # [units, draws], [units, draws, 5]
    own = np.stack([x["own_class"] for x in rec])
    kinds = np.array([x["kind"] for x in rec]); night = np.array([x["night"] for x in rec])
    res = {"units": len(rec), "draws": DRAWS, "calibration": {}, "concentration": {}, "ladder": {}, "deleted_form": {}}
    for kn in range(1, 5):
        for tag, sel in (("day", ~night), ("night", night), ("all", np.ones_like(night))):
            m = (kinds == kn) & sel
            if not m.any():
                continue
            key = f"{KINDS[kn]}|{tag}"
            res["calibration"][key] = {"units": int(m.sum()), "real_frame_kept": float(np.mean([rec[i]["real_kept"] for i in np.nonzero(m)[0]])),
                                       "world_kept": float(kept[m].mean()), "s2_step1_deleted": float(np.mean([rec[i]["s2_rate"] for i in np.nonzero(m)[0]]))}
            if tag == "all":
                continue
            n_del = (~kept[m]).sum(1); q = n_del.mean() / DRAWS
            binom = np.array([comb(DRAWS, k) * q ** k * (1 - q) ** (DRAWS - k) for k in range(DRAWS + 1)])
            res["concentration"][key] = {"units": int(m.sum()), "pooled_deletion": float(q),
                                         "dispersion_index": float(n_del.var() / max(DRAWS * q * (1 - q), 1e-9)),
                                         "never_deleted": [float((n_del == 0).mean()), float(binom[0])],
                                         "deleted_4_of_8_or_more": [float((n_del >= 4).mean()), float(binom[4:].sum())],
                                         "share_of_deletions_in_those_units": float(n_del[n_del >= 4].sum() / max(n_del.sum(), 1)),
                                         "histogram": np.bincount(n_del, minlength=DRAWS + 1).tolist()}
    for name, m in (("all", np.ones(len(rec), bool)), ("cows", kinds == 2), ("zombies", kinds == 1), ("night", night), ("day", ~night)):
        Pm, km = P[m].reshape(-1, 5), kept[m].reshape(-1)
        res["ladder"][name] = {"draws_deleted": int((~km).sum()), "draws_kept": int(km.sum()),
                               "mean_P_kind_by_step_deleted": Pm[~km].mean(0).round(4).tolist(),
                               "mean_P_kind_by_step_kept": Pm[km].mean(0).round(4).tolist(),
                               "auc_step_predicts_deletion": [auc(1 - Pm[~km, s], 1 - Pm[km, s]) for s in range(5)]}
        fin = Pm[~km, -1]
        res["deleted_form"][name] = {"final_P_kind_max_over_neighbourhood_deleted": {"<0.05": float((fin < .05).mean()), "0.05-0.2": float(((fin >= .05) & (fin < .2)).mean()),
                                                                                     ">=0.2": float((fin >= .2).mean())} if len(fin) else None,
                                     "own_cell_class_deleted": dict(zip(KINDS, np.bincount(own[m].reshape(-1)[~km], minlength=5).tolist())),
                                     "real_frame_P_kind": float(np.mean([rec[i]["real_P"] for i in np.nonzero(m)[0]]))}
    tag = ("_smoke" if smoke else "") + ("" if ck == 10 else f"_ckpt{ck:02d}")
    if ck != 10:                                                            # C6: no probe
        print(json.dumps(res, indent=1), flush=True)
        (HERE / f"c5_calibration{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
        return
    # ---- Mamba probe -----------------------------------------------------------------------------------------------
    rng = np.random.default_rng(20261121)                                   # C3's order; every DEV episode without units
    eps = [e for e in rng.permutation(len(L["ep_T"])) if L["ep_T"][e] >= 10 and e not in all_unit_eps][:24 if smoke else None]
    data = {}
    for e in eps:
        st = C2.episode(model, cfg, dl, e)
        T = int(L["ep_T"][e])
        keep = np.sort(np.random.default_rng(int(e)).choice(np.arange(1, T + 1), min(120, T), replace=False))
        rr = lookup[loff[e] + keep]
        pk = kind_probs(R, st[1].p[0, keep - 1].float(), st[1].v[0, keep - 1].float())[:, :, 1:].cpu()   # the reader on t - 1
        data[e] = (st[3][0, keep].float().cpu(), pk.flatten(1),
                   cover["any_entity"][rr][:, :63], cover["cows"][rr][:, :63])
    cut = int(0.8 * len(eps)); fit, held = eps[:cut], eps[cut:]
    cat = lambda es, k: torch.cat([data[e][k] for e in es]) if k < 2 else np.concatenate([data[e][k] for e in es])
    ub, up = torch.stack([x[0] for x in bp_u]), torch.stack([x[1] for x in bp_u])
    res["mamba_probe"] = {"fit_episodes": len(fit), "held_episodes": len(held)}
    for tname, k in (("any_entity", 2), ("cows", 3)):
        ytr, yte = cat(fit, k), cat(held, k)
        out = {"base_rate": float(yte.mean()), "positives_fit": int(ytr.sum())}
        pk = cat(held, 1).reshape(len(yte), 63, 4)                          # the unfitted copy: the reader's P at t - 1
        out["previous_frame_copy_unfitted"] = C3P.average_precision(yte.reshape(-1).astype(float), (pk.sum(-1) if k == 2 else pk[..., 1]).reshape(-1).numpy())
        for fname, Xtr, Xte, Xu in (("belief", cat(fit, 0), cat(held, 0), ub), ("previous_frame", cat(fit, 1), cat(held, 1), up),
                                    ("both", torch.cat([cat(fit, 0), cat(fit, 1)], 1), torch.cat([cat(held, 0), cat(held, 1)], 1), torch.cat([ub, up], 1))):
            f, pte = fit_multi(Xtr, ytr, Xte)
            out[fname] = C3P.average_precision(yte.reshape(-1).astype(float), pte.reshape(-1))
            if tname == "any_entity" and fname in ("belief", "previous_frame"):
                pu = f(Xu); conf = np.array([pu[n, x[2]].max() for n, x in enumerate(bp_u)])
                d = (~kept).mean(1); qs = np.quantile(conf, [0.25, 0.5, 0.75]); bins = np.digitize(conf, qs)
                res["mamba_probe"][f"units_deletion_by_{fname}_probe_quartile"] = {
                    "quartile_upper_P": qs.round(3).tolist() + [1.0], "deletion": [float(d[bins == q].mean()) for q in range(4)],
                    "corr": float(np.corrcoef(conf, d)[0, 1])}
        res["mamba_probe"][tname] = out
        print(tname, json.dumps(out), flush=True)
    print(json.dumps({k: v for k, v in res.items()}, indent=1), flush=True)
    np.savez_compressed(HERE / f"c5_units{tag}.npz", P=P, kept=kept, kinds=kinds, night=night,
                        e=np.array([x["e"] for x in rec]), t=np.array([x["t"] for x in rec]),
                        real_P=np.array([x["real_P"] for x in rec]), real_own_P=np.array([x["real_own_P"] for x in rec]))
    (HERE / f"c5_calibration{tag}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main("smoke" in a, int(a[a.index("ckpt") + 1]) if "ckpt" in a else 10)
