"""Stage B (2) of the switching dissection (PLAN.md): how much the denoiser relies on its record, by region, in P1.5 and
the switching rerun at 4,000 updates.

- **Windows:** the P1 monitor's held-out windows (TR._monitor's picks, fixations and noise), plus target frames at
  food and energy drops (S2's HUD reader on the real tokens).
- **Error:** the flow term's per-cell x-error on the target frame's periphery at each of the 8 finest-grid levels.
- **Three records:**
  - real;
  - none (the denoiser's record input zeroed, record_read's offset kept; Mamba's belief from the real record);
  - switched: frames replaced, by a fixed p = 0.5 draw, with the world's own one-step estimates. These are made in
    64-frame chunks with 8 prefix frames, at random finest-grid levels, from the real memory and record, clamped, as
    `switch_ft` does. Mamba re-reads the switched record (rec route).
- **Regions:** S2's frozen entity reader on the real frames.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.b2 [N_WINDOWS N_DROPS]
"""
import importlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
WORLDS = {"P1.5": ROOT / "artifacts/experiments/20261116_p15/run/ckpt_10.pt",
          "R4k": ROOT / "artifacts/experiments/20261119_switching/run/sw_rec2/ckpt_sw4000.pt"}
ARMS = ("real", "none", "switched")
REGIONS = ("mob", "near_mob", "vacated", "map_other", "food", "energy", "hud_other")
FOOD, ENERGY, SEED = 64, 66, 20261120


def episode_state(model, cfg, ep_list, i, dev):
    """The episode's tokens, chunk arrays, real record / belief, and the switched record / belief (rec route)."""
    m, e = cfg["p1"]["monitor"], cfg["edm"]
    P, k, clamp = cfg["data"]["context_prefix"], e["shortcut"]["k_max"], cfg["perception"]["periphery"]["tanh_clamp"]
    ep = ep_list[i]
    recs = [[(i, 1, t, *D.fixation_prior(np.random.default_rng([m["fixed_fixation_seed"], int(i), t]), ())) for t in range(ep.T + 1)]]
    c = TR.tensors(D.gather(ep_list, recs), dev)
    Pe = M.Percept(*(torch.cat(z_, 1) for z_ in zip(*[model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]).parts()
                                                    for a in range(0, ep.T + 1, 512)])), c["fixation"])
    T1 = Pe.p.shape[1]

    def record_and_belief(p):
        """The record's denoiser input (bf16, as autocast reads it) and the predicted beliefs, both scanned in 512-frame
        segments with the state carried (whole-episode stacks at 1,024 dims per cell do not fit in 6 GB)."""
        state, carry, rins, ys = model.record.zero(1, dev), model.belief.initial(1, dev), [], []
        for a in range(0, T1, 512):
            sl = slice(a, min(T1, a + 512))
            first = torch.zeros(1, sl.stop - a, dtype=torch.bool, device=dev); first[:, 0] = a == 0
            ri, si = model.record.scan(p[:, sl], first, state)
            state = si[:, -1]
            P_ = M.Percept(Pe.p[:, sl], Pe.g[:, sl], Pe.v[:, sl], Pe.fix[:, sl])
            y, carry = model.belief(model.belief.tokens(model.belief.percept_embed(P_, si), c["reward"][:, sl], c["terminated"][:, sl],
                                                        c["action"][:, sl]), carry)
            rins.append(ri.to(torch.bfloat16)); ys.append(y[:, 1::2]); del ri, si
        y = torch.cat(ys, 1)
        return torch.cat(rins, 1), torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], 1)
    rin, bp = record_and_belief(Pe.p)
    g = torch.Generator(device=dev).manual_seed(SEED * 1000 + int(i))
    est = Pe.p.clone()
    mem = model.belief.memory(bp)
    for a in range(0, T1, 64):
        sl = slice(max(0, a - P), min(T1, a + 64)); n = sl.stop - sl.start
        x0 = tuple(t[:, sl] for t in Pe.parts())
        lv = [M.shortcut_levels((1, n), k, torch.zeros(1, dtype=torch.bool, device=dev), dev, g) for _ in range(2)]
        tau = [s.float() / k for s in M._parts(tuple(l[0] for l in lv))]
        z = tuple((1 - M._bcast(tt, x)) * torch.randn(x.shape, device=dev, generator=g, dtype=torch.float32) + M._bcast(tt, x) * x.float()
                  for x, tt in zip(x0, tau))
        out = model.denoiser(tuple(zz.to(x.dtype) for zz, x in zip(z, x0)), tuple(M.shortcut_code(l[0], l[1], k) for l in lv),
                             c["prev_action"][:, sl], (mem[:, sl], rin[:, sl]), c["fixation"][:, sl], c["episode"][:, sl], c["valid"][:, sl])
        est[:, a:sl.stop] = out[0].float().clamp(-clamp, clamp)[:, a - sl.start:].to(est.dtype)
    mask = torch.rand(1, T1, device=dev, generator=g) < 0.5
    rin_sw, bp_sw = record_and_belief(torch.where(mask[..., None, None], est, Pe.p))
    return {"Pe": Pe, "c": c, "rin": rin, "bp": bp, "rin_sw": rin_sw, "bp_sw": bp_sw}


def window_err(model, cfg, st, i, t, dev):
    """[arms, 8 levels, 81] ramp-weighted squared x-error per periphery cell (channel mean) on target frame t."""
    m, e = cfg["p1"]["monitor"], cfg["edm"]
    P, k, sc = cfg["data"]["context_prefix"], e["shortcut"]["k_max"], e["shortcut"]
    sl = slice(max(0, t - P), t + 1); n = sl.stop - sl.start
    levels = torch.arange(k, device=dev)
    rep = lambda a: a[:, sl].expand(len(levels), *a[:, sl].shape[1:])
    Pe, c = st["Pe"], st["c"]
    x0 = tuple(rep(a) for a in Pe.parts())
    g = torch.Generator(device=dev).manual_seed(m["fixed_noise_seed"] * 1_000_003 + int(i) * 4_099 + t)
    noise = tuple(torch.randn(a.shape, device=dev, generator=g) for a in x0)
    sg = torch.full((len(levels), n), sc["context_signal"], device=dev, dtype=torch.long); sg[:, -1] = levels
    em = torch.full_like(sg, k.bit_length() - 1)
    tau = [s.float() / k for s in M._parts((sg, sg))]
    zt = tuple((1 - M._bcast(tt, x)) * nn + M._bcast(tt, x) * x for x, nn, tt in zip(x0, noise, tau))
    code = (M.shortcut_code(sg, em, k), M.shortcut_code(sg, em, k))
    w = sc["ramp"][0] * tau[0] + sc["ramp"][1]
    out = []
    for arm in ARMS:
        bp, rin = (st["bp_sw"], st["rin_sw"]) if arm == "switched" else (st["bp"], st["rin"])
        rin = torch.zeros_like(rin) if arm == "none" else rin
        o = model.denoiser(zt, code, rep(c["prev_action"]), (model.belief.memory(rep(bp)), rep(rin)), rep(c["fixation"]),
                           rep(c["episode"]), rep(c["valid"]))
        err = M._bcast(w, x0[0]) * (o[0].float() - x0[0].float()) ** 2
        out.append(err[:, -1].mean(-1))
    return torch.stack(out).float().cpu().numpy()


def monitor_g1(model, cfg, st, i, t, dev):
    """TR._monitor's own G1 (shortcut_loss, flow term) for window target t from this script's state: the smoke checks
    that the chunked record and belief reproduce the monitor's whole-episode pass."""
    m, e = cfg["p1"]["monitor"], cfg["edm"]
    P, k = cfg["data"]["context_prefix"], e["shortcut"]["k_max"]
    levels = torch.arange(k, device=dev)
    sl = slice(max(0, t - P), t + 1); n = sl.stop - sl.start
    rep = lambda a: a[:, sl].expand(len(levels), *a[:, sl].shape[1:])
    g = torch.Generator(device=dev).manual_seed(m["fixed_noise_seed"] * 1_000_003 + int(i) * 4_099 + t)
    x0 = tuple(rep(a) for a in st["Pe"].parts())
    noise = tuple(torch.randn(a.shape, device=dev, generator=g) for a in x0)
    sg = torch.full((len(levels), n), e["shortcut"]["context_signal"], device=dev, dtype=torch.long); sg[:, -1] = levels
    lv = [(sg, torch.full_like(sg, k.bit_length() - 1))] * 2
    c = st["c"]
    l = M.shortcut_loss(model.denoiser, x0, rep(c["prev_action"]), (model.belief.memory(rep(st["bp"])), rep(st["rin"])),
                        rep(c["fixation"]), rep(c["episode"]), rep(c["valid"]), e["shortcut"],
                        torch.zeros(len(levels), dtype=torch.bool, device=dev), False, lv, noise)
    return l[:, -1].float().cpu().numpy()


def regions(R, Pe, t):
    """Region index per periphery cell of target frame t (REGIONS order), from the reader on frames t - 1, t."""
    kind = S2.read(R, Pe.p[:, t - 1:t + 1].float(), Pe.v[:, t - 1:t + 1].float(), np.zeros((1, 2), int))[0][0]
    now, before = kind[1] > 0, kind[0] > 0
    pad = np.pad(now, 1); near = np.zeros_like(now)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            near |= pad[1 + dr:8 + dr, 1 + dc:10 + dc]
    reg = np.full(81, REGIONS.index("hud_other"))
    mp = np.where(now, 0, np.where(before, 2, np.where(near, 1, 3)))      # mob, vacated over near, map_other
    reg[:63] = mp.reshape(-1)
    reg[FOOD], reg[ENERGY] = REGIONS.index("food"), REGIONS.index("energy")
    return reg


def picks(cfg, ep_list, n_windows, n_drops, R, model, dev):
    """(episode, target t, set) for the monitor's windows and for food / energy drop targets read on real tokens."""
    m = cfg["p1"]["monitor"]
    lengths = np.array([ep.T for ep in ep_list])
    rng = np.random.default_rng(m["fixed_noise_seed"])
    pk = rng.choice(int(lengths.sum()), n_windows, replace=False)          # TR._monitor's own draw for this count
    off = np.r_[0, np.cumsum(lengths)]
    ep_of = np.searchsorted(off, pk, side="right") - 1
    out = [(int(i), int(p - off[i] + 1), "monitor") for i, p in zip(ep_of, pk)]
    drops = {"food": [], "energy": []}
    for i, ep in enumerate(ep_list):
        recs = [[(i, 1, t, 1, 1) for t in range(ep.T + 1)]]
        c = TR.tensors(D.gather(ep_list, recs), dev)
        with TR.autocast(dev, cfg):
            parts = [model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]) for a in range(0, ep.T + 1, 512)]
            p, v = torch.cat([x.p for x in parts], 1), torch.cat([x.v for x in parts], 1)
        hud = S2.read(R, p.float(), v.float(), np.zeros((1, 2), int))[1][0]
        for name, col in (("food", 1), ("energy", 3)):
            drops[name] += [(i, int(t), f"{name}_drop") for t in np.flatnonzero(np.diff(hud[:, col].astype(int)) < 0) + 1]
    r2 = np.random.default_rng(SEED)
    for name in ("food", "energy"):
        sel = r2.permutation(len(drops[name]))[:n_drops]
        out += [drops[name][j] for j in sorted(sel)]
    return out


@torch.no_grad()
def main(n_windows=None, n_drops=300):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    tag = "" if n_windows is None else "_smoke"
    model, cfg = AI.load(WORLDS["P1.5"])
    model.requires_grad_(False).eval()
    dev = next(model.parameters()).device
    ep_list = D.load_split(cfg)["monitor"]
    R = S2.load_readers(model, cfg)
    W = picks(cfg, ep_list, n_windows or cfg["p1"]["monitor"]["windows"], n_drops, R, model, dev)
    print(f"{len(W)} windows: " + ", ".join(f"{s} {sum(w[2] == s for w in W)}" for s in ("monitor", "food_drop", "energy_drop")), flush=True)
    reg, check = {}, []
    res = {"windows": [list(w) for w in W]}
    for name, ck in WORLDS.items():
        if name != "P1.5":
            del model; torch.cuda.empty_cache()
            model, cfg = AI.load(ck)
            model.requires_grad_(False).eval()
        errs, t0 = np.zeros((len(W), len(ARMS), 8, 81), np.float32), time.time()
        for i in sorted({w[0] for w in W}):
            with TR.autocast(dev, cfg):
                st = episode_state(model, cfg, ep_list, i, dev)
                for j, (e, t, _) in enumerate(W):
                    if e != i:
                        continue
                    errs[j] = window_err(model, cfg, st, i, t, dev)
                    if name == "P1.5":
                        reg[j] = regions(R, st["Pe"], t)
                    if n_windows is not None and name == "P1.5" and W[j][2] == "monitor":   # smoke: the monitor's own G1
                        check.append(monitor_g1(model, cfg, st, i, t, dev))
            print(f"{name} episode {i} {time.time() - t0:.0f}s", flush=True)
        res[name] = errs
        if check and name == "P1.5":
            mine = np.mean(check, 0); ref = np.array(TR._monitor(model, cfg, ep_list, n_windows)["g1_by_level"])
            print("smoke check, G1 by level: mine", np.round(mine, 6).tolist(), "monitor", np.round(ref, 6).tolist(),
                  "max rel diff", float(np.abs(mine / ref - 1).max()), flush=True)
    rg = np.stack([reg[j] for j in range(len(W))])
    np.savez_compressed(HERE / f"b2{tag}.npz", regions=rg, sets=np.array([w[2] for w in W]), **{k: res[k] for k in WORLDS})
    summarise(tag)


def summarise(tag=""):
    """Reliance (none - real) and the switching penalty (switched - real) per region, at levels 0-3 and all levels, per
    world, with window-bootstrap SEs paired between the worlds; and each region's share of the periphery loss."""
    z = np.load(HERE / f"b2{tag}.npz")
    rg, sets = z["regions"], z["sets"]
    rng = np.random.default_rng(SEED)
    out = {}
    for lv_name, lv in (("levels 0-3", slice(0, 4)), ("all levels", slice(0, 8))):
        for wset, regs in (("monitor", REGIONS), ("food_drop", ("food",)), ("energy_drop", ("energy",))):
            rows = np.flatnonzero(sets == wset)
            if not len(rows):
                continue
            for r in regs:
                cell = rg[rows] == REGIONS.index(r)                         # [n, 81]
                have = cell.any(1)
                if not have.any():
                    continue
                per = {}
                for wd in WORLDS:
                    e = z[wd][rows][:, :, lv].mean(2)                       # [n, arms, 81]
                    val = (e * cell[:, None]).sum(-1) / np.maximum(cell.sum(-1), 1)[:, None]   # [n, arms]
                    per[wd] = val[have]
                n = int(have.sum())
                idx = [rng.integers(0, n, n) for _ in range(300)]
                row = {"windows": n}
                for wd, v in per.items():
                    rel, pen = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
                    row[wd] = {"real": float(v[:, 0].mean()), "reliance": float(rel.mean()), "reliance_se": float(np.std([rel[q].mean() for q in idx])),
                               "penalty": float(pen.mean()), "penalty_se": float(np.std([pen[q].mean() for q in idx]))}
                a, b = per["P1.5"], per["R4k"]
                d_rel, d_pen = (b[:, 1] - b[:, 0]) - (a[:, 1] - a[:, 0]), (b[:, 2] - b[:, 0]) - (a[:, 2] - a[:, 0])
                row["R4k_minus_P1.5"] = {"reliance": float(d_rel.mean()), "reliance_se": float(np.std([d_rel[q].mean() for q in idx])),
                                         "penalty": float(d_pen.mean()), "penalty_se": float(np.std([d_pen[q].mean() for q in idx]))}
                out[f"{lv_name}|{wset}|{r}"] = row
                print(f"{lv_name:10s} {wset:11s} {r:9s} n={n:5d}", " | ".join(
                    f"{wd}: real {x['real']:.5f} rel {x['reliance']:+.5f} pen {x['penalty']:+.5f}" for wd, x in row.items() if wd in WORLDS),
                      f"| R4k-P1.5 rel {row['R4k_minus_P1.5']['reliance']:+.5f}±{row['R4k_minus_P1.5']['reliance_se']:.5f}"
                      f" pen {row['R4k_minus_P1.5']['penalty']:+.5f}±{row['R4k_minus_P1.5']['penalty_se']:.5f}", flush=True)
    rows = np.flatnonzero(sets == "monitor")
    share = {}
    for wd in WORLDS:
        e = z[wd][rows].mean(2)                                             # [n, arms, 81], all levels
        tot = e.sum(-1)                                                     # [n, arms]
        share[wd] = {r: {"loss_share_real": float((e[:, 0] * (rg[rows] == i)).sum() / tot[:, 0].sum()),
                         "penalty_share": float(((e[:, 2] - e[:, 0]) * (rg[rows] == i)).sum() / (tot[:, 2] - tot[:, 0]).sum()),
                         "penalty_rel_to_loss": float((tot[:, 2] - tot[:, 0]).sum() / tot[:, 0].sum())}
                     for i, r in enumerate(REGIONS)}
        print(wd, "periphery loss share / switching-penalty share by region:",
              ", ".join(f"{r} {v['loss_share_real']:.3f}/{v['penalty_share']:.3f}" for r, v in share[wd].items()),
              f"| total penalty / loss {share[wd]['mob']['penalty_rel_to_loss']:+.4f}", flush=True)
    out["shares"] = share
    (HERE / f"b2{tag}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1])) if a else main()
