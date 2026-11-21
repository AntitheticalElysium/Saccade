"""P1.6's readings 2 and 4-7 (PLAN.md), run after the evaluation queue on the selected checkpoint; reading 3 (A5's paired
look) waits on the S2 port. The real past is built as the switching dissection's (C2.episode, the monitor's random prior
fixations): a cell is looked at if it lies in a frame's fixation window. Random in the game state, not in position
(edge cells are covered less), so read "looked" within position bands.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_p16.readings reliance CKPT     (2) does the world use it?
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_p16.readings calibration CKPT  (4, 6) on P1.6 and P1.5
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_p16.readings zombies CKPT      (5) C7 on P1.6
    python -m artifacts.experiments.20261120_p16.readings trace                                     (7) the eye in training (CPU)
"""
import importlib, json, os, sys
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

SD = "artifacts.experiments.20261120_switching_dissection."
AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
C1, C2, C4, C5, DL = (importlib.import_module(SD + n) for n in ("c1", "c2_eye", "c4_sample", "c5_calibration", "deletions"))
HERE = Path(__file__).resolve().parent
OUT = HERE / "readings"
P15 = HERE.parent / "20261116_p15/run/ckpt_10.pt"
DEV = torch.device("cuda")
WIN = M.window


def label_frames():
    """C2's 3,000 DEV target frames (its draw, exactly)."""
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    eps = np.sort(np.random.default_rng(20261120).choice(len(L["ep_T"]), 150, replace=False))
    rng = np.random.default_rng(7)
    picks = {int(e): np.sort(rng.choice(np.arange(9, int(L["ep_T"][e]) + 1), min(20, int(L["ep_T"][e]) - 8), replace=False))
             for e in eps if L["ep_T"][e] >= 10}
    return L, picks


@torch.no_grad()
def episode(model, cfg, dl, e, zero_from=None):
    """C2.episode (with the glimpse); with `zero_from`, Mamba reads the record with its features from that index on
    zeroed (reading 2's arm: Record.size(cfg, coarse=True), the fovea group; the full size zeroes nothing, the smoke's
    check that this copy is C2's pass)."""
    if zero_from is None:
        return C2.episode(model, cfg, dl, e)
    m = cfg["p1"]["monitor"]; T = int(dl[e].T)
    recs = [[(0, 1, t, *D.fixation_prior(np.random.default_rng([m["fixed_fixation_seed"], int(e), t]), ())) for t in range(T + 1)]]
    c = TR.tensors(D.gather([dl[e]], recs), DEV)
    with TR.autocast(DEV, cfg):
        Pe = M.Percept(*(torch.cat(z, 1) for z in zip(*[model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]).parts()
                                                        for a in range(0, T + 1, 512)])), c["fixation"])
        state, carry, rins, ys = model.record.zero(1, DEV), model.belief.initial(1, DEV), [], []
        for a in range(0, T + 1, 512):
            sl = slice(a, min(T + 1, a + 512))
            first = torch.zeros(1, sl.stop - a, dtype=torch.bool, device=DEV); first[:, 0] = a == 0
            ri, si = model.record.scan(model.record.input(Pe.p[:, sl], Pe.g[:, sl], Pe.fix[:, sl]), first, state); state = si[:, -1]
            si = si.clone(); si[..., zero_from:] = 0                            # Mamba's read without the fovea group
            y, carry = model.belief(model.belief.tokens(model.belief.percept_embed(M.Percept(Pe.p[:, sl], Pe.g[:, sl], Pe.v[:, sl], Pe.fix[:, sl]), si),
                                                        c["reward"][:, sl], c["terminated"][:, sl], c["action"][:, sl]), carry)
            rins.append(ri.to(torch.bfloat16)); ys.append(y[:, 1::2])
        y = torch.cat(ys, 1)
    return c, Pe, torch.cat(rins, 1), torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], 1)


@torch.no_grad()
def window_err(model, cfg, st, e, t, zero_read_fovea):
    """B2's form: the ramp-weighted per-cell x-error of target frame t's periphery at each of the 8 finest levels [8, 81],
    8 context frames at the context signal, the monitor's noise seeding."""
    c, Pe, rin, bp = st
    m, sc = cfg["p1"]["monitor"], cfg["edm"]["shortcut"]
    P, k = cfg["data"]["context_prefix"], sc["k_max"]
    sl = slice(t - P, t + 1); levels = torch.arange(k, device=DEV)
    rep = lambda a: a[:, sl].expand(len(levels), *a[:, sl].shape[1:])
    x0 = tuple(rep(a) for a in Pe.parts())
    g = torch.Generator(device=DEV).manual_seed(m["fixed_noise_seed"] * 1_000_003 + int(e) * 4_099 + int(t))
    noise = tuple(torch.randn(a.shape, device=DEV, generator=g) for a in x0)
    sg = torch.full((len(levels), P + 1), sc["context_signal"], device=DEV, dtype=torch.long); sg[:, -1] = levels
    em = torch.full_like(sg, k.bit_length() - 1)
    tau = [s.float() / k for s in M._parts((sg, sg))]
    zt = tuple((1 - M._bcast(tt, x)) * nn + M._bcast(tt, x) * x.float() for x, nn, tt in zip(x0, noise, tau))
    r = rep(rin)
    if zero_read_fovea:
        r = r.clone(); r[..., M.Record.size(cfg, coarse=True):] = 0
    with TR.autocast(DEV, cfg):
        o = model.denoiser(zt, (M.shortcut_code(sg, em, k),) * 2, rep(c["prev_action"]), (model.belief.memory(rep(bp)), r), rep(Pe.fix),
                           rep(c["episode"]), rep(c["valid"]))
    w = sc["ramp"][0] * tau[0][:, -1] + sc["ramp"][1]
    return (w[:, None] * (o[0][:, -1].float() - x0[0][:, -1].float()).square().mean(-1)).cpu().numpy()


def reliance(ck, smoke=False):
    """(2) Error with the fovea group zeroed in the denoiser's read (Mamba intact), and in both reads, against intact; by
    region (labels), light, camera still / moved, and whether the cell was looked at in the last 1 / 8 frames."""
    model, cfg = AI.load(ck); model.requires_grad_(False).eval()
    assert model.record.fovea, "a P1.6 checkpoint (the fovea group)"
    L, picks = label_frames()
    groups, light, lookup, loff, _ = C1.semantics(L)
    dl = S.episodes(L)
    check = {}
    if smoke:
        picks = dict(list(picks.items())[:3])
        e = next(iter(picks)); a, b = C2.episode(model, cfg, dl, e), episode(model, cfg, dl, e, M.Record.size(cfg))
        check = {"copy_equals_c2_rin": float((a[2].float() - b[2].float()).abs().max()), "copy_equals_c2_bp": float((a[3].float() - b[3].float()).abs().max())}
    rows = []
    for e, ts in picks.items():
        st, stz = episode(model, cfg, dl, e), episode(model, cfg, dl, e, M.Record.size(cfg, coarse=True))
        fix = st[1].fix[0].cpu()
        for t in (int(x) for x in ts):
            arms = [window_err(model, cfg, st, e, t, False), window_err(model, cfg, st, e, t, True), window_err(model, cfg, stz, e, t, True)]
            look1, look8 = np.zeros(81, bool), np.zeros(81, bool)
            look1[WIN(fix[t - 1]).numpy()] = True
            for dt in range(1, 9):
                look8[WIN(fix[t - dt]).numpy()] = True
            moved = bool((L["s_pos"][loff[e] + t] != L["s_pos"][loff[e] + t - 1]).any())
            rows.append((np.stack(arms), lookup[loff[e] + t], look1, look8, moved))
        print("reliance", e, len(rows), flush=True)
    err = np.stack([r[0] for r in rows])                                  # [frames, 3 arms, 8 levels, 81]
    rw = np.array([r[1] for r in rows]); night = light[rw] < 0.5
    l1, l8, mv = np.stack([r[2] for r in rows]), np.stack([r[3] for r in rows]), np.array([r[4] for r in rows])
    res = {"frames": len(rows), "arms": ["intact", "fovea_zeroed_denoiser_read", "fovea_zeroed_both_reads"], "smoke_checks": check, "by": {}}
    for g in C1.GROUPS:
        gm = groups[g][rw]                                                # [frames, 81]
        for tag, fsel in (("day", ~night), ("night", night), ("camera_still", ~mv), ("camera_moved", mv)):
            for lk, lm in (("all", np.ones_like(l1)), ("looked_1", l1), ("looked_8", l8), ("unlooked_8", ~l8)):
                cell = gm & lm & fsel[:, None]
                if cell.sum() < 20:
                    continue
                e_arm = [(err[:, a].sum(1) * cell).sum() / cell.sum() for a in range(3)]   # summed over levels, per cell
                res["by"][f"{g}|{tag}|{lk}"] = {"cells": int(cell.sum()), "intact": float(e_arm[0]),
                                                "rel_increase_denoiser": float(e_arm[1] / e_arm[0] - 1), "rel_increase_both": float(e_arm[2] / e_arm[0] - 1)}
    np.savez_compressed(OUT / f"reliance_{Path(ck).stem}{'_smoke' if smoke else ''}.npz", err=err, rows=rw, look1=l1, look8=l8, moved=mv)
    return res


def units(L):
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)        # C5's units
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    z = np.load(DL.EV.S2 / "model.npz"); mg, mp = z["m_grid"], z["m_pos"].astype(int)
    return [(int(rows[u[0], 0]), int(rows[u[0], 3]) + 1, int(u[1]), int(u[2]), int(u[3]), bool(u[7])) for u in U
            if 9 <= int(rows[u[0], 3]) + 1 <= L["ep_T"][int(rows[u[0], 0])] and DL.deleted(mg, mp, u)]


def calibration(ck, smoke=False):
    """(4) C5's draws (default sampler, 8 per unit, own S1 reader) with each unit's mob looked at in frame t - 1 or not;
    (6) spurious mobs (drawn where the real frame has none) per cell inside t - 1's window against outside. On ck and on
    P1.5 checkpoint 10 (the control: its record holds no glimpse; with C5's reader, seeds and units it must reproduce
    C5's keeps exactly)."""
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    dl = S.episodes(L); us_all = units(L)
    if smoke:
        us_all = [x for x in us_all if x[0] in sorted({u[0] for u in us_all})[:2]]
    res = {}
    for name, path in (("p16", Path(ck)), ("p15", P15)):
        model, cfg = AI.load(path); model.requires_grad_(False).eval()
        R = {"kind": S.load_reader(path.parent / "s1/results.reader.pt", path, DEV)}   # each world's S1 kind reader (C5's for P1.5)
        rec, spur = [], {"in": [0., 0], "out": [0., 0], "real_in": [0., 0], "real_out": [0., 0]}
        for e in sorted({x[0] for x in us_all}):
            st = C2.episode(model, cfg, dl, e)
            us = [x for x in us_all if x[0] == e]; ts = sorted({x[1] for x in us})
            real = C5.kind_probs(R, st[1].p[0, ts].float(), st[1].v[0, ts].float()).argmax(-1).cpu()
            dr = torch.stack([C5.kind_probs(R, *C4.samples(model, cfg, st, e, ts, d)).argmax(-1).cpu() for d in range(C5.DRAWS)])
            fix = st[1].fix[0].cpu()
            for qi, t in enumerate(ts):                                       # (6) cells of t - 1's window that the real
                win = np.zeros(63, bool); cells = WIN(fix[t - 1]).numpy(); win[cells[cells < 63]] = True   # frame t shows empty
                empty = (real[qi] == 0).numpy()
                drawn = (dr[:, qi] > 0).numpy()                               # [D, 63]
                for key, sel in (("in", win & empty), ("out", ~win & empty)):
                    spur[key][0] += float(drawn[:, sel].sum()); spur[key][1] += int(sel.sum()) * C5.DRAWS
            for _, t, kind, i, j, night in us:
                qi = ts.index(t); dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
                cells = [r * 9 + c for r in range(i - dpos[0] - 1, i - dpos[0] + 2) for c in range(j - dpos[1] - 1, j - dpos[1] + 2) if 0 <= r < 7 and 0 <= c < 9]
                if not cells:
                    continue
                looked = bool((WIN(fix[t - 1]).numpy() == i * 9 + j).any())
                rec.append({"kind": kind, "night": night, "looked": looked, "moved": bool((dpos != 0).any()),
                            "kept": (dr[:, qi][:, cells] == kind).any(-1).numpy(), "real": bool((real[qi, cells] == kind).any())})
            print(name, e, len(rec), flush=True)
        kinds = np.array([x["kind"] for x in rec]); night = np.array([x["night"] for x in rec]); looked = np.array([x["looked"] for x in rec])
        moved = np.array([x["moved"] for x in rec]); kept = np.stack([x["kept"] for x in rec])
        out = {"units": len(rec), "looked_share": float(looked.mean()), "keep": {},
               "spurious_per_cell": {k: v[0] / max(v[1], 1) for k, v in spur.items() if not k.startswith("real")}}
        for kn in (2, 1, 3):
            for tag, sel in (("day", ~night), ("night", night)):
                for lk, lm in (("looked", looked), ("unlooked", ~looked), ("looked_still", looked & ~moved), ("unlooked_still", ~looked & ~moved)):
                    m = (kinds == kn) & sel & lm
                    if m.sum() >= 5:
                        out["keep"][f"{C5.KINDS[kn]}|{tag}|{lk}"] = {"units": int(m.sum()), "deletion": float(1 - kept[m].mean()),
                                                                     "se": float((1 - kept[m].mean(1)).std(ddof=1) / np.sqrt(m.sum()))}
        if name == "p15" and not smoke:
            z5 = np.load(Path(C5.__file__).parent / "c5_units.npz")["kept"]
            out["reproduces_c5"] = float((kept == z5).mean()) if kept.shape == z5.shape else f"shape {kept.shape} vs {z5.shape}"
        res[name] = out
        np.savez_compressed(OUT / f"calibration_{name}{'_smoke' if smoke else ''}.npz", kept=kept, kinds=kinds, night=night, looked=looked, moved=moved)
        del model; torch.cuda.empty_cache()
    return res


def zombies(ck):
    """(5) C7 on ck with its own reader."""
    C7 = importlib.import_module(SD + "c7_zombies")
    C7.C5.CK = Path(ck)
    C7.C5.S2.load_readers = lambda model, cfg: {"kind": S.load_reader(Path(ck).parent / "s1/results.reader.pt", Path(ck), DEV)}
    C7.HERE = OUT                                                         # c7_zombies.json written here, not over C7's
    C7.main()
    return json.loads((OUT / "c7_zombies.json").read_text())


def trace():
    """(7) C11b's coverage (chosen against paired uniform window) per pass on the 400 labelled training episodes, for
    P1.6's trace and P1.5's."""
    T11 = importlib.import_module(SD + "c11_trace")
    res = {}
    for name, run in (("p16", HERE / "run"), ("p15", P15.parent)):
        T11.RUN, T11.HERE = run, OUT / name
        (OUT / name).mkdir(parents=True, exist_ok=True)
        T11.main()
        res[name] = json.loads((OUT / name / "c11_trace.json").read_text())
    return res


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    what = sys.argv[1]
    if what != "trace":
        assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    smoke = "smoke" in sys.argv[3:]
    res = {"reliance": lambda: reliance(sys.argv[2], smoke), "calibration": lambda: calibration(sys.argv[2], smoke),
           "zombies": lambda: zombies(sys.argv[2]), "trace": trace}[what]()
    (OUT / f"{what}{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1)[:4000], flush=True)
