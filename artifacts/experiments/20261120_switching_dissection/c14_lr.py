"""C14 (PLAN.md): P1.5 continued from checkpoint 10 for one checkpoint interval at checkpoint 5's learning-rate factor,
held constant; saves checkpoint 11 in c14_run/ (copies of P1.5's state, never links). Then `readings`: C5's calibration
at checkpoints 10 and 11 with their own S1 readers, and C13's levels at 11.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c14_lr train
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c14_lr readings
"""
import copy, importlib, os, shutil, sys
from pathlib import Path

import torch

from saccade import train as TR

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "20261116_p15/run"
RUN = HERE / "c14_run"


def train():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    ck = torch.load(SRC / "ckpt_10.pt", map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(ck["cfg"])
    total, frac = cfg["p1"]["frames"], cfg["p1"]["checkpoint_every_fraction"]
    f5 = TR.lr_factor(5 * total * frac, cfg)                            # checkpoint 5's factor
    cfg["p1"] = dict(cfg["p1"], frames=int(round(total * 1.1)), checkpoint_every_fraction=1 / 11)   # one more interval; ckpt 11
    assert abs(cfg["p1"]["frames"] * cfg["p1"]["checkpoint_every_fraction"] - total * frac) < 1
    RUN.mkdir(exist_ok=True)
    if not (RUN / "latest.pt").exists():
        shutil.copyfile(SRC / "ckpt_10.pt", RUN / "latest.pt")          # copies: the trainer overwrites latest.pt
        for f in ("gaze.bin", "chooser.jsonl", "log.jsonl"):
            shutil.copyfile(SRC / f, RUN / f)
    print("constant lr factor", f5, "frames", total, "->", cfg["p1"]["frames"], flush=True)
    TR.lr_factor = lambda frames, cfg_: f5                              # the manipulation: held at checkpoint 5's rate
    TR.validate_resume = lambda *a, **k: None                           # the budget and schedule differ by design
    TR.p1(cfg, ck["world_seed"], RUN)


@torch.no_grad()
def calibrate(ck, tag):
    """C5's draws and keep rule (8 draws of the default sampler per start mob, C4.samples) at checkpoint `ck`, read with
    its own S1 reader (C6's protocol); per-unit outcomes to c14_calib_{tag}.npz. C5's own files are untouched."""
    import numpy as np
    AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
    C2 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c2_eye")
    C4 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c4_sample")
    C5 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c5_calibration")
    DL = importlib.import_module("artifacts.experiments.20261120_switching_dissection.deletions")
    from saccade.eval import s1 as S
    model, cfg = AI.load(ck); model.requires_grad_(False).eval()
    R = {"kind": C5.reader(model, cfg, False)}
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    dl = S.episodes(L)
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    z = np.load(DL.EV.S2 / "model.npz"); mg, mp = z["m_grid"], z["m_pos"].astype(int)
    units = [(int(rows[u[0], 0]), int(rows[u[0], 3]) + 1, int(u[1]), int(u[2]), int(u[3]), bool(u[7])) for u in U
             if 9 <= int(rows[u[0], 3]) + 1 <= L["ep_T"][int(rows[u[0], 0])] and DL.deleted(mg, mp, u)]
    out = {"kept": [], "real": [], "kind": [], "night": [], "e": [], "t": []}
    for e in sorted({x[0] for x in units}):
        st = C2.episode(model, cfg, dl, e)
        us = [x for x in units if x[0] == e]; ts = sorted({x[1] for x in us})
        real = C5.kind_probs(R, st[1].p[0, ts].float(), st[1].v[0, ts].float()).argmax(-1).cpu()
        dr = torch.stack([C5.kind_probs(R, *C4.samples(model, cfg, st, e, ts, d)).argmax(-1).cpu() for d in range(C5.DRAWS)])
        for _, t, kind, i, j, night in us:
            qi = ts.index(t); dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
            cells = [r * 9 + c for r in range(i - dpos[0] - 1, i - dpos[0] + 2) for c in range(j - dpos[1] - 1, j - dpos[1] + 2) if 0 <= r < 7 and 0 <= c < 9]
            if not cells:
                continue
            for key, val in (("kept", (dr[:, qi][:, cells] == kind).any(-1).numpy()), ("real", bool((real[qi, cells] == kind).any())),
                             ("kind", kind), ("night", night), ("e", e), ("t", t)):
                out[key].append(val)
        print(tag, e, len(out["kind"]), flush=True)
    np.savez_compressed(HERE / f"c14_calib_{tag}.npz", **{k: np.array(v) for k, v in out.items()})


def readings():
    import json
    import numpy as np
    calibrate(SRC / "ckpt_10.pt", "ckpt10"); calibrate(RUN / "ckpt_11.pt", "ckpt11")
    a, b = (np.load(HERE / f"c14_calib_{t}.npz") for t in ("ckpt10", "ckpt11"))
    assert (a["e"] == b["e"]).all() and (a["t"] == b["t"]).all() and (a["kind"] == b["kind"]).all()
    res, rng = {}, np.random.default_rng(0)
    for kn, name in ((2, "cows"), (1, "zombies"), (3, "skeletons")):
        for tag, sel in (("day", ~a["night"]), ("night", a["night"])):
            m = (a["kind"] == kn) & sel
            if not m.any():
                continue
            da, db = 1 - a["kept"][m].mean(1), 1 - b["kept"][m].mean(1)
            boot = [(lambda ix: (db[ix] - da[ix]).mean())(rng.integers(0, m.sum(), m.sum())) for _ in range(4000)]
            res[f"{name}|{tag}"] = {"units": int(m.sum()), "deletion_ckpt10": float(da.mean()), "deletion_ckpt11": float(db.mean()),
                                    "paired_change": float((db - da).mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                                    "real_floor_ckpt10": float(a["real"][m].mean()), "real_floor_ckpt11": float(b["real"][m].mean())}
    log = lambda run: [r for r in map(json.loads, (run / "log.jsonl").read_text().splitlines()) if "monitor" in r]
    m10, m11 = log(SRC)[-1]["monitor"], log(RUN)[-1]["monitor"]
    res["monitor"] = {"g1_ckpt10": m10["g1"], "g1_ckpt11": m11["g1"], "by_level_ckpt10": m10["g1_by_level"], "by_level_ckpt11": m11["g1_by_level"]}
    (HERE / "c14_readings.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)
    C13 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c13_levels")
    C13.main(False, str(RUN / "ckpt_11.pt"), True)


if __name__ == "__main__":
    {"train": train, "readings": readings}[sys.argv[1]]()
