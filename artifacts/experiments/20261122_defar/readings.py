"""Readings of the drift-row fine-tune (PLAN.md). Each checkpoint is read with its own S1 reader on the day-coin panel
(2,079 mobs, 8 draws). Native 4-step deletion and the guards come from the sampling-time remedies' native arm, 8-step
deletion from the deletion-by-signal-level sampler, plus the token-space chain readings, the drift-term curve and the
monitor. Paired against P1.5 checkpoint 10 and the higher-learning-rate continuation's checkpoint 11, whose per-unit
files already exist.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261122_defar.readings run       # GPU: the three readers
    python -m artifacts.experiments.20261122_defar.readings compare                         # CPU: the tables
"""
import importlib, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SD = HERE.parent / "20261120_switching_dissection"
CK = HERE / "defar_run" / "ckpt_11.pt"
TAG = "_" + CK.parent.parent.name + "_" + CK.stem                        # the remedies' and token reader's tag
KINDS = {2: "cows", 1: "zombies", 3: "skeletons"}


def run():
    mod = lambda n: importlib.import_module("artifacts.experiments.20261120_switching_dissection." + n)
    C15 = mod("c15_remedies"); C15.ARMS, C15.HERE = C15.ARMS[:1], HERE   # the native arm only, written here
    C15.main(False, str(CK))
    C13 = mod("c13_levels"); C13.HERE = HERE
    C13.main(False, str(CK), True)
    mod("midsignal_tokens").main(False, str(CK), HERE)
    compare()


def canonical():
    """The day-coin panel's units in C5's order (the enumeration of c14_lr.calibrate and c15_remedies, CPU only):
    (episode, frame, kind, i, j, night) for the units with cells."""
    mod = lambda n: importlib.import_module("artifacts.experiments.20261120_switching_dissection." + n)
    DL, S = mod("deletions"), importlib.import_module("saccade.eval.s1")
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    z = np.load(DL.EV.S2 / "model.npz"); mg, mp = z["m_grid"], z["m_pos"].astype(int)
    units = [(int(rows[u[0], 0]), int(rows[u[0], 3]) + 1, int(u[1]), int(u[2]), int(u[3]), bool(u[7])) for u in U
             if 9 <= int(rows[u[0], 3]) + 1 <= L["ep_T"][int(rows[u[0], 0])] and DL.deleted(mg, mp, u)]
    out = []
    for e in sorted({x[0] for x in units}):
        for x in [u for u in units if u[0] == e]:
            e_, t, kind, i, j, night = x
            dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
            if any(0 <= r < 7 and 0 <= c < 9 for r in range(i - dpos[0] - 1, i - dpos[0] + 2) for c in range(j - dpos[1] - 1, j - dpos[1] + 2)):
                out.append(x)
    return np.array(out)


def paired(new, old, ep, rng):
    """Deletion per unit (1 - keep over draws), the paired change, and its 95% interval from resampling paired episodes
    (units of one episode share history)."""
    d = (1 - new.mean(1)) - (1 - old.mean(1))
    eps, inv = np.unique(ep, return_inverse=True)
    s, n = np.bincount(inv, d, len(eps)), np.bincount(inv, None, len(eps))
    boot = [(lambda ix: s[ix].sum() / n[ix].sum())(rng.integers(0, len(eps), len(eps))) for _ in range(4000)]
    return {"new": float(1 - new.mean()), "ref": float(1 - old.mean()), "change": float(d.mean()),
            "ci95_episodes": np.quantile(boot, [0.025, 0.975]).tolist(), "episodes": int(len(eps))}


def compare():
    rng = np.random.default_rng(0)
    can = canonical()                                                    # [U, 6]
    n4 = np.load(HERE / f"c15_units{TAG}.npz"); p4 = np.load(SD / "c15_units.npz"); c4 = np.load(SD / "c14_calib_ckpt11.npz")
    n8 = np.load(HERE / "c13_units_ckpt_11.npz"); p8 = np.load(SD / "c13_units.npz"); c8 = np.load(SD / "c13_units_ckpt_11.npz")
    c5 = np.load(SD / "c5_units.npz")
    pairing = {"units": len(can)}
    for nm, z in (("drift_4step", n4), ("drift_8step", n8)):                # new outputs carry their units' identity
        pairing[nm + "_ids_exact"] = bool((z["ids"] == can[:, :5]).all())
    for nm, z in (("ckpt10_4step", p4), ("ckpt10_8step", p8), ("continuation_8step", c8)):   # older files: kind and light
        pairing[nm + "_kind_light"] = bool((z["kinds"] == can[:, 2]).all() and (z["night"] == can[:, 5]).all())
    for nm, z in (("continuation_4step", c4), ("c5_panel", c5)):          # these carry episode and frame
        kk = z["kind"] if "kind" in z.files else z["kinds"]
        pairing[nm + "_episode_frame_kind"] = bool((z["e"] == can[:, 0]).all() and (z["t"] == can[:, 1]).all() and (kk == can[:, 2]).all())
    assert all(pairing[k] for k in pairing if k != "units"), pairing
    ep, kinds, night = can[:, 0], can[:, 2], can[:, 5].astype(bool)
    res = {"checkpoint": str(CK), "pairing": pairing, "four_step": {}, "eight_step": {}}
    for kn, name in KINDS.items():
        for light, sel in (("day", ~night), ("night", night)):
            m = (kinds == kn) & sel
            if not m.any():
                continue
            key = f"{name}|{light}"
            res["four_step"][key] = {"units": int(m.sum()),
                                     "vs_checkpoint_10": paired(n4["A0_unguided_kept"][m], p4["A0_unguided_kept"][m], ep[m], rng),
                                     "vs_continuation": paired(n4["A0_unguided_kept"][m], c4["kept"][m], ep[m], rng)}
            res["eight_step"][key] = {"units": int(m.sum()),
                                      "vs_checkpoint_10": paired(n8["sampler_kept"][m][..., -1], p8["sampler_kept"][m][..., -1], ep[m], rng),
                                      "vs_continuation": paired(n8["sampler_kept"][m][..., -1], c8["sampler_kept"][m][..., -1], ep[m], rng)}
    g_new, g_old = (json.loads((d / f"c15_remedies{t}.json").read_text())["arms"]["A0_unguided"] for d, t in ((HERE, TAG), (SD, "")))
    guards = {"spurious_ratio": g_new["spurious_mobs_per_frame"] / g_old["spurious_mobs_per_frame"],
              "static_hud_ratio": g_new["token_error_by_region"]["hud_static"] / g_old["token_error_by_region"]["hud_static"]}
    guards["pass"] = guards["spurious_ratio"] <= 1.10 and guards["static_hud_ratio"] <= 1.5   # PLAN.md's two bars
    res["guards"] = {**guards, "new": {k: v for k, v in g_new.items() if k != "keep"}, "checkpoint_10": {k: v for k, v in g_old.items() if k != "keep"}}
    day = res["four_step"]["cows|day"]
    fell = all(day[r]["change"] <= -0.01 and day[r]["ci95_episodes"][1] < 0 for r in ("vs_checkpoint_10", "vs_continuation"))
    res["primary"] = {"deletion_fell_1pt_both": fell, "guards_pass": guards["pass"], "success": fell and guards["pass"],
                      "null": abs(day["vs_continuation"]["change"]) <= 0.005}
    tn, to = (np.load(d / f"midsignal_tokens_units{t}.npz") for d, t in ((HERE, TAG), (SD, "")))
    common = np.intersect1d(tn["units"], to["units"])                    # day cows both checkpoints' readers admit
    pick = lambda z: np.searchsorted(z["units"], common)
    trace = {}
    for nm, z in (("new", tn), ("checkpoint_10", to)):
        ix = pick(z); kept = z["kept_native"][ix]; dele = ~kept
        trace[nm] = {"keep": float(kept.mean()), "M0_same": float(z["star0_finest"][ix][..., 2].mean()),
                     "M0_all": float(z["star0_finest"][ix][..., 3].mean()),
                     "deleted_draws_mass_by_step": [float(z[f"native{i}"][ix][..., 2][dele].mean()) if dele.any() else None for i in range(4)],
                     "deleted_draws": int(dele.sum())}
    res["token_chain_common_units"] = {"units": int(len(common)), **trace}
    terms = [json.loads(l) for l in (HERE / "drift_terms.jsonl").read_text().splitlines()]
    q = np.array_split(np.arange(len(terms)), 4)
    res["drift_terms_by_quarter_training_diagnostic"] = [{k: float(np.mean([terms[i][k] for i in ix])) for k in ("drift_4", "drift_6", "flow")} for ix in q]
    mon = lambda run: [r for r in map(json.loads, (run / "log.jsonl").read_text().splitlines()) if "monitor" in r][-1]["monitor"]["g1"]
    res["monitor_g1"] = {"new": mon(CK.parent), "continuation": mon(SD / "c14_run"), "checkpoint_10": mon(HERE.parent / "20261116_p15/run")}
    (HERE / "readings.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]]()
