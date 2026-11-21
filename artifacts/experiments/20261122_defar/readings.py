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


def paired(new, old, rng):
    """Deletion per unit (1 - keep over draws), the paired change and its unit-bootstrap 95% interval."""
    dn, do = 1 - new.mean(1), 1 - old.mean(1)
    boot = [(lambda ix: (dn[ix] - do[ix]).mean())(rng.integers(0, len(dn), len(dn))) for _ in range(4000)]
    return {"new": float(dn.mean()), "ref": float(do.mean()), "change": float((dn - do).mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist()}


def compare():
    rng = np.random.default_rng(0)
    n4 = np.load(HERE / f"c15_units{TAG}.npz"); p4 = np.load(SD / "c15_units.npz"); c4 = np.load(SD / "c14_calib_ckpt11.npz")
    n8 = np.load(HERE / "c13_units_ckpt_11.npz"); p8 = np.load(SD / "c13_units.npz"); c8 = np.load(SD / "c13_units_ckpt_11.npz")
    for z in (p4, n8, p8, c8):
        assert (z["kinds"] == n4["kinds"]).all() and (z["night"] == n4["night"]).all(), "unit order differs"
    assert (c4["kind"] == n4["kinds"]).all() and (c4["night"] == n4["night"]).all()
    res = {"checkpoint": str(CK), "four_step": {}, "eight_step": {}}
    for kn, name in KINDS.items():
        for light, sel in (("day", ~n4["night"]), ("night", n4["night"])):
            m = (n4["kinds"] == kn) & sel
            if not m.any():
                continue
            key = f"{name}|{light}"
            res["four_step"][key] = {"units": int(m.sum()),
                                     "vs_checkpoint_10": paired(n4["A0_unguided_kept"][m], p4["A0_unguided_kept"][m], rng),
                                     "vs_continuation": paired(n4["A0_unguided_kept"][m], c4["kept"][m], rng)}
            res["eight_step"][key] = {"units": int(m.sum()),
                                      "vs_checkpoint_10": paired(n8["sampler_kept"][m][..., -1], p8["sampler_kept"][m][..., -1], rng),
                                      "vs_continuation": paired(n8["sampler_kept"][m][..., -1], c8["sampler_kept"][m][..., -1], rng)}
    day = res["four_step"]["cows|day"]
    res["primary"] = {"success": all(day[r]["change"] <= -0.01 and day[r]["ci95"][1] < 0 for r in ("vs_checkpoint_10", "vs_continuation")),
                      "null": abs(day["vs_continuation"]["change"]) <= 0.005}
    g_new, g_old = (json.loads((d / f"c15_remedies{t}.json").read_text())["arms"]["A0_unguided"] for d, t in ((HERE, TAG), (SD, "")))
    res["guards_vs_checkpoint_10"] = {"new": {k: v for k, v in g_new.items() if k != "keep"}, "checkpoint_10": {k: v for k, v in g_old.items() if k != "keep"}}
    t_new, t_old = (json.loads(p.read_text())["1_posterior_vs_chain"] for p in (HERE / f"midsignal_tokens{TAG}.json", SD / "midsignal_tokens.json"))
    res["token_chain"] = {"new": t_new, "checkpoint_10": t_old}
    terms = [json.loads(l) for l in (HERE / "drift_terms.jsonl").read_text().splitlines()]
    q = np.array_split(np.arange(len(terms)), 4)
    res["drift_terms_by_quarter"] = [{k: float(np.mean([terms[i][k] for i in ix])) for k in ("drift_4", "drift_6", "flow")} for ix in q]
    mon = lambda run: [r for r in map(json.loads, (run / "log.jsonl").read_text().splitlines()) if "monitor" in r][-1]["monitor"]["g1"]
    res["monitor_g1"] = {"new": mon(CK.parent), "continuation": mon(SD / "c14_run"), "checkpoint_10": mon(HERE.parent / "20261116_p15/run")}
    (HERE / "readings.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]]()
