"""Stage A6 of the switching dissection (PLAN.md): how the learned fixation developed across the learned-gaze worlds.
The F2 review's trace reading (`20261104_p1f2_review/trace.py`) runs unchanged on each world's executed training gaze
(`run/gaze.bin`), with only its run and output folders redirected. It gives crop coverage of each target class
against the paired uniform draw on the labelled training subset, with physical-episode CIs, plus dwell and the
logged C / S / entropy clock. The F2 rerun must reproduce the review's archived TRACE.json.

    JAX_PLATFORMS=cpu python -m artifacts.experiments.20261120_switching_dissection.gaze_dev [WORLD ...]
"""
import importlib, json, sys
from pathlib import Path

T = importlib.import_module("artifacts.experiments.20261104_p1f2_review.trace")
HERE = Path(__file__).resolve().parent
WORLDS = {"F2": "20261103_p1f2", "F3": "20261105_p1f3",                     # F2.1 (craftax_union_v1): other frames "F4": "20261106_p1f4",
          "HUD-Change": "20261109_p1_hud_change", "F5": "20261111_p1f5", "MultiStep": "20261111_p1_multistep_mamba",
          "F6": "20261113_p1f6", "F6.1": "20261114_p1f61", "P1.5": "20261116_p15"}
KEYS = ("any_entity", "cows", "zombies", "skeletons", "arrows", "changed_hud", "terrain_edit", "entering_map")


def summary():
    rows = {}
    for name in WORLDS:
        f = HERE / "gaze" / name / "TRACE.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        cov = r["coverage"]
        rows[name] = {k: (cov[k]["all"]["learned"]["mean"], cov[k]["all"]["paired_uniform"]["mean"]) for k in KEYS if k in cov}
        rows[name].update({f"entity_{g}": (cov["any_entity"][g]["learned"]["mean"], cov["any_entity"][g]["paired_uniform"]["mean"])
                           for g in ("day", "night") if g in cov["any_entity"]})
        rows[name].update({k: (cov[k][k[8:] + "_attempt"]["learned"]["mean"], cov[k][k[8:] + "_attempt"]["paired_uniform"]["mean"])
                           for k in cov if k.startswith("counter_") and k[8:] + "_attempt" in cov[k]})
        clk = r["logged_clock"].get("10", {})
        rows[name]["clock10"] = (clk.get("C"), clk.get("S"), clk.get("entropy"))
        rows[name]["same_window"] = r["temporal"]["learned"]["same_screen_window"]
    keys = sorted({k for v in rows.values() for k in v if k not in ("clock10", "same_window")})
    print("crop coverage when the target is present: learned (paired uniform)")
    for k in keys:
        print(f"  {k:24s}", " | ".join(f"{n}: {v[k][0]:.3f} ({v[k][1]:.3f})" for n, v in rows.items() if k in v))
    print("  same screen window as the previous frame:", " | ".join(f"{n}: {v['same_window']:.3f}" for n, v in rows.items()))
    print("  logged C / S / entropy, last tenth:", " | ".join(f"{n}: {v['clock10']}" for n, v in rows.items()))
    (HERE / "gaze" / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        summary()
    else:
        for name in sys.argv[1:] or WORLDS:
            T.RUN = HERE.parent / WORLDS[name]
            T.HERE = HERE / "gaze" / name; T.HERE.mkdir(parents=True, exist_ok=True)
            print("==", name, flush=True)
            T.main()
        summary()
