"""Smoke of switch_ft.py (GPU, minutes): (1) identity: at p = 0 its G1 equals train.p1_losses' G1 on the same chunk and
RNG state (the reimplemented path is P1.5's G1); (2) the switch arm (p = 0.5, both routes) for 20 real updates: the input
identity (checked inside g1), the masked share, the substitute's error by signal, peak memory, seconds per update;
(3) the control arm (the same mask, no substitute) for 2 updates: its draws (G1 levels and noise, mask, frames,
fixations) equal the switch arm's (Codex's review).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261119_switching.smoke SCRATCH_DIR
"""
import dataclasses, importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, model as M, train as TR

SW = importlib.import_module("artifacts.experiments.20261119_switching.switch_ft")


def main():
    scratch = Path(sys.argv[1]); scratch.mkdir(parents=True, exist_ok=True)
    ck = torch.load(SW.P15, map_location="cpu", weights_only=False)
    model, cfg = SW.AI.load(SW.P15)
    dev = next(model.parameters()).device
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    split = D.load_split(cfg)
    streams = D.Streams(split["train"], d["streams"], P, seed=SW.WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    carry = tuple(type(c)(*(getattr(c, f.name).to(dev) for f in dataclasses.fields(c))) for c in ck["carry"])
    cache, record = ck["cache"].to(dev), ck["record"].to(dev)
    ctl = C.Controller(model, cfg, scratch / "identity", dev); (scratch / "identity").mkdir(exist_ok=True)
    ctl.load_state(dict(ck["fixation_controller"], trace_rows=0, log_bytes=0))
    chunk = streams.next(64); ctl.prepare(chunk, streams)
    rows = np.arange(8); u = ck["state"]["update"] + 1
    boot = TR.bootstrap_rows(cfg, SW.WORLD_SEED, u)
    model.eval()
    with torch.no_grad():
        torch.manual_seed(7)
        gens = (None, torch.Generator(device=dev).manual_seed(1), torch.Generator(device=dev).manual_seed(2))
        mine = SW.g1(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows], record[rows], cfg, boot[rows], u,
                     gens, 0.0, 0, ("ctx", "mem"))
        torch.manual_seed(7)
        ref = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows], None, cfg, boot[rows], u, None,
                           record[rows], TR.gaze_rows(ctl, rows, 64))
    out = {"identity": {"switch_ft_g1": float(mine[0]), "p1_losses_g1": float(ref[0]["g1"]),
                        "abs_diff": abs(float(mine[0]) - float(ref[0]["g1"])),
                        "cache_equal": bool(torch.allclose(mine[2], ref[2], atol=1e-5)),
                        "record_equal": bool(torch.allclose(mine[3].float(), ref[3].float(), atol=1e-5))}}
    print(json.dumps(out), flush=True)
    assert out["identity"]["abs_diff"] < 1e-5 and out["identity"]["cache_equal"] and out["identity"]["record_equal"]
    del model; torch.cuda.empty_cache()
    sys.argv = ["switch_ft", str(scratch / "run"), "--updates", "20", "--reads", "20"]
    SW.main()
    res = json.loads((scratch / "run/switch_ft.json").read_text())
    sys.argv = ["switch_ft", str(scratch / "control"), "--updates", "2", "--reads", "2", "--substitute", "0"]
    SW.main()
    ctl_res = json.loads((scratch / "control/switch_ft.json").read_text())
    out["run"] = {"monitor_0": res["monitor"]["0"], "monitor_20": res["monitor"]["20"],
                  "fingerprints_equal_across_arms": res["fingerprints"] == ctl_res["fingerprints"],
                  "fingerprints": res["fingerprints"], "control_fingerprints": ctl_res["fingerprints"]}
    (SW.HERE / "smoke.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
