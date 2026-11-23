"""Measure the proposed fine-minus-blurred belief bonus on fixed-state noise repeats."""

import argparse
import importlib
import json
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from saccade import chooser as C, model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
INFO = importlib.import_module("artifacts.experiments.20261123_look_information.look_information")
STATIC = HERE.parent / "20261102_p1f15_readiness"


@torch.no_grad()
def compare(model, cfg, en, t, images):
    dev = INFO.DEV
    frames = torch.from_numpy(images)
    blurred = INFO.DV.blurred(frames, model.perception.acuity)
    views = torch.stack((frames, blurred), dim=1)
    draws, windows = len(frames), len(C.CENTRES)
    n = draws * 2 * windows
    fix = C.CENTRES.repeat(draws * 2, 1)
    inputs = views.flatten(0, 1).repeat_interleave(windows, dim=0)
    g = INFO.DV.glimpses(model, cfg, inputs, fix).float()[:, None]
    with TR.autocast(dev, cfg):
        history = M.Percept(*(en[k][:t].to(dev).float()[None] for k in ("p", "g", "v")),
                            en["fix"][:t].to(dev)[None])
        keys = ("reward", "terminated", "action")
        obs = model.belief.tokens(S.embed(model, history, en["reads"][:t].to(dev).float()[None]),
                                  *(en[k][:t].to(dev)[None] for k in keys))
        _, carry = model.belief(obs, model.belief.initial(1, dev))
        p, v, reads = (en[k][None, t:t + 1].to(dev).float().expand(n, *en[k][None, t:t + 1].shape[1:])
                       for k in ("p", "v", "reads"))
        percept = M.Percept(p, g, v, fix.to(dev)[:, None])
        tokens = model.belief.tokens(S.embed(model, percept, reads),
                                     *(en[k][t:t + 1].to(dev).expand(n, 1) for k in keys))
        output, _ = model.belief(tokens, INFO.rep_carry(carry, n))
    states = output[:, 0].float().reshape(draws, 2, windows, -1)
    bonus = (states[:, 0] - states[:, 1]).norm(dim=-1)
    pairs = list(combinations(range(draws), 2))
    delta = torch.stack([(states[i, 0] - states[j, 0]).norm(dim=-1) for i, j in pairs])
    return bonus.cpu().numpy(), delta.cpu().numpy()


def report(bonuses, deltas):
    bonuses, deltas = np.stack(bonuses), np.stack(deltas)
    top = bonuses.argmax(-1)
    switches = np.stack([top[:, i] != top[:, j] for i, j in combinations(range(top.shape[1]), 2)], 1)
    return {
        "scenes": len(top),
        "mean_fine_minus_blur_bonus": float(bonuses.mean()),
        "mean_bonus_span_per_window": float(np.ptp(bonuses, axis=1).mean()),
        "mean_belief_displacement_from_static_only": float(deltas.mean()),
        "top_bonus_switches": int(switches.sum()),
        "repeat_pairs": int(switches.size),
        "scenes_with_any_switch": int(switches.any(1).sum()),
        "bonus_by_scene_repeat_window": bonuses.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    model, cfg = INFO.S2.AI.load(INFO.S2.P15)
    model.requires_grad_(False).eval()
    labels = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 12)
    history = S.encode(model, labels, cfg)
    result = {}
    for name, source in (("dark", STATIC), ("night_onset", STATIC / "static_onset_initial")):
        with np.load(source / "static_repeats.npz") as saved:
            rows, frames = saved["rows"], saved["frames"]
        bonuses, deltas = zip(*(compare(model, cfg, history[e], t, copies)
                                for (e, t), copies in zip(rows, frames)))
        result[name] = {"rows": rows.tolist(), **report(bonuses, deltas)}
        print(name, {k: v for k, v in result[name].items() if k != "bonus_by_scene_repeat_window"})
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
