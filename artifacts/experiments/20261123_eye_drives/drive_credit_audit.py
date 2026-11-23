"""Price sensor static and positive-only policy credit using existing Saccade banks."""

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
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
RETURNS = importlib.import_module("artifacts.experiments.20261119_s2.first_action_returns")
STATIC = HERE.parent / "20261102_p1f15_readiness"


@torch.no_grad()
def score_repeats(model, cfg, frames, context, t, live_periphery):
    frames = torch.from_numpy(frames).to(S2.DEV)
    draws = len(frames)
    with TR.autocast(S2.DEV, cfg):
        if live_periphery:
            fix = torch.full((draws, 2), 4, device=S2.DEV, dtype=torch.long)
            percept = model.perception(frames, fix)
            p, v = percept.p, percept.v
        else:
            p, v = (context[k][t].to(S2.DEV).float().unsqueeze(0).expand(draws, *context[k].shape[1:])
                    for k in ("p", "v"))
        bp = context["bp"][t].to(S2.DEV).float().expand(draws, -1)
        cells = torch.arange(81, device=S2.DEV).expand(draws, -1)
        mean, scale = model.pixels(model.tsp.predict(bp, p, v, cells), v, cells)
    pixels = frames.reshape(draws, 9, 7, 9, 7, 3).permute(0, 1, 3, 2, 4, 5).reshape(draws, 81, 147)
    score = M.pixel_nll(mean, scale, pixels).mean(-1)
    return score[:, C.WIN.to(S2.DEV)].mean(-1).cpu().numpy()


def describe_repeats(scores):
    scores = np.asarray(scores)
    top = scores.argmax(-1)
    switches = np.stack([top[:, i] != top[:, j] for i, j in combinations(range(scores.shape[1]), 2)], axis=1)
    span = np.ptp(scores, axis=1)
    return {
        "scenes": len(scores),
        "repeats_per_scene": scores.shape[1],
        "mean_nll_nats_per_pixel": float(scores.mean()),
        "mean_window_span_nats_per_pixel": float(span.mean()),
        "max_window_span_nats_per_pixel": float(span.max()),
        "highest_nll_window_switch_fraction": float(switches.mean()),
        "scenes_with_change_in_highest_nll_window": int(switches.any(1).sum()),
    }


def sensor_static():
    model, cfg = S2.AI.load(S2.P15)
    model.requires_grad_(False).eval()
    labels = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 12)
    history = S.encode(model, labels, cfg)
    results = {}
    for name, directory in (("dark", STATIC), ("night_onset", STATIC / "static_onset_initial")):
        with np.load(directory / "static_repeats.npz") as data:
            rows, frames = data["rows"], data["frames"]
        results[name] = {"rows": rows.tolist()}
        for key, live_periphery in (("fixed_recorded_periphery", False), ("repeat_specific_periphery", True)):
            scores = [score_repeats(model, cfg, image, history[e], t, live_periphery)
                      for (e, t), image in zip(rows, frames)]
            results[name][key] = describe_repeats(scores)
            results[name][key]["scores"] = np.asarray(scores).tolist()
    return results


def positive_replay():
    with np.load(S2.HERE / "bank_competent.npz") as bank:
        do = np.flatnonzero(bank["kind"] == 1)
        noop = do + 1
        rows, plans = bank["rows"], bank["plans"]
        np.testing.assert_array_equal(rows[do], rows[noop])
        np.testing.assert_array_equal(plans[do, 1:], plans[noop, 1:])
    with np.load(S2.HERE / "oracle_competent.npz") as oracle:
        values = RETURNS.returns(oracle["lab_reward"], oracle["lab_done"])
    a, b = np.split(values, 2, axis=1)
    out = {}
    for horizon in (1, 4, 15):
        qa = a[:, :, horizon - 1].mean(1)
        qb = b[:, :, horizon - 1].mean(1)
        diff_a, diff_b = qa[do] - qa[noop], qb[do] - qb[noop]
        worse, better = np.where(diff_a > 0, noop, do), np.where(diff_a > 0, do, noop)
        baseline = (qa[do] + qa[noop]) / 2
        consensus = (diff_a * diff_b > 0) & (np.abs(diff_a) > RETURNS.EPS) & (np.abs(diff_b) > RETURNS.EPS)
        use = consensus & (qb[worse] < baseline - RETURNS.EPS) & (qb[better] > baseline + RETURNS.EPS)
        bad = b[worse[use], :, horizon - 1] - baseline[use, None]
        good = b[better[use], :, horizon - 1] - baseline[use, None]
        bad_positive = np.maximum(bad, 0)
        good_positive = np.maximum(good, 0)
        out[str(horizon)] = {
            "starts": int(use.sum()),
            "episodes": int(len(np.unique(rows[do[use], 0]))),
            "worse_action_positive_fraction": float((bad > RETURNS.EPS).mean()),
            "worse_action_starts_with_positive_sample_fraction": float((bad > RETURNS.EPS).any(1).mean()),
            "share_of_positive_weight_on_worse_action": float(bad_positive.sum() / (bad_positive + good_positive).sum()),
            "mean_worse_advantage": float(bad.mean()),
            "mean_better_advantage": float(good.mean()),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    results = {"positive_replay": positive_replay()}
    print(json.dumps(results, indent=2), flush=True)
    if not args.replay_only:
        results["sensor_static"] = sensor_static()
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
