"""Compare last-seen tokens and world-addressed P1.5 records with oracle addresses.

This is a readout experiment on real frames. The geometry is shared; it does
not train the world, estimate robot pose, or insert memory into imagination.
"""

import argparse
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
from torch import nn

from saccade import chooser as C, data as D, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
DEV = S2.DEV
SEED = 20261123
SIDE = 64
CAP = 384
FORMATS = ("last_seen", "compact_ema", "frozen_trace", "elapsed_trace")


@torch.no_grad()
def coarse(model, cfg, episode, length):
    rows = [[(0, 1, i, 4, 4) for i in range(length)]]
    frames = D.gather([episode], rows)["frames"][0]
    p, v = [], []
    for raw in np.array_split(frames, max(1, (length + 127) // 128)):
        with TR.autocast(DEV, cfg):
            t, l = C.coarse(model.perception, torch.as_tensor(raw, device=DEV))
        p.append(t[:, :63].float().cpu().numpy())
        v.append(l.float().cpu().numpy())
    return np.concatenate(p), np.concatenate(v)


def formats(p, v, positions, terrain, a, b):
    """For each re-entering cell, retrieve only its state prior to the new view."""
    count, channels = SIDE * SIDE, p.shape[-1]
    rr, cc = np.indices((7, 9)).reshape(2, -1)
    recent = np.full(count, -1, dtype=int)
    seen = np.zeros(count, dtype=int)
    last_p = np.zeros((count, channels), np.float32)
    ema = np.zeros_like(last_p)
    last_v = np.zeros((count, v.shape[-1]), np.float32)
    last_terrain = np.zeros(count, dtype=np.int16)
    frozen = np.zeros((count, channels, len(a)), np.complex64)
    aged = np.zeros_like(frozen)
    decay = (a[None] ** np.arange(len(p))[:, None]).astype(np.complex64)
    rows = {k: [] for k in (*FORMATS, "class", "gap", "seen", "changed", "surprising")}
    for t, (y, x) in enumerate(positions):
        ys, xs = y + rr - 3, x + cc - 4
        visible = (0 <= ys) & (ys < SIDE) & (0 <= xs) & (xs < SIDE)
        key = ys[visible] * SIDE + xs[visible]
        yv, xv = rr[visible], cc[visible]
        label = terrain[t, yv, xv]
        gap = t - recent[key]
        absent = np.maximum(gap - 1, 0)
        aged_before = aged[key] * decay[absent, None]
        revisit = (recent[key] >= 0) & (gap >= 2)
        if t > 0 and revisit.any():
            dy, dx = positions[t] - positions[t - 1]
            iy, ix = (yv - dy).clip(0, 6), (xv - dx).clip(0, 8)
            surprising = label != terrain[t, iy, ix]
            k = key[revisit]
            # Light and age are shared inputs. The readouts never see the current tile.
            context = np.c_[last_v[k], np.log1p(gap[revisit]).astype(np.float32)]

            def real_imag(s):
                return np.concatenate((s.real.reshape(len(k), -1), s.imag.reshape(len(k), -1)), 1)

            for name, z in (
                ("last_seen", np.c_[last_p[k], context]),
                ("compact_ema", np.c_[ema[k], context]),
                ("frozen_trace", np.c_[real_imag(frozen[k]), context]),
                ("elapsed_trace", np.c_[real_imag(aged_before[revisit]), context]),
                ("class", label[revisit]), ("gap", gap[revisit]), ("seen", seen[k]),
                ("changed", label[revisit] != last_terrain[k]), ("surprising", surprising[revisit]),
            ):
                rows[name].append(z)
        current = p[t, visible, :, None]
        frozen[key] = a * frozen[key] + b * current
        aged[key] = a * aged_before + b * current
        ema[key] = np.where(seen[key, None] > 0, .875 * ema[key] + .125 * current[..., 0], current[..., 0])
        last_p[key] = p[t, visible]
        last_v[key] = v[t]
        last_terrain[key] = label
        recent[key] = t
        seen[key] += 1
    if not rows["class"]:
        return None
    return {k: np.concatenate(x) for k, x in rows.items()}


def sample(model, cfg, filename, size, seed, cap):
    labels = dict(np.load(S.LABELS / filename))
    offsets = np.r_[0, np.cumsum(labels["ep_T"] + 1)]
    ids = np.sort(np.random.default_rng(seed).choice(
        np.flatnonzero((labels["ep_eps"] < .99) & (labels["ep_T"] >= 64)), size, replace=False))
    episodes = S.episodes(labels)
    a, b = (x.cpu().numpy() for x in (model.record.a, model.record.b))
    assert model.record.fovea == 0
    rows = []
    for i in ids:
        n = min(cap, int(episodes[i].T) + 1)
        p, v = coarse(model, cfg, episodes[i], n)
        sl = slice(offsets[i], offsets[i] + n)
        z = formats(p, v, labels["s_pos"][sl].astype(int), labels["s_view"][sl], a, b)
        if z is not None:
            rows.append({"episode": np.full(len(z["class"]), i), **z})
    return {k: np.concatenate([e[k] for e in rows]) for k in rows[0]}, ids.tolist()


def fit_predict(train, test, name):
    torch.manual_seed(SEED)
    rng = torch.Generator(device=DEV).manual_seed(SEED)
    x, q = (torch.as_tensor(data[name], device=DEV, dtype=torch.float32) for data in (train, test))
    y = torch.as_tensor(train["class"], device=DEV, dtype=torch.long)
    mu, sd = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-5)
    x, q = (z.sub(mu).div(sd) for z in (x, q))
    probe = nn.Sequential(nn.Linear(x.shape[1], 96), nn.GELU(), nn.Linear(96, 17)).to(DEV)
    opt = torch.optim.AdamW(probe.parameters(), lr=7e-4)
    for _ in range(18):
        for i in torch.randperm(len(x), generator=rng, device=DEV).split(256):
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(probe(x[i]), y[i]).backward()
            opt.step()
    with torch.no_grad():
        return torch.cat([probe(z).argmax(-1).cpu() for z in q.split(8192)]).numpy()


def summarize(test, predictions):
    gap, seen = test["gap"], test["seen"]
    subsets = {"all": gap >= 2, "single_write": seen == 1, "surprising": test["surprising"],
               "gap_2_16": gap <= 16, "gap_17_64": (gap > 16) & (gap <= 64), "gap_gt64": gap > 64}
    out = {}
    rng = np.random.default_rng(SEED)
    for name, use in subsets.items():
        y = test["class"][use]
        if not len(y):
            continue
        accuracy = {k: p[use] == y for k, p in predictions.items()}
        eps, ix = np.unique(test["episode"][use], return_inverse=True)
        n = np.bincount(ix)
        resample = rng.integers(len(eps), size=(3000, len(eps)))
        intervals = {}
        for ref in ("last_seen", "compact_ema"):
            difference = accuracy["frozen_trace"].astype(float) - accuracy[ref]
            d = np.bincount(ix, weights=difference)
            intervals[ref] = np.quantile(d[resample].sum(1) / n[resample].sum(1), (.025, .975)).tolist()
        out[name] = {"cells": len(y), "episodes": len(eps), "accuracy": {k: float(v.mean()) for k, v in accuracy.items()},
                     "frozen_minus_baseline_ci95": intervals}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    model, cfg = S2.AI.load(S2.P15)
    model.requires_grad_(False).eval()
    counts = (6, 8) if args.smoke else (48, 64)
    cap = 192 if args.smoke else CAP
    tr, ti = sample(model, cfg, "s0_train400_labels.npz", counts[0], SEED, cap)
    ev, vi = sample(model, cfg, "s0_dev_labels.npz", counts[1], SEED + 1, cap)
    print("calibration and held-out revisit cells", len(tr["class"]), len(ev["class"]), flush=True)
    predictions = {k: fit_predict(tr, ev, k) for k in FORMATS}
    a = model.record.a.abs().cpu().numpy()
    impulse = {i: [float((a ** i).min()), float((a ** i).max())] for i in (16, 64, 256)}
    result = {"checkpoint": str(S2.P15), "train_episodes": ti, "dev_episodes": vi, "cap": cap,
              "train_cells": len(tr["class"]), "test_cells": len(ev["class"]), "changed": int(ev["changed"].sum()),
              "mode_amplitude_ranges": impulse, "record_width": model.record.dim,
              "results": summarize(ev, predictions)}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
