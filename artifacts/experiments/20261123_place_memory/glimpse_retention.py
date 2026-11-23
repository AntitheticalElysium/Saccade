"""Follow a one-crop intervention through P1.5's recurrent belief on shared real suffixes."""

import importlib
import json
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "20261102_p1f15_readiness"
INFO = importlib.import_module("artifacts.experiments.20261123_look_information.look_information")
DEV = INFO.DEV
LAGS = (0, 1, 4, 8, 16)
WINDOWS = np.array([(r, c) for r in (1, 3, 5, 7) for c in (1, 4, 7)])


def embed(model, enc, sl):
    percept = M.Percept(*(enc[k][sl].to(DEV)[None].float() for k in ("p", "g", "v")),
                        enc["fix"][sl].to(DEV)[None])
    return S.embed(model, percept, enc["reads"][sl].to(DEV)[None].float())


@torch.no_grad()
def forecasts(model, b):
    h = model.outcome.features(b.float(), None)
    p = model.outcome.bin_logp(h).exp()
    bins = model.outcome.bin_value
    cont = model.outcome.cont_logit(h.unsqueeze(-2).expand(*h.shape[:-1], len(bins), h.shape[-1]),
                                   bins.expand(*h.shape[:-1], -1))
    return torch.stack((p[..., bins > 0.5].sum(-1), p[..., bins < -0.025].sum(-1),
                        (p * torch.sigmoid(-cont.float())).sum(-1)), -1)


@torch.no_grad()
def intervention(model, cfg, enc, t, images):
    frames = torch.from_numpy(images)
    blurry = INFO.DV.blurred(frames, model.perception.acuity)
    views = torch.stack((frames, blurry), 1)
    repeats, nwin = len(frames), len(WINDOWS)
    n = repeats * 2 * nwin
    fix_cpu = torch.as_tensor(WINDOWS).repeat(repeats * 2, 1)
    inputs = views.flatten(0, 1).repeat_interleave(nwin, 0)
    g = INFO.DV.glimpses(model, cfg, inputs, fix_cpu).float()[:, None]
    keys = ("reward", "terminated", "action")

    with TR.autocast(DEV, cfg):
        prior = model.belief.tokens(embed(model, enc, slice(0, t)),
                                    *(enc[k][:t].to(DEV)[None] for k in keys))
        _, carry = model.belief(prior, model.belief.initial(1, DEV))
        look = slice(t, t + 1)
        p, v, read = (enc[k][look].to(DEV)[None].float().expand(n, *enc[k][look].shape)
                      for k in ("p", "v", "reads"))
        now = S.embed(model, M.Percept(p, g, v, fix_cpu.to(DEV)[:, None]), read)
        later = embed(model, enc, slice(t + 1, t + max(LAGS) + 1)).expand(n, -1, -1)
        obs = torch.cat((now, later), 1)
        raw = [enc[k][t:t + max(LAGS) + 1].to(DEV)[None].expand(n, -1) for k in keys]
        y, _ = model.belief(model.belief.tokens(obs, *raw), INFO.rep_carry(carry, n))
    bp = y[:, 1::2].float().reshape(repeats, 2, nwin, -1, y.shape[-1])[:, :, :, LAGS]
    pred = forecasts(model, bp)
    sensor_pairs = list(combinations(range(repeats), 2))
    resolution_state = (bp[:, 0] - bp[:, 1]).norm(dim=-1).mean((0, 1))
    noise_state = torch.stack([(bp[i, 0] - bp[j, 0]).norm(dim=-1) for i, j in sensor_pairs]).mean((0, 1))
    resolution_forecast = (pred[:, 0] - pred[:, 1]).abs().mean((0, 1))
    noise_forecast = torch.stack([(pred[i, 0] - pred[j, 0]).abs() for i, j in sensor_pairs]).mean((0, 1))
    return dict(zip(("resolution_state_l2", "static_state_l2", "resolution_forecast", "static_forecast"),
                    [x.cpu().tolist() for x in (resolution_state, noise_state, resolution_forecast, noise_forecast)]))


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    model, cfg = INFO.S2.AI.load(INFO.S2.P15)
    model.requires_grad_(False).eval()
    labels = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 12)
    histories = S.encode(model, labels, cfg)
    result = {"lags": LAGS, "forecast_columns": ("positive_reward", "negative_reward", "termination"), "windows": WINDOWS.tolist()}
    for name, directory in (("dark", STATIC), ("night_onset", STATIC / "static_onset_initial")):
        with np.load(directory / "static_repeats.npz") as z:
            rows, frames = z["rows"], z["frames"]
        cases = [{"episode": int(e), "t": int(t), **intervention(model, cfg, histories[e], int(t), x)}
                 for (e, t), x in zip(rows, frames) if t + max(LAGS) < int(labels["ep_T"][e])]
        summary = {k: np.mean([c[k] for c in cases], axis=0).tolist()
                   for k in cases[0] if k not in ("episode", "t")}
        result[name] = {"scenes": len(cases), "mean": summary, "per_scene": cases}
        print(name, result[name]["mean"], flush=True)
    (HERE / "glimpse_retention.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
