"""What a dreamed step commits to: the two checks of PLAN.md (2026-11-22).

Check 1 (real frames): how much of the outcome the committed (pre-look) state holds, and the contradiction rate a
perfect world would show if the glimpse were drawn without the committed outcome, 2p(1 - p).
Check 2 (P1.5's dreams): one-step dreams from real histories, the coarse view drawn first, the look set to the health
window (7, 1), the glimpse drawn there; the committed outcome (before-crop probe on the generated coarse view) against
what the coarse view and the fine crop show.

    JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261122_dream_commitment.commitment_checks [--smoke]
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
from torch.nn import functional as F

from saccade import data as D, model as M, train as TR
from saccade.eval import gates as G, s1 as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
AUD = importlib.import_module("artifacts.experiments.20261119_s2.dream_outcome_audit")
DEV = S2.DEV
SEED = 20261122
LOOK = (7, 1)                                  # rows 6-8, cols 0-2: health (cell 63) is the window's cell 3
IN_WINDOW = 3
ATLAS = ROOT / "artifacts/experiments/20261104_learned_fine_candidates/hud_atlas.npy"
DRAWS, BATCH = 4, 128
BINS = [0, .01, .05, .2, .5, .8, .95, .99, 1.0000001]
CODEX_AP = {"positive_reward": 0.9749, "negative_reward": 0.9665, "termination": 0.9999}


def fit_probe(train, crop_visible):
    """AUD.fit_predict's fit exactly (seeds, normalisation, net, schedule), returning the predictor instead of scores."""
    torch.manual_seed(SEED)
    rng = torch.Generator(device=DEV).manual_seed(SEED + 1)
    coarse, fine, labels = (x.to(DEV, torch.float32) for x in train)
    stats = [(x.mean(0), x.std(0).clamp_min(1e-3)) for x in (coarse, fine)]
    norm = lambda x, s: (x.to(DEV, torch.float32) - s[0]) / s[1]
    coarse, fine = norm(coarse, stats[0]), norm(fine, stats[1])
    if not crop_visible:
        fine.zero_()
    x = torch.cat((coarse, fine), dim=-1)
    net = nn.Sequential(nn.Linear(x.shape[1], 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3)).to(DEV)
    with torch.no_grad():
        net[-1].bias.copy_(labels.mean(0).clamp(1e-5, 1 - 1e-5).logit())
    optimizer = torch.optim.AdamW(net.parameters(), lr=7e-4)
    for _ in range(16):
        ids = torch.randperm(len(x), generator=rng, device=DEV)
        for ix in ids.split(1024):
            optimizer.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(net(x[ix]), labels[ix]).backward()
            optimizer.step()
    net.eval()

    @torch.no_grad()
    def predict(c, f):
        c, f = norm(c, stats[0]), norm(f, stats[1])
        if not crop_visible:
            f = torch.zeros_like(f)
        return torch.cat([net(torch.cat((a, b), -1)).sigmoid().cpu() for a, b in zip(c.split(8192), f.split(8192))])
    return predict


def feature_rows(model_inputs):
    """The probe's coarse and fine feature blocks from (bp, p, v, reads, prev_action, g, fix), as AUD.features."""
    bp, p, v, reads, act, g, fix = model_inputs
    coarse = torch.cat((bp.half(), p.flatten(1).half(), v.half(), reads.half(), F.one_hot(act, 17).half()), -1)
    fine = torch.cat((g.flatten(1).half(), F.one_hot(fix, 9).flatten(1).half()), -1)
    return coarse, fine


def index_rows(episodes):
    """(episode position, frame j) for every transition row of AUD.features, in its order."""
    return np.array([(i, j) for i, r in enumerate(episodes) for j in range(1, len(r["reward"]))])


@torch.no_grad()
def fine_health(model, cfg, g, v, fix):
    """P1.5's own pixel decoder on the health cell's glimpse token -> nearest exact renderer patch (health 0-9)."""
    atlas = torch.from_numpy(np.load(ATLAS)[0].reshape(10, -1)).float().to(DEV)
    with TR.autocast(DEV, cfg):
        cells = M.window(fix)
        mean, _ = model.pixels(model.tsp.reconstruct(g, v, cells), v, cells)
    pix = (mean[:, IN_WINDOW].float().clamp(-1, 1) + 1) * 127.5
    return (pix[:, None] - atlas[None]).square().mean(-1).argmin(-1).cpu()


@torch.no_grad()
def coarse_health(R, p, v):
    o = R["hud"](S2.hud_x(p, v).reshape(-1, 296).half())
    return o["hud"].reshape(len(p), 16, 10).argmax(-1)[:, 0].cpu()


@torch.no_grad()
def native_negative(model, cfg, bp, p, g, v, fix, reads):
    """The native reactive head (crop included): P(reward < -0.025)."""
    with TR.autocast(DEV, cfg):
        emb = S.embed(model, M.Percept(p[:, None], g[:, None], v[:, None], fix[:, None]), reads[:, None])
        prob = model.outcome.bin_logp(model.outcome.features(bp[:, None], emb)[:, 0]).exp()
    return prob[:, model.outcome.bin_value < -0.025].sum(-1).float().cpu()


@torch.no_grad()
def real_glimpses(model, cfg, ep):
    """Every frame of the episode seen at the health window (perception on the stored frames): g, v, fix per frame."""
    c = TR.tensors(D.gather([ep], [[(0, 1, t, *LOOK) for t in range(ep.T + 1)]]), DEV)
    with TR.autocast(DEV, cfg):
        parts = [model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]) for a in range(0, ep.T + 1, 512)]
    return torch.cat([q.g[0] for q in parts]).float(), torch.cat([q.v[0] for q in parts]).float(), c["fixation"][0]


@torch.no_grad()
def dream(model, cfg, enc, e, js, seed):
    """One-step dreams after real frame j - 1 with the recorded action, coarse view first, the look set to LOOK."""
    P, den, en = cfg["data"]["context_prefix"], model.denoiser, enc[e]
    jj = np.repeat(np.asarray(js), DRAWS)
    B = len(jj)
    rin = S.window(model, enc, [(e, int(j)) for j in jj], P)            # record through the previous frame, frames j-P..j
    take = lambda key, off: torch.stack([en[key][int(j) + off] for j in jj]).to(DEV)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    cache = den.new_cache()
    with TR.autocast(DEV, cfg):
        for dt in range(-P, 0):
            cache = M.context_pass(den, tuple(take(k, dt).float()[:, None] for k in ("p", "g", "v")),
                                   take("prev_action", dt)[:, None],
                                   S.cond(model.belief.memory(take("bp", dt).float())[:, None], rin[:, P + dt][:, None]),
                                   take("fix", dt)[:, None], cache, cfg, gen)
        mem = S.cond(model.belief.memory(take("bp", 0).float())[:, None], rin[:, P][:, None])
        look = torch.tensor(LOOK, device=DEV).expand(B, 1, 2)
        dims = [tuple(en[k].shape[1:]) for k in ("p", "g", "v")]
        (p, g, v), fix = M.sample(den, dims, take("prev_action", 0)[:, None], mem, take("fix", 0)[:, None], cache, cfg,
                                  gen, choose=lambda p_, v_: look)
        p, g, v, fix = p[:, 0].float(), g[:, 0].float(), v[:, 0].float(), fix[:, 0]
        reads, _, _ = S.history(model, p, state=rin[:, P])
    return jj, (take("bp", 0).float(), p, v, reads.float(), take("prev_action", 0), g, fix), rin[:, P]


def calibration(p, y):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p >= lo) & (p < hi)
        out.append({"bin": [lo, min(hi, 1.0)], "n": int(m.sum()), "predicted": float(p[m].mean()) if m.any() else None,
                    "observed": float(y[m].mean()) if m.any() else None})
    return out


def ideal(p):
    """Perfect world, glimpse drawn without the committed outcome: expected disagreement 2p(1 - p), and the share of
    committed events an unconditioned glimpse contradicts, E[p(1 - p)] / E[p]."""
    p = p.astype(float)
    return {"n": int(len(p)), "expected_contradiction": float(np.mean(2 * p * (1 - p))),
            "share_of_committed_contradicted": float(np.sum(p * (1 - p)) / max(np.sum(p), 1e-12)),
            "committed_rate": float(p.mean())}


def disagreement(p, d):
    p, d = p.astype(float), d.astype(float)
    return {"expected_disagreement": float(np.mean(p * (1 - d) + (1 - p) * d)),
            "share_of_committed_contradicted": float(np.sum(p * (1 - d)) / max(np.sum(p), 1e-12)),
            "share_of_shown_drops_uncommitted": float(np.sum((1 - p) * d) / max(np.sum(d), 1e-12)),
            "shown_drop_rate": float(d.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "commitment_checks.json")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("highest")
    model, cfg = S2.AI.load(S2.P15)
    model.requires_grad_(False).eval()
    R = S2.load_readers(model, cfg)
    train, ti = AUD.encode_sample(model, cfg, "train400", 8 if args.smoke else 128, SEED)
    test, vi = AUD.encode_sample(model, cfg, "dev", 8 if args.smoke else 160, SEED + 1)
    a, b = AUD.features(train), AUD.features(test)
    rows = index_rows(test)
    assert len(rows) == len(b[-1])
    print("encoded transitions", len(a[-1]), len(b[-1]), flush=True)
    before, after = fit_probe(a, False), fit_probe(a, True)
    pb, pa = before(b[0], b[1]), after(b[0], b[1])
    y = b[-1].numpy().astype(bool)
    res = {"checkpoint": str(S2.P15), "train_episodes": ti, "dev_episodes": vi, "look": LOOK, "draws": DRAWS}

    # ---- check 1: real frames ----
    full = rows[:, 1] >= cfg["data"]["context_prefix"]
    res["check1"] = {
        "verification": {"before_crop": AUD.scores(pb, b[-1]), "after_crop": AUD.scores(pa, b[-1]),
                         "codex_before_crop_ap": CODEX_AP},
        "calibration_before_crop": {k: calibration(pb[:, i].numpy(), y[:, i]) for i, k in enumerate(AUD.EVENTS)},
        "ideal": {k: {"all_full_context": {"before_crop": ideal(pb[full, i].numpy()), "after_crop": ideal(pa[full, i].numpy())},
                      "recorded_event": {"before_crop": ideal(pb[full & y[:, i], i].numpy()),
                                         "after_crop": ideal(pa[full & y[:, i], i].numpy())}}
                  for i, k in enumerate(AUD.EVENTS)}}
    print(json.dumps(res["check1"]["verification"], indent=1), flush=True)

    # ---- check 2: one-step dreams ----
    rng = np.random.default_rng(SEED)
    drop = np.flatnonzero(full & y[:, 1])
    other = np.flatnonzero(full & ~y[:, 1])
    other = np.sort(rng.choice(other, min(len(other), len(drop)), replace=False))
    starts = np.sort(np.r_[drop, other])
    if args.smoke:
        starts = np.sort(np.r_[drop[:6], other[:6]])
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    health = lambda i, j: int(min(L["s_body"][so[vi[i]] + j, 0], 9))
    dev_eps = S.episodes(S2.sublabels(L, np.asarray(vi)))
    col = {k: [] for k in ("row", "cohort", "h_prev", "h_true", "p_dream", "p_real", "p_after_dream", "q_native",
                           "fine_dream", "coarse_dream", "fine_real", "coarse_real")}
    record_check = []
    for i in np.unique(rows[starts, 0]):
        rs = starts[rows[starts, 0] == i]
        g_all, v_all, f_all = real_glimpses(model, cfg, dev_eps[int(i)])
        for chunk in np.array_split(rs, max(1, -(-len(rs) * DRAWS // BATCH))):
            js = rows[chunk, 1]
            jj, (bp, p, v, reads, act, g, fix), state = dream(model, cfg, test, int(i), js, SEED + 7919 * int(i) + int(js[0]))
            c, f = feature_rows((bp, p, v, reads, act, g, fix))
            r = np.repeat(chunk, DRAWS)
            col["row"].append(r); col["cohort"].append(y[r, 1])
            col["h_prev"].append(np.array([health(i, j - 1) for j in jj])); col["h_true"].append(np.array([health(i, j) for j in jj]))
            col["p_dream"].append(before(c, f)[:, 1].numpy()); col["p_after_dream"].append(after(c, f)[:, 1].numpy())
            col["p_real"].append(pb[r, 1].numpy())
            col["q_native"].append(native_negative(model, cfg, bp, p, g, v, fix, reads).numpy())
            col["fine_dream"].append(fine_health(model, cfg, g, v, fix).numpy())
            col["coarse_dream"].append(coarse_health(R, p, v).numpy())
            sel = torch.as_tensor(jj, device=DEV)
            col["fine_real"].append(fine_health(model, cfg, g_all[sel], v_all[sel], f_all[sel]).numpy())
            col["coarse_real"].append(coarse_health(R, torch.stack([test[i]["p"][j] for j in jj]).to(DEV).float(),
                                                    torch.stack([test[i]["v"][j] for j in jj]).to(DEV).float()).numpy())
            if len(record_check) < 4:                                    # record state at the branch == history's last
                j0 = int(jj[0])
                with TR.autocast(DEV, cfg):
                    last = S.history(model, test[i]["p"][:j0][None].to(DEV).float())[2]
                record_check.append(float((last[0].float() - state[0].float()).abs().max()))
        print("episode", int(i), "starts", int(sum(len(x) for x in col["row"]) // DRAWS), flush=True)
    z = {k: np.concatenate(v) for k, v in col.items()}
    np.savez_compressed(HERE / ("draws_smoke.npz" if args.smoke else "draws.npz"), **z)
    drop_fine, drop_coarse = z["fine_dream"] < z["h_prev"], z["coarse_dream"] < z["h_prev"]
    drop_fine_real, drop_coarse_real = z["fine_real"] < z["h_prev"], z["coarse_real"] < z["h_prev"]
    out = {"record_state_max_abs_diff": record_check,
           "instrument": {"fine_reader_exact_on_real": float(np.mean(z["fine_real"] == z["h_true"])),
                          "coarse_reader_exact_on_real": float(np.mean(z["coarse_real"] == z["h_true"])),
                          "fine_real_drop_vs_label": float(np.mean(drop_fine_real != z["cohort"])),
                          "coarse_real_drop_vs_label": float(np.mean(drop_coarse_real != z["cohort"]))}}
    for name, m in (("recorded_drop", z["cohort"]), ("no_recorded_drop", ~z["cohort"]), ("all", np.ones(len(z["row"]), bool))):
        ur = np.unique(z["row"][m])
        out[name] = {
            "starts": int(len(ur)), "draws": int(m.sum()),
            "committed_p": {"dream_mean": float(z["p_dream"][m].mean()), "real_mean": float(z["p_real"][m].mean()),
                            "recorded_rate": float(z["cohort"][m].mean())},
            "ideal_same_starts": ideal(pb[ur, 1].numpy()),
            "fine_vs_committed": disagreement(z["p_dream"][m], drop_fine[m]),
            "coarse_vs_committed": disagreement(z["p_dream"][m], drop_coarse[m]),
            "coarse_fine_disagreement": float(np.mean(drop_fine[m] != drop_coarse[m])),
            "native_crop_head_vs_fine": disagreement(z["q_native"][m], drop_fine[m]),
            "after_crop_probe_vs_fine": disagreement(z["p_after_dream"][m], drop_fine[m]),
            "real_reference": {"fine_real_vs_label": float(np.mean(drop_fine_real[m] != z["cohort"][m])),
                               "coarse_real_vs_label": float(np.mean(drop_coarse_real[m] != z["cohort"][m]))}}
    out["dream_ap_against_recorded"] = float(G.average_precision(z["cohort"], z["p_dream"]))
    out["real_ap_same_draws"] = float(G.average_precision(z["cohort"], z["p_real"]))
    res["check2"] = out
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["check2"], indent=1), flush=True)


if __name__ == "__main__":
    main()
