"""Audit soft place matching, temporal write coverage and moving-target labels."""

import json
from pathlib import Path

import numpy as np

from .dream_write_audit import constrain
from .offscreen_routes import FIRST, FIX, sources
from .pose_memory_audit import keys

HERE = Path(__file__).resolve().parent
S2 = HERE.parent / "20261119_s2"
LABELS = HERE.parents[1] / "eda/s0/s0_dev_labels.npz"
SIDE = 41
WORLD = 64


def neighborhood(truth, estimated, done):
    alive = np.c_[np.ones((len(done), 2), bool), ~np.maximum.accumulate(done[:, :-1], axis=1)]
    truth = truth.astype(int) - truth[:, :1]
    estimated = estimated.astype(int) - estimated[:, :1]
    known = np.full((len(truth), SIDE**2), -1, dtype=np.int16)
    observed = np.zeros(known.shape, dtype=bool)
    ri = np.arange(len(truth))[:, None]
    previous = None
    total = 0
    scores = {r: np.zeros(4, dtype=float) for r in (0, 1, 3)}
    for j in range(truth.shape[1]):
        actual_key, query_key = keys(truth[:, j]), keys(estimated[:, j])
        mask = alive[:, j]
        if j:
            entering = (actual_key[..., None] != previous[:, None, :]).all(-1)
            select = entering & observed[ri, actual_key] & mask[:, None]
            a, b = np.nonzero(select)
            wanted, q = actual_key[a, b], query_key[a, b]
            total += len(q)
            for radius, count in scores.items():
                o = np.arange(-radius, radius + 1)
                dr, dc = np.meshgrid(o, o, indexing="ij")
                d2 = (dr * dr + dc * dc).reshape(-1)
                y = q[:, None] // SIDE + dr.reshape(-1)
                x = q[:, None] % SIDE + dc.reshape(-1)
                valid = (y >= 0) & (y < SIDE) & (x >= 0) & (x < SIDE)
                entries = known[a[:, None], np.where(valid, y * SIDE + x, 0)]
                present = valid & (entries >= 0)
                correct = present & (entries == wanted[:, None])
                wrong = present & ~correct
                weight = present * np.exp(-.5 * d2)
                mass = (weight * correct).sum(1) / weight.sum(1).clip(min=1e-12)
                count += np.array([correct.any(1).sum(), wrong.any(1).sum(), mass.sum(), present.sum()])
        observed[ri[mask], actual_key[mask]] = True
        known[ri[mask], query_key[mask]] = actual_key[mask]
        previous = actual_key
    names = ("contains_correct_place", "contains_wrong_place", "gaussian_mass_correct", "mean_number_of_candidates")
    return {"revisits": total, "radius": {r: dict(zip(names, (c / total).tolist())) for r, c in scores.items()}}


def check_pose():
    with np.load(S2 / "bank_competent.npz") as b:
        rows = b["kind"] == 0
        plans = np.repeat(b["plans"][rows], 8, axis=0)
    with np.load(S2 / "oracle_competent.npz") as z:
        true = z["lab_pos"][rows, 8:].reshape(-1, 17, 2)
        read = z["rd_pos"][rows, 8:].reshape(-1, 17, 2)
        done = z["lab_done"][rows, 8:].reshape(-1, 16)
    result = {"oracle": neighborhood(true, true, done), "unconstrained_visual": neighborhood(true, read, done),
              "action_constrained_visual": neighborhood(true, constrain(read, plans), done)}
    assert result["oracle"]["radius"][0]["contains_correct_place"] == 1
    return result


def fourier():
    """Pure positional similarity if both keys use equal sine/cosine amplitudes."""
    hd = 64
    f_space = 100.0 ** (-np.arange(0, hd, 4) / hd)
    d = np.arange(WORLD)
    c = np.cos(d[:, None] * f_space).mean(1)
    reversals = [
        {"near": i, "far": j, "near_cosine": float(c[i]), "far_cosine": float(c[j])}
        for i in range(1, len(c)) for j in range(i + 1, len(c)) if c[j] > c[i]
    ]
    return {"cosine_by_axial_shift": c.tolist(), "short_axial_reversals": sum(x["near"] <= 8 for x in reversals),
            "reversals": reversals[:8]}


def gaps(labels):
    episode_ids = np.flatnonzero(labels["ep_eps"] < .99)[:120]
    offset = np.r_[0, np.cumsum(labels["ep_T"] + 1)]
    rr, cc = np.indices((7, 9))
    all_gaps = []
    for ep in episode_ids:
        pos = labels["s_pos"][offset[ep]:offset[ep + 1]].astype(int)
        recent = np.full((WORLD, WORLD), -1, dtype=np.int32)
        for t, (y, x) in enumerate(pos):
            yy, xx = y + rr - 3, x + cc - 4
            valid = (yy >= 0) & (yy < WORLD) & (xx >= 0) & (xx < WORLD)
            key = yy[valid], xx[valid]
            time = recent[key]
            if t >= 9:
                keep = (time >= 0) & (time < t - 1)
                all_gaps.extend((t - time[keep]).tolist())
            recent[key] = t
    age = np.asarray(all_gaps)
    return {"revisits": len(age), "episodes": len(episode_ids), "median": float(np.median(age)),
            "fraction_gap_le64": float((age <= 64).mean()), "p90": float(np.quantile(age, .9)),
            "uniform_phase_same_chunk_fraction": {n: float(np.maximum(1 - age / n, 0).mean()) for n in (64, 256)}}


def fine_credit():
    labels, selected, probs = sources()
    offsets = np.r_[0, np.cumsum(labels["ep_T"] + 1)]
    rng = np.random.default_rng(20261124)
    rr, cc = np.indices((7, 9))
    totals = np.zeros(5, dtype=float)
    episodes = 0
    for i, ep in enumerate(selected):
        if labels["ep_eps"][ep] >= .99:
            continue
        episodes += 1
        pos = labels["s_pos"][offsets[ep]:offsets[ep + 1]].astype(int)
        p = probs["p"][probs["episode"] == i]
        assert len(p) == max(0, len(pos) - FIRST)
        p = np.concatenate((np.full((min(FIRST, len(pos)), 49), 1 / 49), p))
        p /= p.sum(-1, keepdims=True)
        looks = np.minimum((rng.random((4, len(pos), 1)) > np.cumsum(p, axis=-1)).sum(-1), 48)
        seen = np.full((WORLD, WORLD), -1, dtype=np.int32)
        fine = np.full((4, WORLD, WORLD), -1, dtype=np.int32)
        for t, (y, x) in enumerate(pos):
            yy, xx = y + rr - 3, x + cc - 4
            valid = (yy >= 0) & (yy < WORLD) & (xx >= 0) & (xx < WORLD)
            rv, cv = rr[valid], cc[valid]
            a, b = yy[valid], xx[valid]
            old = seen[a, b]
            if t >= 9:
                use = (old >= 0) & (old < t - 1)
                last = fine[:, a[use], b[use]]
                available = last >= 0
                age = t - last
                totals += [use.sum(), last.size, available.sum(),
                           (available * np.maximum(1 - age / 64, 0)).sum(),
                           (available * np.maximum(1 - age / 256, 0)).sum()]
            seen[a, b] = t
            for draw in range(len(looks)):
                r, c = FIX[looks[draw, t]]
                window = (np.abs(rv - r) <= 1) & (np.abs(cv - c) <= 1)
                fine[draw, a[window], b[window]] = t
    n, exposures, present, grad64, grad256 = totals
    return {"episodes": episodes, "coarse_revisits": int(n), "gaze_draws_per_episode": 4,
            "fraction_with_any_prior_fine_observation": present / exposures,
            "same_chunk_fraction_among_all_revisits": {"64": grad64 / exposures, "256": grad256 / exposures},
            "same_chunk_fraction_conditioned_on_fine_write": {"64": grad64 / present, "256": grad256 / present}}


def sector(offset):
    r, c = offset
    return int((np.arctan2(r, c) + np.pi) / (2 * np.pi) * 8) % 8


def mob_targets(labels):
    rng = np.random.default_rng(20261124)
    pool = np.flatnonzero(labels["ep_eps"] < .99)
    eps = np.sort(rng.choice(pool, 360, replace=False))
    test = eps[rng.permutation(len(eps))[260:]]
    positions = labels["s_pos"].astype(int)
    offsets = np.r_[0, np.cumsum(labels["ep_T"] + 1)]
    result = {}
    for name in ("zombies", "cows"):
        observed, truth = [], []
        xy = labels[f"s_{name}_p"].astype(int)
        alive = labels[f"s_{name}_m"]
        for e in test:
            s, end = offsets[e:e + 2]
            seen = np.full(alive.shape[-1], -10**6)
            last = np.zeros_like(xy[s])
            for t in range(s, end):
                p = positions[t]
                dis = xy[t] - p
                visible = (np.abs(dis) <= (3, 4)).all(-1)
                if t >= s + 16 and (t - s - 16) % 3 == 0:
                    cand = np.flatnonzero(alive[t] & ~visible & (t - seen <= 32))
                    if len(cand):
                        k = cand[np.argmin(np.square(dis[cand]).sum(1))]
                        observed.append(last[k] - p)
                        truth.append(dis[k])
                seen[~alive[t]] = -10**6
                now = alive[t] & visible
                seen[now] = t
                last[now] = xy[t, now]
        estimate, target = np.asarray(observed), np.asarray(truth)
        disagreement = [sector(a) != sector(b) for a, b in zip(estimate, target)]
        result[name] = {"probe_targets": len(target), "direction_changed_since_last_sighting": float(np.mean(disagreement)),
                        "location_changed_since_last_sighting": float((estimate != target).any(1).mean()),
                        "median_manhattan_displacement": float(np.median(np.abs(estimate - target).sum(1)))}
    return result


def main():
    with np.load(LABELS) as z:
        labels = {k: z[k] for k in ("ep_T", "ep_eps", "s_pos", "s_zombies_p", "s_zombies_m", "s_cows_p", "s_cows_m")}
    results = {"pose": check_pose(), "fourier": fourier(), "chunks": gaps(labels),
               "fine_write_credit": fine_credit(), "mob_targets": mob_targets(labels)}
    (HERE / "place_keys_audit.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({**results, "fourier": results["fourier"]["reversals"][:5]}, indent=2))


if __name__ == "__main__":
    main()
