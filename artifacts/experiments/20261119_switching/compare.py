"""The switching arms' S2 stage-1 readings beside P1.5's (CPU, seconds): mobs per frame by step, 4-step persistence and
the S3 blocker by kind, interior appearances and in-view vanishing per step (the churn signature of the first arm), the
guards (energy / food / drink drop curves, death beside zombies, N5, return, internal consistency) and the held-out
one-step monitor. Every number comes from S2's saved files (gates_*.json, model_*.npz) or the arm's switch_ft.json.

    python -m artifacts.experiments.20261119_switching.compare TAG [TAG ...]     # e.g. sw_rec_2000 sw_rec2_2000 sw_rec2_4000
"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S2 = HERE.parent / "20261119_s2"
KINDS = ("zombies", "cows", "skeletons", "arrows")


def per_step(grid, done):
    """Mobs per frame at steps 0 / 1 / 4 / 8 / 16 over live frames (a frame counts until the transition that ends it)."""
    N, F, S = grid.shape[:3]
    ended = np.concatenate([np.zeros((N, F, 1), bool), np.cumsum(done, 2) > 0], 2)[:, :, :S]
    live = np.concatenate([np.ones((N, F, 1), bool), ~ended[:, :, :-1]], 2)
    out = {}
    for k, n in enumerate(KINDS, 1):
        c = (grid == k).sum((-1, -2)).astype(float)
        out[n] = [float(c[:, :, s][live[:, :, s]].mean()) for s in (0, 1, 4, 8, 16)]
    out["all"] = list(np.sum([out[n] for n in KINDS], 0))
    return out


def main(tags):
    o = np.load(S2 / "oracle.npz")
    rows = {"simulator": (None, per_step(o["rd_grid"][:, 8:16], o["lab_done"][:, 8:16]))}
    rows["P1.5"] = (json.load(open(S2 / "gates.json")), per_step(*(np.load(S2 / "model.npz")[k] for k in ("m_grid", "m_done"))))
    for t in tags:
        z = np.load(S2 / f"model_{t}.npz")
        rows[t] = (json.load(open(S2 / f"gates_{t}.json")), per_step(z["m_grid"], z["m_done"]))
    names = list(rows)
    w = max(map(len, names)) + 2
    print("mobs per frame, steps 0 / 1 / 4 / 8 / 16")
    for kind in ("all", *KINDS):
        print(" ", kind)
        for n in names:
            print(f"    {n:{w}s}", " ".join(f"{v:.3f}" for v in rows[n][1][kind]))
    g = {n: r[0] for n, r in rows.items() if r[0] is not None}
    print("4-step persistence (simulator) and count change against the simulator; * = S3 blocker fires")
    for k in KINDS:
        ref = g["P1.5"]["S3_bookkeeping_blocker"][k]["persistence4_reference"]
        print(f"  {k:10s} ({ref:.3f})", " | ".join(
            f"{n}: {x['S3_bookkeeping_blocker'][k]['persistence4_model']:.3f} {x['S3_bookkeeping_blocker'][k]['count_relative']:+.2f}"
            f"{'*' if x['S3_bookkeeping_blocker'][k]['fires'] else ''}" for n, x in g.items()))
    print("appearances / vanishings per step, model (simulator); '-' = the gate does not reject")
    gates = ("appear_interior_cows_day", "appear_interior_cows_night", "appear_interior_zombies_night", "appear_interior_skeletons_day",
             "vanish_cows_day", "vanish_cows_night", "vanish_zombies_night", "appear_edge_cows_day", "appear_edge_zombies_night")
    for e in gates:
        vals = [x["L1_model_vs_B"]["event_gates_rejecting"].get(e) for x in g.values()]
        ref = next((v[0] for v in vals if v), None)
        print(f"  {e:32s}", f"({ref:.4f})" if ref is not None else "(-)", " | ".join(
            f"{n}: {v[1]:.4f}" if v else f"{n}: -" for n, v in zip(g, vals)))
    print("guards: max |model - simulator| over the time-to-event curve")
    for c in ("energy_drop|all", "energy_drop|energy_since_last_drop_99_99", "food_drop|all", "food_drop|food_since_last_drop_20_98",
              "food_drop|food_since_last_drop_99_99", "drink_drop|drink_since_last_drop_20_98", "death|zombie_within_2",
              "damage|zombie_within_2", "wake|asleep_at_start", "unlock|all"):
        print(f"  {c:42s}", " | ".join(f"{n}: {x['L2b_time_to_event'][c]['max_abs_diff']:.3f}" for n, x in g.items()))
    print("  return per future (simulator {:.4f})".format(g["P1.5"]["L2b_return_per_future"]["reference"]),
          " | ".join(f"{n}: {x['L2b_return_per_future']['model']:.4f}" for n, x in g.items()))
    print("  N5 max |generated - real| belief HUD", " | ".join(
        f"{n}: {max(abs(v['diff']) for v in x['N5_belief_hud_exact'].values()):.4f}" for n, x in g.items()))
    print("  consistency rejects", " | ".join(f"{n}: {[k for k, v in x['L2b_internal_consistency'].items() if v['rejects']]}" for n, x in g.items()))
    print("  multivariate energy score (simulator {:.3f})".format(g["P1.5"]["L1_model_vs_B"]["energy_score"]["reference"]),
          " | ".join(f"{n}: {x['L1_model_vs_B']['energy_score']['model']:.3f}" for n, x in g.items()))
    for t in tags:
        arm, u = t.rsplit("_", 1)
        f = HERE / f"run/{arm}/switch_ft.json"
        if f.exists():
            m = json.load(open(f))["monitor"]
            print(f"  held-out one-step monitor {arm}: 0 {m['0']['g1']:.6f}", f"{u} {m[u]['g1']:.6f}" if u in m else "")


if __name__ == "__main__":
    main(sys.argv[1:])
