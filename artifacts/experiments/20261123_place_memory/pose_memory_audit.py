"""Price the effect of the archived motion reader on lookup from a place store."""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S2 = HERE.parent / "20261119_s2"


def keys(pos):
    row, col = np.indices((7, 9)).reshape(2, -1)
    y = pos[:, 0, None] + row - 3 + 20
    x = pos[:, 1, None] + col - 4 + 20
    assert ((0 <= y) & (y < 41) & (0 <= x) & (x < 41)).all()
    return y * 41 + x


def read(truth, estimate, terminated):
    """Query newly entering cells before writing the current rendered observation."""
    assert truth.shape == estimate.shape
    alive = np.c_[np.ones(len(terminated), dtype=bool),
                  np.ones(len(terminated), dtype=bool), ~np.maximum.accumulate(terminated[:, :-1], axis=1)]
    truth = truth.astype(int) - truth[:, :1]
    estimate = estimate.astype(int) - estimate[:, :1]
    lookup = np.full((len(truth), 41 * 41), -1, dtype=np.int16)
    observed = np.zeros_like(lookup, dtype=bool)
    rows = np.arange(len(truth))[:, None]
    previous = None
    per_step = []
    for t in range(truth.shape[1]):
        actual_key = keys(truth[:, t])
        estimated_key = keys(estimate[:, t])
        mask = alive[:, t]
        if t:
            entering = ((actual_key[:, :, None] != previous[:, None, :]).all(-1)) & mask[:, None]
            seen = observed[rows, actual_key]
            stored = lookup[rows, estimated_key]
            revisit = entering & seen
            per_step.append({
                "step": t,
                "live_states": int(mask.sum()),
                "same_position": int(((truth[:, t] == estimate[:, t]).all(-1) & mask).sum()),
                "revisits": int(revisit.sum()),
                "correct": int(((stored == actual_key) & revisit).sum()),
                "wrong_place": int(((stored != actual_key) & (stored >= 0) & revisit).sum()),
                "missing": int(((stored < 0) & revisit).sum()),
                "first_visits": int((entering & ~seen).sum()),
                "false_memory_on_first_visit": int(((stored >= 0) & entering & ~seen).sum()),
            })
        observed[rows[mask], actual_key[mask]] = True
        lookup[rows[mask], estimated_key[mask]] = actual_key[mask]
        previous = actual_key
    keys_to_sum = ("revisits", "correct", "wrong_place", "missing", "first_visits", "false_memory_on_first_visit")
    return {"overall": {k: sum(r[k] for r in per_step) for k in keys_to_sum}, "per_step": per_step}


def main():
    with np.load(S2 / "bank_competent.npz") as b:
        take = b["kind"] == 0
        plans = b["plans"][take]
    with np.load(S2 / "oracle_competent.npz") as z:
        true = z["lab_pos"][take, 8:].reshape(-1, 17, 2)
        readback = z["rd_pos"][take, 8:].reshape(-1, 17, 2)
        done = z["lab_done"][take, 8:].reshape(-1, 16)
    direction = np.array([(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)])
    action_steps = direction[np.where(plans <= 4, plans, 0)]
    action_positions = np.insert(np.cumsum(action_steps, axis=1), 0, 0, axis=1)
    action_positions = np.repeat(action_positions, 8, axis=0)
    visual_steps = np.diff(readback.astype(int), axis=1)
    intended = np.repeat(action_steps, 8, axis=0)
    # The saved reader exposes only its chosen shift. Fall back to zero if it contradicts the action.
    permitted = (visual_steps == intended).all(-1, keepdims=True)
    constrained = np.insert(np.cumsum(np.where(permitted, visual_steps, 0), axis=1), 0, 0, axis=1)
    report = {"starts": len(plans), "trajectories": len(true), "true_pose": read(true, true, done),
              "visual_reader": read(true, readback, done), "attempted_action_only": read(true, action_positions, done),
              "action_constrained_visual_reader": read(true, constrained, done)}
    assert report["true_pose"]["overall"]["correct"] == report["true_pose"]["overall"]["revisits"]
    (HERE / "pose_memory_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, value in report.items():
        if isinstance(value, dict):
            print(name, value["overall"], "step 16", value["per_step"][-1])


if __name__ == "__main__":
    main()
