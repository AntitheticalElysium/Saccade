"""How available is the evidence for each Craftax-Classic rule we could vary per episode (user 11-22: scope native
parameters, hand-written variants and full Craftax)? From the DEV labels (1,009 episodes, the expert-mixed corpus):
for each rule, the event that reveals its value, its frequency per episode, the share of episodes that ever see it,
the step it first appears at, and the step a second one appears at (rates need two).

    python -m artifacts.experiments.20261122_rule_change.variation_eda
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent.parent / "eda/s0/s0_dev_labels.npz"
DO, HEALTH, FOOD, DRINK, ENERGY = 5, 0, 1, 2, 3


def main():
    L = dict(np.load(LABELS)); T = L["ep_T"]; so, ao = np.r_[0, np.cumsum(T + 1)], np.r_[0, np.cumsum(T)]
    events = {k: [] for k in ("water_yield", "cow_food_yield", "plant_food_yield", "zombie_damage_awake",
                              "zombie_damage_asleep", "thirst_rate", "hunger_rate", "fatigue_rate", "zombie_cooldown",
                              "zombie_health_kill", "cow_health_kill", "table_cost", "night_onset")}
    for e in range(len(T)):
        s, a = slice(so[e], so[e + 1]), slice(ao[e], ao[e + 1])
        body = L["s_body"][s].astype(int); sleep = L["s_sleep"][s].astype(bool); pos = L["s_pos"][s].astype(int)
        act = L["t_action"][a].astype(int); inv = L["s_inv"][s].astype(int); light = L["s_light"][s].astype(float)
        zm, zp = L["s_zombies_m"][s].astype(bool), L["s_zombies_p"][s].astype(int)
        cm, cp = L["s_cows_m"][s].astype(bool), L["s_cows_p"][s].astype(int)
        ev = {k: [] for k in events}
        hits = []                                                   # (t, zombie slot) of awake/asleep zombie damage
        for t in range(1, T[e] + 1):
            d = body[t] - body[t - 1]
            adjz = [k for k in range(zm.shape[1]) if zm[t - 1, k] and np.abs(zp[t - 1, k] - pos[t - 1]).sum() == 1]
            adjc = [k for k in range(cm.shape[1]) if cm[t - 1, k] and np.abs(cp[t - 1, k] - pos[t - 1]).sum() == 1]
            if d[DRINK] > 0 and body[t - 1, DRINK] <= 7:
                ev["water_yield"].append(t)                         # a +1 with room to show +2
            if d[FOOD] == 6:
                ev["cow_food_yield"].append(t)                      # unambiguous cow (uncapped)
            if d[FOOD] == 4 and body[t - 1, FOOD] <= 5:
                ev["plant_food_yield"].append(t)                    # unambiguous plant
            if d[HEALTH] < 0 and adjz:
                (ev["zombie_damage_asleep"] if sleep[t - 1] else ev["zombie_damage_awake"]).append(t)
                hits.append((t, adjz[0]))
            if d[DRINK] < 0:
                ev["thirst_rate"].append(t)
            if d[FOOD] < 0:
                ev["hunger_rate"].append(t)
            if d[ENERGY] < 0 and not sleep[t - 1]:
                ev["fatigue_rate"].append(t)
            if act[t - 1] == DO and any(not zm[t, k] for k in adjz):
                ev["zombie_health_kill"].append(t)
            if act[t - 1] == DO and any(not cm[t, k] for k in adjc):
                ev["cow_health_kill"].append(t)
            if inv[t - 1, 0] - inv[t, 0] == 2 and act[t - 1] == 8:   # wood index 0, PLACE_TABLE: the cost is visible
                ev["table_cost"].append(t)
            if light[t - 1] >= 0.5 > light[t]:
                ev["night_onset"].append(t)
        ev["zombie_cooldown"] = [t2 for (t1, k1), (t2, k2) in zip(hits, hits[1:]) if k1 == k2 and t2 - t1 <= 12]
        for k in events:
            events[k].append(ev[k])
    n, res = len(T), {"episodes": int(len(T)), "episode_length_median": float(np.median(T)),
                      "episode_length_quartiles": np.quantile(T, [0.25, 0.75]).tolist(), "rules": {}}
    for k, per in events.items():
        has = [p for p in per if p]; has2 = [p for p in per if len(p) >= 2]
        res["rules"][k] = {"share_of_episodes_seeing_it": len(has) / n, "events_per_episode_mean": float(np.mean([len(p) for p in per])),
                           "first_step_median": float(np.median([p[0] for p in has])) if has else None,
                           "second_step_median": float(np.median([p[1] for p in has2])) if has2 else None,
                           "share_seeing_two": len(has2) / n}
    (HERE / "variation_eda.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
