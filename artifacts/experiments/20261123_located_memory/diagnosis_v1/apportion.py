import importlib, torch, numpy as np
from collections import Counter
LD = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v1.localizer_diagnosis"); LM = LD.LM
loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / "localizer_v1.pt", map_location="cpu")); loc.eval()
data = torch.load(LM.DATA / "tokens_panel.pt")[:4]
offs = torch.stack([LM.NR, LM.NC], -1)
cen = torch.zeros(63, dtype=torch.bool); cen[31] = True
plus = cen.clone(); plus[[22, 30, 32, 40]] = True
res = Counter(); n = 0; nw = 0; trust = []
with torch.no_grad():
  for r in data:
    mp = LM.Map(1, torch.device("cpu")); pos = r["pos"]; tc = LM.truth_cells(pos)
    for t in range(r["T"] + 1):
        one = lambda k: r[k][t][None]
        first = torch.tensor([t == 0]); valid = torch.tensor([True]); tt = torch.tensor([t])
        before = mp.pose.clone(); ent = mp.view()[0]
        if t > 0 and int(r["prev_action"][t]) in LM.GT.MOVE and (pos[t] - pos[t - 1]).tolist() != [0, 0]:
            d = (pos[t] - pos[t - 1]).tolist()
            hyp = (before[:, None] + offs[None]).clamp(0, LM.G - 1)
            s, pr = LD.cell_scores(loc, ent, one("p")[:, :LM.MAP].float(), one("v").float(), tt.float(), hyp)
            it = [i for i in range(9) if offs[i].tolist() == d][0]
            logK = loc.kernel(mp.last_p, mp.last_v, one("prev_action"))[0]
            n += 1
            tot = s.sum(1)
            if int(tot.argmax()) != it:
                nw += 1
                res["wrong_choice_stay"] += int(tot.argmax()) == 4
                res["true_wins_without_center"] += int(s[:, ~cen].sum(1).argmax()) == it
                res["true_wins_without_center_plus4"] += int(s[:, ~plus].sum(1).argmax()) == it
                res["true_wins_with_prior"] += int((tot + logK).argmax()) == it
                res["true_wins_with_prior_without_center"] += int((s[:, ~cen].sum(1) + logK).argmax()) == it
                ch = int(tot.argmax()); both = pr[it] & pr[ch] & ~cen
                res["true_wins_common_cells_without_center"] += int(s[it][both].sum() > s[ch][both].sum())
                res["gap_choice_minus_true"] += float(tot[ch] - tot[it])
                res["center_contrib"] += float(s[ch][31] - s[it][31])
                res["noncommon_contrib"] += float((s[ch][~(pr[it] & pr[ch])].sum() - s[it][~(pr[it] & pr[ch])].sum()))
            else:
                res["right_without_center_stays_right"] += int(s[:, ~cen].sum(1).argmax()) == it
        LM.advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"), one("prev_action"), tt, tt, first, valid, truth=tc[t][None])
print("moved steps", n, "match wrong", nw)
for k, v in res.items(): print(k, round(v / (nw if k != "right_without_center_stays_right" else n - nw), 3))
# the learned trust by age (lighting = mean)
age = torch.tensor([1., 2., 4., 16., 64., 256.]); present = torch.ones(6, dtype=torch.bool)
h = loc.trust(age, present, torch.zeros(6, 8))
print("age", age.tolist()); print("scale b", h[:, 0].exp().tolist()); print("trust", torch.sigmoid(h[:, 1]).tolist()); print("b_bg", float(loc.log_b_bg.exp()))
