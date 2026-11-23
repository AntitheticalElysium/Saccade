import importlib, torch, numpy as np
LD = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v1.localizer_diagnosis"); LM = LD.LM
loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / "localizer_v1.pt", map_location="cpu")); loc.eval()
data = torch.load(LM.DATA / "tokens_panel.pt")[:3]
offs = torch.stack([LM.NR, LM.NC], -1)
adv, win, cnt = torch.zeros(63), torch.zeros(63), torch.zeros(63)
with torch.no_grad():
  for r in data:
    mp = LM.Map(1, torch.device("cpu")); pos = r["pos"]; tc = LM.truth_cells(pos)
    for t in range(r["T"] + 1):
        one = lambda k: r[k][t][None]
        first = torch.tensor([t == 0]); valid = torch.tensor([True]); tt = torch.tensor([t])
        before = mp.pose.clone(); ent = mp.view()[0]
        if t > 0 and int(r["prev_action"][t]) in LM.GT.MOVE:
            d = (pos[t] - pos[t - 1]).tolist()
            if d != [0, 0]:
                hyp = (before[:, None] + offs[None]).clamp(0, LM.G - 1)
                s, pr = LD.cell_scores(loc, ent, one("p")[:, :LM.MAP].float(), one("v").float(), tt.float(), hyp)
                it = [i for i in range(9) if offs[i].tolist() == d][0]
                both = pr[4] & pr[it]
                adv += torch.where(both, s[4] - s[it], 0.); win += (both & (s[4] > s[it])).float(); cnt += both.float()
        LM.advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"), one("prev_action"), tt, tt, first, valid, truth=tc[t][None])
np.set_printoptions(precision=1, suppress=True, linewidth=200)
print("mean (stay - true) per cell, 7x9:"); print((adv / cnt.clamp(min=1)).reshape(7, 9).numpy())
print("P(stay beats true) per cell:"); print((win / cnt.clamp(min=1)).reshape(7, 9).numpy())
print("total mean advantage of stay", float(adv.sum() / cnt.max()))
