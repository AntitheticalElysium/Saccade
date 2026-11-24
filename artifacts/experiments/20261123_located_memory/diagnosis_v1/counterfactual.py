import importlib, math, torch, numpy as np, torch.nn.functional as F
from collections import defaultdict
LD = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v1.localizer_diagnosis"); LM = LD.LM
loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / "localizer_v1.pt", map_location="cpu")); loc.eval()
data = torch.load(LM.DATA / "tokens_panel.pt")[:6]
offs = torch.stack([LM.NR, LM.NC], -1)
KR, KC = LM.KR, LM.KC
# inner neighbour of each map cell: one step toward the view centre (row 3, col 4); centre -> itself
nb = []
for k in range(63):
    r, c = divmod(k, 9)
    dr, dc = np.sign(3 - r), np.sign(4 - c)
    if abs(3 - r) >= abs(4 - c): dc = 0
    else: dr = 0
    nb.append((r + dr) * 9 + (c + dc))
nb = torch.tensor(nb)
acc = defaultdict(list)
with torch.no_grad():
  for r in data:
    mp = LM.Map(1, torch.device("cpu")); pos = r["pos"]; tc = LM.truth_cells(pos)
    for t in range(r["T"] + 1):
        one = lambda k: r[k][t][None]
        first = torch.tensor([t == 0]); valid = torch.tensor([True]); tt = torch.tensor([t])
        before = mp.pose.clone(); ent = mp.view()[0]
        if t > 0 and int(r["prev_action"][t]) in LM.GT.MOVE:
            d = (pos[t] - pos[t - 1]).tolist()
            hyp = (before[:, None] + offs[None]).clamp(0, LM.G - 1)
            c = one("p")[:, :LM.MAP].float(); v = one("v").float()
            s, pr = LD.cell_scores(loc, ent, c, v, tt.float(), hyp)
            it = [i for i in range(9) if offs[i].tolist() == d][0]
            # coverage-neutral: absent cells scored by a Laplace mixture around the inner neighbour's token (age-1 scale/trust)
            h = loc.trust(torch.ones(63), torch.ones(63, dtype=torch.bool), v.expand(63, 8))
            snb = loc.score(c[0], c[0, nb], torch.ones(63), nb != torch.arange(63), v.expand(63, 8), h=h)   # [63]
            s_cov = torch.where(pr, s - snb[None], torch.zeros_like(s))
            # sprite: the new view's centre and the cell holding the last frame's centre entry are not evidence
            addr = hyp[0][:, None, :] + torch.stack([KR, KC], -1)[None]          # [9, 63, 2]
            trail = (addr == before[0]).all(-1)
            spr = torch.zeros_like(pr); spr[:, 31] = True; spr |= trail
            s_spr = torch.where(spr, torch.zeros_like(s), s)
            s_both = torch.where(spr, torch.zeros_like(s_cov), s_cov)
            dark = float(r["light"][t]) < 0.5
            key = ("moved" if d != [0, 0] else "blocked") + ("_dark" if dark else "_day")
            for name, S in (("current", s), ("coverage_neutral", s_cov), ("sprite_removed", s_spr), ("both", s_both)):
                acc[(key, name)].append(int(S.sum(1).argmax()) == it)
        LM.advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"), one("prev_action"), tt, tt, first, valid, truth=tc[t][None])
for k in sorted(acc): print(k, len(acc[k]), round(float(np.mean(acc[k])), 3))
