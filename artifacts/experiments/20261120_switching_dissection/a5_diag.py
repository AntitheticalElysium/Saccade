import importlib, numpy as np, torch
A5 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.a5")
S2, S, TR, D, DEV = A5.S2, A5.S, A5.TR, A5.D, A5.DEV
model, cfg = S2.load()
L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
rows, plans = S2.bank(None)
T = A5.targets(len(rows))
ws = sorted(T)
rng = np.random.default_rng(0); pick = sorted(rng.choice(ws, 120, replace=False))
eps = np.unique(rows[pick, 0]); sub = {e: i for i, e in enumerate(eps)}
enc = S.encode(model, S2.sublabels(L, eps), cfg)
dl = S.episodes(L)
out = []
for w in pick:
    e, t = int(rows[w, 0]), int(rows[w, 3]); en = enc[sub[e]]
    with torch.no_grad():
        for fx in (tuple(int(x) for x in en["fix"][t]), T[w]["on"]):
            c = TR.tensors(D.gather([dl[e]], [[(0, 1, t, *fx)]]), DEV)
            with TR.autocast(DEV, cfg):
                pc = model.perception(c["frames"], c["fixation"])
            d = (pc.p[0, 0].float().cpu() - en["p"][t].float()).abs()
            out.append((w, fx == tuple(int(x) for x in en["fix"][t]), float(d.max()), float(d.mean()), float(en["p"][t].float().abs().max()),
                        float((pc.g[0, 0].float().cpu() - en["g"][t].float()).abs().max()) if fx == tuple(int(x) for x in en["fix"][t]) else -1))
a = np.array([o[2:] for o in out])
print("max|dp| quantiles (seeded-fix re-encode, then on-fix):", np.quantile(a[:, 0], [0.5, 0.9, 0.99, 1.0]).round(4))
print("mean|dp| max:", a[:, 1].max().round(5), " |p| max:", a[:, 2].max().round(3))
seeded = [o for o in out if o[1]]
print("seeded re-encode: glimpse max|dg| quantiles", np.quantile([o[5] for o in seeded], [0.5, 0.99, 1.0]).round(4))
print("worst cases:", sorted(out, key=lambda o: -o[2])[:4])
