"""Effect of the (696, 1) layout bug of covid_transfer_eval.py on the phys augmentation.

Applies each component to 200 random COVID spectra in the correct (1, 696) and
the buggy (696, 1) layout and prints the pooled RMS change, the number of
changed points, the largest change, and the within-spectrum SD of the change.
"""
import sys, random
import numpy as np, torch
sys.path.insert(0, ".")
import utils
z = np.load("data/covid/preprocessed/covid_696.npz")
X = np.concatenate([z[k] for k in z.files if k.endswith("_X")]).astype(np.float32)
rng = np.random.RandomState(0)
idx = rng.choice(len(X), 200, replace=False)
print(f"{'aug':22s} {'layout':8s} {'RMS change':>10s} {'pts changed (median)':>21s} {'max|change|':>11s} {'within-spectrum SD of change':>28s}")
for at in ["phys_only_powerline", "phys_only_emg", "phys_only_shift", "phys_only_wander", "phys"]:
    tf = utils.get_trans_from_augtype(at)
    for layout in ("(1,696)", "(696,1)"):
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        d = []
        for i in idx:
            x = X[i]
            if layout == "(1,696)":
                y = np.asarray(tf(x[None, :])).reshape(-1)
            else:
                y = np.asarray(tf(x[:, None])).squeeze()
            d.append(y - x)
        d = np.array(d)
        nchg = np.median((np.abs(d) > 1e-7).sum(1))
        print(f"{at:22s} {layout:8s} {np.sqrt((d**2).mean()):10.4f} {nchg:21.0f} {np.abs(d).max():11.3f} {np.median(d.std(1)):28.4f}")
