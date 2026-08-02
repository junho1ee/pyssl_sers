"""Recompute Figure 3B ROC/AUC from the archived class2 checkpoints."""
import sys, json
import numpy as np, torch
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
sys.path.insert(0, ".")
from nets.resnet import ResNet

ROOT = Path("results/bacteria-id/finetuning/class2/phys/supervised/"
            "version_ho_adam_es10_aug/ho_pre_adam_es10_last_ft_ho_split_adam_class2")
x = np.load("data/bacteria-id/preprocessed/X_test_binary.npy")
y = np.load("data/bacteria-id/preprocessed/y_test_binary.npy").astype(int)
xt = torch.tensor(x[:, None, :], dtype=torch.float32)

T, S, P = [], [], []
for fold in range(5):
    ck = ROOT / f"cv{fold}" / "best.ckpt"
    m = ResNet(hidden_sizes=[100]*6, num_blocks=[2]*6, input_dim=x.shape[1],
               in_channels=64, n_classes=2)
    f = m.fc.in_features
    m.fc = torch.nn.Sequential(torch.nn.Linear(f, f), torch.nn.ReLU(), torch.nn.Linear(f, 2))
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    m.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()})
    m.eval()
    with torch.no_grad():
        lg = torch.cat([m(xt[i:i+256]) for i in range(0, len(xt), 256)])
        pr = torch.softmax(lg, 1).numpy()
    T.append(y); S.append(pr[:, 1]); P.append(pr.argmax(1))
    print("  cv%d acc=%.2f" % (fold, 100*(pr.argmax(1) == y).mean()))

T = np.concatenate(T); S = np.concatenate(S); P = np.concatenate(P)
fpr, tpr, thr = roc_curve(T, S)
a = auc(fpr, tpr)
cm = confusion_matrix(T, P)
row = cm * 100.0 / cm.sum(1, keepdims=True)
print("\npooled acc = %.4f" % (100*(P == T).mean()))
print("row-normalised confusion:\n", np.round(row, 2))
print("ROC AUC (positive = MSSA) = %.6f" % a)
json.dump({"auc": float(a), "n_test_per_fold": int(len(y)), "folds": 5,
           "confusion_row_percent": row.tolist(),
           "pooled_accuracy": float(100*(P == T).mean()),
           "fpr": fpr.tolist(), "tpr": tpr.tolist(),
           "y_true": T.tolist(), "y_score": S.tolist()},
          open("results/fig3_roc.json", "w"))
print("wrote results/fig3_roc.json")
