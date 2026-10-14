#!/usr/bin/env python
"""Write the Bacteria-ID subsets with the extrapolated points held.

The last four of the 696 grid points lie above 1792.4 cm-1, where Bacteria-ID
has no measurements, and were filled by cubic extrapolation during
preprocessing. Here they are set to the value of the last measured point, so a
model sees a flat edge instead. Used by data_variant "edgehold" in fine-tuning
(step 3) and pretraining (step 3b).
"""

import hashlib
import os
import shutil

import numpy as np

SRC = "data/bacteria-id/preprocessed"
DST = "data_rev1/bacteria-id/preprocessed_edgehold"
N_MEASURED = 692

os.makedirs(DST, exist_ok=True)
for name in ["X_reference.npy", "X_finetune.npy", "X_test.npy"]:
    x = np.load(os.path.join(SRC, name))
    x[:, N_MEASURED:] = x[:, N_MEASURED - 1:N_MEASURED]
    np.save(os.path.join(DST, name), x)
for name in ["y_reference.npy", "y_finetune.npy", "y_test.npy"]:
    shutil.copyfile(os.path.join(SRC, name), os.path.join(DST, name))
with open(os.path.join(DST, "MD5"), "w") as f:
    for name in sorted(os.listdir(DST)):
        if name.endswith(".npy"):
            h = hashlib.md5(open(os.path.join(DST, name), "rb").read()).hexdigest()
            f.write(f"{h}  {name}\n")
print(open(os.path.join(DST, "MD5")).read())
