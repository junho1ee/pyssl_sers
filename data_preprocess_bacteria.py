import os
import shutil

import numpy as np
import scipy as scp
from sklearn import preprocessing
from tqdm import tqdm

import signalpreprocess as spp


def preprocess_data(raman_shift=None, peaks=None, wave_number_in=None):
    """
    raman_shift has shape (n_wavenumbers, )
    """
    raman_data = np.concatenate((raman_shift[None, :], peaks), axis=0)
    raman_data = spp.clip_data_by_shift(raman_data, (398, 1792))

    shift = raman_data[0, :]
    value = raman_data[1:, :]

    value = scp.signal.savgol_filter(value, 11, 3, axis=1)
    value = preprocessing.minmax_scale(value, axis=1)

    y_cubics = np.zeros((value.shape[0], wave_number_in.shape[0]))
    for iv in tqdm(range(value.shape[0])):
        fcubic = scp.interpolate.interp1d(
            shift.ravel(), value[iv, :].ravel(), kind="cubic"
        )
        y_cubic = fcubic(wave_number_in)
        y_cubics[iv, :] = y_cubic

    return y_cubics


wave_number_in = np.linspace(400, 1790, 696)

wavenumberpath = "./data/bacteria-id/org/wavenumbers.npy"
wavenumber = np.load(wavenumberpath)

X_reference = np.load("./data/bacteria-id/org/X_reference.npy")
X_finetune = np.load("./data/bacteria-id/org/X_finetune.npy")
X_test = np.load("./data/bacteria-id/org/X_test.npy")

"""
bacteria-id original data has reversed wavenumber, datas
so we need to reverse it back. ex) wavenumber[::-1]
"""
X_reference_preprocessed = preprocess_data(
    raman_shift=wavenumber[::-1],
    peaks=X_reference[:, ::-1],
    wave_number_in=wave_number_in,
)
X_finetune_preprocessed = preprocess_data(
    raman_shift=wavenumber[::-1],
    peaks=X_finetune[:, ::-1],
    wave_number_in=wave_number_in,
)
X_test_preprocessed = preprocess_data(
    raman_shift=wavenumber[::-1], peaks=X_test[:, ::-1], wave_number_in=wave_number_in
)

print(f"X_reference_preprocessed.shape: {X_reference_preprocessed.shape}")
print(f"X_finetune_preprocessed.shape: {X_finetune_preprocessed.shape}")
print(f"X_test_preprocessed.shape: {X_test_preprocessed.shape}")

preproceed_dir = "./data/bacteria-id/preprocessed/"
os.makedirs(preproceed_dir, exist_ok=True)

np.save(preproceed_dir + "X_reference.npy", X_reference_preprocessed)
np.save(preproceed_dir + "X_finetune.npy", X_finetune_preprocessed)
np.save(preproceed_dir + "X_test.npy", X_test_preprocessed)

# For B groupings
y_reference = np.load("./data/bacteria-id/org/y_reference.npy")
y_finetune = np.load("./data/bacteria-id/org/y_finetune.npy")
y_test = np.load("./data/bacteria-id/org/y_test.npy")

B_GROUPINGS = {
    16: 0,
    17: 0,
    14: 1,
    15: 1,
    18: 1,
}

for i in range(30):
    if i in B_GROUPINGS.keys():
        pass
    else:
        B_GROUPINGS[i] = 2


# %%
y_reference_grouped = np.array([B_GROUPINGS[i] for i in y_reference])
y_finetune_grouped = np.array([B_GROUPINGS[i] for i in y_finetune])
y_test_grouped = np.array([B_GROUPINGS[i] for i in y_test])

# %%
MRSA_inds = np.arange(y_reference_grouped.shape[0])[y_reference_grouped == 0]
print(MRSA_inds.shape)
MSSA_inds = np.arange(y_reference_grouped.shape[0])[y_reference_grouped == 1]
print(MSSA_inds.shape)

# %%
X_reference_preprocessed_MRSA = X_reference_preprocessed[MRSA_inds, :]
X_reference_preprocessed_MSSA = X_reference_preprocessed[MSSA_inds, :]
X_reference_preprocessed_B = np.concatenate(
    [X_reference_preprocessed_MRSA, X_reference_preprocessed_MSSA], axis=0
)

y_reference_grouped_MRSA = y_reference_grouped[MRSA_inds]
y_reference_grouped_MSSA = y_reference_grouped[MSSA_inds]
y_reference_grouped_B = np.concatenate(
    [y_reference_grouped_MRSA, y_reference_grouped_MSSA], axis=0
)


X_finetune_preprocessed_MRSA = X_finetune_preprocessed[y_finetune_grouped == 0, :]
X_finetune_preprocessed_MSSA = X_finetune_preprocessed[y_finetune_grouped == 1, :]
X_finetune_preprocessed_B = np.concatenate(
    [X_finetune_preprocessed_MRSA, X_finetune_preprocessed_MSSA], axis=0
)

y_finetune_grouped_MRSA = y_finetune_grouped[y_finetune_grouped == 0]
y_finetune_grouped_MSSA = y_finetune_grouped[y_finetune_grouped == 1]
y_finetune_grouped_B = np.concatenate(
    [y_finetune_grouped_MRSA, y_finetune_grouped_MSSA], axis=0
)


X_test_preprocessed_MRSA = X_test_preprocessed[y_test_grouped == 0, :]
X_test_preprocessed_MSSA = X_test_preprocessed[y_test_grouped == 1, :]
X_test_preprocessed_B = np.concatenate(
    [X_test_preprocessed_MRSA, X_test_preprocessed_MSSA], axis=0
)

y_test_grouped_MRSA = y_test_grouped[y_test_grouped == 0]
y_test_grouped_MSSA = y_test_grouped[y_test_grouped == 1]
y_test_grouped_B = np.concatenate([y_test_grouped_MRSA, y_test_grouped_MSSA], axis=0)

print(X_reference_preprocessed_B.shape, y_reference_grouped_B.shape)
print(X_finetune_preprocessed_B.shape, y_finetune_grouped_B.shape)
print(X_test_preprocessed_B.shape, y_test_grouped_B.shape)

preproceed_dir = "./data/bacteria-id/preprocessed/"

np.save(preproceed_dir + "X_reference_binary.npy", X_reference_preprocessed_B)
np.save(preproceed_dir + "X_finetune_binary.npy", X_finetune_preprocessed_B)
np.save(preproceed_dir + "X_test_binary.npy", X_test_preprocessed_B)
np.save(preproceed_dir + "y_reference_binary.npy", y_reference_grouped_B)
np.save(preproceed_dir + "y_finetune_binary.npy", y_finetune_grouped_B)
np.save(preproceed_dir + "y_test_binary.npy", y_test_grouped_B)

# copy remaining files
shutil.copy(
    "./data/bacteria-id/org/wavenumbers.npy", preproceed_dir + "wavenumbers.npy"
)
shutil.copy(
    "./data/bacteria-id/org/y_reference.npy", preproceed_dir + "y_reference.npy"
)
shutil.copy("./data/bacteria-id/org/y_finetune.npy", preproceed_dir + "y_finetune.npy")
shutil.copy("./data/bacteria-id/org/y_test.npy", preproceed_dir + "y_test.npy")
