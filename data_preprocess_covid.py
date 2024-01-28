# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scp
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from tqdm import tqdm

import signalpreprocess as spp

# %%
bacteria_wavenumber_path = "./data/bacteria-id/org/wavenumbers.npy"
bacteria_wavenumber = np.load(bacteria_wavenumber_path)

# %%
wave_number_in = np.linspace(400, 1790, 696)

# %%
print(wave_number_in.shape)

# %%
datapath = "./data/covid/org/data.mat"

# %%
matfile = scp.io.loadmat(datapath)

# %%
data_covid = matfile["raw_COVID"]
data_healthy = matfile["raw_Helthy"]
data_suspected = matfile["raw_Suspected"]

# %%
wave_number_cov = matfile["wave_number"][0]

# %%
print(data_covid.shape, data_healthy.shape, data_suspected.shape)
print(wave_number_cov.shape)

# %%
wavenumber_resample = wave_number_cov[:688]
intensity = data_covid[:, 0][:688]
value = intensity.copy()
value = preprocessing.minmax_scale(scp.signal.savgol_filter(value, 11, 3), axis=0)

value_poly = scp.signal.resample_poly(value, 1000, 688)
fcubic = scp.interpolate.interp1d(wavenumber_resample, value, kind="cubic")

print(wavenumber_resample.min(), wavenumber_resample.max())

wavenumber_poly = np.linspace(
    wavenumber_resample.min(), wavenumber_resample.max(), 1000, endpoint=True
)
y_cubic = fcubic(wave_number_in)

print(wavenumber_poly.shape)

# %%
plt.plot(wavenumber_resample, intensity, label="original", alpha=0.5, color="black")
plt.plot(wavenumber_resample, value, label="scaled", alpha=0.5, color="red")
plt.plot(wavenumber_poly, value_poly, label="resampled", alpha=0.5, color="blue")
plt.plot(wave_number_in, y_cubic, label="cubic", alpha=0.2, color="green")
plt.legend()

# %%
raman_shift = wave_number_cov  # (900)
peaks = data_covid  # (900, 159)
wave_number_in = np.linspace(400, 1790, 696)

raman_data = np.concatenate((raman_shift[:, None], peaks), axis=1)

# raman_data = preprocess.baseline_als(raman_data, lam=1e5, p=0.05)
raman_data = spp.clip_data_by_shift(raman_data.T, (400, 1790)).T

shift = raman_data[:, 0]
value = raman_data[:, 1:]
value = preprocessing.minmax_scale(scp.signal.savgol_filter(value, 11, 3), axis=0)

y_cubics = np.zeros((wave_number_in.shape[0], value.shape[1]))
for i in tqdm(range(value.shape[1])):
    fcubic = scp.interpolate.interp1d(shift.ravel(), value[:, i].ravel(), kind="cubic")
    y_cubic = fcubic(wave_number_in)
    y_cubics[:, i] = y_cubic

final_data = np.concatenate([wave_number_in[:, None], y_cubics], axis=1).T


# %%
final_data.shape


# %%
def preprocess_data(raman_shift=None, peaks=None, wave_number_in=None):
    # raman_shift = wave_number_cov # (900)
    # peaks = data_covid # (900, 159)
    # wave_number_in = np.linspace(400, 1790, 696)

    raman_data = np.concatenate((raman_shift[:, None], peaks), axis=1)

    # raman_data = preprocess.baseline_als(raman_data, lam=1e5, p=0.05)
    raman_data = preprocess.clip_data_by_shift(raman_data.T, (400, 1790)).T

    shift = raman_data[:, 0]
    value = raman_data[:, 1:]
    value = preprocessing.minmax_scale(scp.signal.savgol_filter(value, 11, 3), axis=0)

    y_cubics = np.zeros((wave_number_in.shape[0], value.shape[1]))
    for i in tqdm(range(value.shape[1])):
        fcubic = scp.interpolate.interp1d(
            shift.ravel(), value[:, i].ravel(), kind="cubic"
        )
        y_cubic = fcubic(wave_number_in)
        y_cubics[:, i] = y_cubic

    final_data = np.concatenate([wave_number_in[:, None], y_cubics], axis=1).T

    return final_data


# %%
wave_number_in = np.linspace(400, 1790, 696)

preprocessed_covid = preprocess_data(
    raman_shift=wave_number_cov, peaks=data_covid, wave_number_in=wave_number_in
)
preprocessed_healthy = preprocess_data(
    raman_shift=wave_number_cov, peaks=data_healthy, wave_number_in=wave_number_in
)
preprocessed_suspected = preprocess_data(
    raman_shift=wave_number_cov, peaks=data_suspected, wave_number_in=wave_number_in
)

# %%
print(preprocessed_covid.shape)

# %%
i = 6

plt.plot(
    preprocessed_covid[0], preprocessed_covid[i], label="covid", alpha=0.5, color="red"
)
plt.plot(
    preprocessed_healthy[0],
    preprocessed_healthy[i],
    label="healthy",
    alpha=0.5,
    color="green",
)
plt.plot(
    preprocessed_suspected[0],
    preprocessed_suspected[i],
    label="suspected",
    alpha=0.5,
    color="blue",
)
plt.legend()

# %%
preprocessed_dir = "./data/covid/preprocessed/"
os.makedirs(preprocessed_dir, exist_ok=True)

np.save(preprocessed_dir + "covid.npy", preprocessed_covid)
np.save(preprocessed_dir + "healthy.npy", preprocessed_healthy)
np.save(preprocessed_dir + "suspected.npy", preprocessed_suspected)


# %%
def split_data(data, labels, p_train=0.7, p_val=0.1, seed=0):
    n = len(labels)
    np.random.seed(seed)
    ind = np.random.permutation(n)

    n_train = int(p_train * n)
    n_val = int(p_val * n_train)

    val_idxs, train_idxs = ind[:n_val], ind[n_val:n_train]
    test_idxs = ind[n_train:]

    data_train = data[train_idxs]
    data_val = data[val_idxs]
    data_test = data[test_idxs]

    labels_train = labels[train_idxs]
    labels_val = labels[val_idxs]
    labels_test = labels[test_idxs]

    return data_train, data_val, data_test, labels_train, labels_val, labels_test

    # data_train = data[ind[0:int(n*p)]]
    # data_test = data[ind[int(n*p):]]
    # labels_train = labels[ind[0:int(n*p)]]
    # labels_test = labels[ind[int(n*p):]]

    # return data_train, data_test, labels_train, labels_test


# %%
covid_data = np.load(preprocessed_dir + "covid.npy")
healthy_data = np.load(preprocessed_dir + "healthy.npy")
suspected_data = np.load(preprocessed_dir + "suspected.npy")


# %%
# save cv data
# group 0
# covid vs healthy


def save_task_cv(group0=None, group1=None, task=0, nfold=50):
    task_dir = f"./data/covid/task{task}/"
    os.makedirs(task_dir, exist_ok=True)

    X0 = group0[1:]
    X1 = group1[1:]

    n0 = X0.shape[0]
    n1 = X1.shape[0]

    for seed in tqdm(range(nfold)):
        Xy0 = split_data(X0, np.ones(n0), seed=seed)
        Xy1 = split_data(X1, np.zeros(n1), seed=seed)

        X_train = np.concatenate([Xy0[0], Xy1[0]], axis=0)
        X_val = np.concatenate([Xy0[1], Xy1[1]], axis=0)
        X_test = np.concatenate([Xy0[2], Xy1[2]], axis=0)

        y_train = np.concatenate([Xy0[3], Xy1[3]], axis=0)
        y_val = np.concatenate([Xy0[4], Xy1[4]], axis=0)
        y_test = np.concatenate([Xy0[5], Xy1[5]], axis=0)

        Xy_train = np.concatenate([X_train, y_train[:, None]], axis=1)
        Xy_val = np.concatenate([X_val, y_val[:, None]], axis=1)
        Xy_test = np.concatenate([X_test, y_test[:, None]], axis=1)

        cv_dir = task_dir + f"/CV{seed}/"
        os.makedirs(cv_dir, exist_ok=True)
        np.save(cv_dir + "wavenumbers.npy", group0[0])
        np.save(cv_dir + "Xy_train.npy", Xy_train)
        np.save(cv_dir + "Xy_val.npy", Xy_val)
        np.save(cv_dir + "Xy_test.npy", Xy_test)


# %%
# task0 covid vs suspected
save_task_cv(group0=covid_data, group1=suspected_data, task=0, nfold=50)
# task1 covid vs healthy
save_task_cv(group0=covid_data, group1=healthy_data, task=1, nfold=50)
# task2 suspected vs healthy
save_task_cv(group0=suspected_data, group1=healthy_data, task=2, nfold=50)

# %%
