import os

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
    raman_data = spp.clip_data_by_shift(raman_data, (400, 1790))

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

    final_data = np.concatenate((wave_number_in[None, :], y_cubics), axis=0)

    return final_data


datapath = "./data/covid/org/data.mat"

matfile = scp.io.loadmat(datapath)

data_covid = matfile["raw_COVID"].T  # (n_samples, n_wavenumbers)
data_healthy = matfile["raw_Helthy"].T
data_suspected = matfile["raw_Suspected"].T
print(f"data_covid.shape: {data_covid.shape}")
print(f"data_healthy.shape: {data_healthy.shape}")
print(f"data_suspected.shape: {data_suspected.shape}")

wave_number_cov = matfile["wave_number"][0]
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

print(f"preprocessed_covid.shape: {preprocessed_covid.shape}")
print(f"preprocessed_healthy.shape: {preprocessed_healthy.shape}")
print(f"preprocessed_suspected.shape: {preprocessed_suspected.shape}")

preprocessed_dir = "./data/covid/preprocessed/"
os.makedirs(preprocessed_dir, exist_ok=True)

np.save(preprocessed_dir + "covid.npy", preprocessed_covid)
np.save(preprocessed_dir + "healthy.npy", preprocessed_healthy)
np.save(preprocessed_dir + "suspected.npy", preprocessed_suspected)


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


# %%
covid_data = np.load(preprocessed_dir + "covid.npy")
healthy_data = np.load(preprocessed_dir + "healthy.npy")
suspected_data = np.load(preprocessed_dir + "suspected.npy")


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

        # Xy_train = np.concatenate([X_train, y_train[:, None]], axis=1)
        # Xy_val = np.concatenate([X_val, y_val[:, None]], axis=1)
        # Xy_test = np.concatenate([X_test, y_test[:, None]], axis=1)

        cv_dir = task_dir + f"/CV{seed}/"
        os.makedirs(cv_dir, exist_ok=True)
        np.save(cv_dir + "wavenumbers.npy", group0[0])
        np.save(cv_dir + "X_train.npy", X_train)
        np.save(cv_dir + "X_val.npy", X_val)
        np.save(cv_dir + "X_test.npy", X_test)
        np.save(cv_dir + "y_train.npy", y_train)
        np.save(cv_dir + "y_val.npy", y_val)
        np.save(cv_dir + "y_test.npy", y_test)
        # np.save(cv_dir + "Xy_train.npy", Xy_train)
        # np.save(cv_dir + "Xy_val.npy", Xy_val)
        # np.save(cv_dir + "Xy_test.npy", Xy_test)


# %%
# task0 covid vs suspected
save_task_cv(group0=covid_data, group1=suspected_data, task=0, nfold=50)
# task1 covid vs healthy
save_task_cv(group0=covid_data, group1=healthy_data, task=1, nfold=50)
# task2 suspected vs healthy
save_task_cv(group0=suspected_data, group1=healthy_data, task=2, nfold=50)
