import multiprocessing as mp
import os
from functools import partial

import numpy as np
import scipy as scp
from sklearn import preprocessing
from tqdm import tqdm

import preprocess as pp


WAVE_NUMBER_MIN = 400
WAVE_NUMBER_MAX = 1800
N_WAVE_POINTS = 696
CLIP_RANGE = (400, 1800)
BASELINE_LAMBDA = 1e5
BASELINE_P = 5e-3
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 3

ORG_DATA_PATH = "./data/covid/org/data.mat"
PREPROCESSED_DIR = "./data/covid/preprocessed/"


def mp_run(func, data, n_jobs=4, **kwargs):
    pool = mp.Pool(n_jobs)
    data_split = np.array_split(data, n_jobs, axis=1)
    pool_func = partial(func, **kwargs)
    result = np.concatenate(pool.map(pool_func, data_split), axis=1)
    pool.close()
    pool.join()
    return result


def preprocess_data(raman_shift=None, peaks=None, wave_number_in=None):
    """
    raman_shift: (n_wavenumbers)
    peaks: (n_samples, n_wavenumbers)
    """
    if raman_shift is None or peaks is None or wave_number_in is None:
        raise ValueError("raman_shift, peaks, and wave_number_in must be provided")

    raman_data = np.concatenate((raman_shift[None, :], peaks), axis=0)
    raman_data = pp.clip_data_by_shift(raman_data, CLIP_RANGE)
    raman_data = pp.baseline_als(raman_data.T, lam=BASELINE_LAMBDA, p=BASELINE_P).T

    shift = raman_data[0, :]
    value = raman_data[1:, :]

    value = scp.signal.savgol_filter(value, SAVGOL_WINDOW, SAVGOL_POLYORDER, axis=1)
    value = preprocessing.minmax_scale(value, axis=1)

    y_cubics = np.zeros((value.shape[0], wave_number_in.shape[0]))
    for iv in tqdm(range(value.shape[0])):
        fcubic = scp.interpolate.interp1d(
            shift.ravel(),
            value[iv, :].ravel(),
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        y_cubics[iv, :] = fcubic(wave_number_in)

    return np.concatenate((wave_number_in[None, :], y_cubics), axis=0)


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

        cv_dir = task_dir + f"/CV{seed}/"
        os.makedirs(cv_dir, exist_ok=True)
        np.save(cv_dir + "wavenumbers.npy", group0[0])
        np.save(cv_dir + "X_train.npy", X_train)
        np.save(cv_dir + "X_val.npy", X_val)
        np.save(cv_dir + "X_test.npy", X_test)
        np.save(cv_dir + "y_train.npy", y_train)
        np.save(cv_dir + "y_val.npy", y_val)
        np.save(cv_dir + "y_test.npy", y_test)


def main():
    matfile = scp.io.loadmat(ORG_DATA_PATH)

    data_covid = matfile["raw_COVID"].T
    data_healthy = matfile["raw_Helthy"].T
    data_suspected = matfile["raw_Suspected"].T
    print(f"data_covid.shape: {data_covid.shape}")
    print(f"data_healthy.shape: {data_healthy.shape}")
    print(f"data_suspected.shape: {data_suspected.shape}")

    wave_number_cov = matfile["wave_number"][0]
    wave_number_in = np.linspace(WAVE_NUMBER_MIN, WAVE_NUMBER_MAX, N_WAVE_POINTS)

    preprocessed_covid = preprocess_data(
        raman_shift=wave_number_cov, peaks=data_covid, wave_number_in=wave_number_in
    )
    preprocessed_healthy = preprocess_data(
        raman_shift=wave_number_cov,
        peaks=data_healthy,
        wave_number_in=wave_number_in,
    )
    preprocessed_suspected = preprocess_data(
        raman_shift=wave_number_cov,
        peaks=data_suspected,
        wave_number_in=wave_number_in,
    )

    print(f"preprocessed_covid.shape: {preprocessed_covid.shape}")
    print(f"preprocessed_healthy.shape: {preprocessed_healthy.shape}")
    print(f"preprocessed_suspected.shape: {preprocessed_suspected.shape}")

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(PREPROCESSED_DIR, "covid.npy"), preprocessed_covid)
    np.save(os.path.join(PREPROCESSED_DIR, "healthy.npy"), preprocessed_healthy)
    np.save(os.path.join(PREPROCESSED_DIR, "suspected.npy"), preprocessed_suspected)

    covid_data = np.load(os.path.join(PREPROCESSED_DIR, "covid.npy"))
    healthy_data = np.load(os.path.join(PREPROCESSED_DIR, "healthy.npy"))
    suspected_data = np.load(os.path.join(PREPROCESSED_DIR, "suspected.npy"))

    save_task_cv(group0=covid_data, group1=suspected_data, task=0, nfold=50)
    save_task_cv(group0=covid_data, group1=healthy_data, task=1, nfold=50)
    save_task_cv(group0=suspected_data, group1=healthy_data, task=2, nfold=50)


if __name__ == "__main__":
    main()
