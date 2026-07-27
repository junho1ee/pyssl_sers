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

ORG_DIR_CANDIDATES = ("./data/bacteria-id/org/", "./data/bacteria-id/data/")
FULL_PREPROCESSED_DIR = "./data/bacteria-id/preprocessed/"
MINIMAL_PREPROCESSED_DIR = "./data/bacteria-id/preprocessed_minimal/"
REQUIRED_ORG_FILES = (
    "wavenumbers.npy",
    "X_reference.npy",
    "X_finetune.npy",
    "X_test.npy",
    "y_reference.npy",
    "y_finetune.npy",
    "y_test.npy",
)

B_GROUPINGS = {i: 2 for i in range(30)}
B_GROUPINGS.update(
    {
        16: 0,
        17: 0,
        14: 1,
        15: 1,
        18: 1,
    }
)


def mp_run(func, data, n_jobs=4, **kwargs):
    pool = mp.Pool(n_jobs)
    data_split = np.array_split(data, n_jobs, axis=1)
    pool_func = partial(func, **kwargs)
    result = np.concatenate(pool.map(pool_func, data_split), axis=1)
    pool.close()
    pool.join()
    return result


def preprocess_data(
    raman_shift=None,
    peaks=None,
    wave_number_in=None,
    parallel=False,
    n_jobs=4,
    apply_baseline=True,
    apply_savgol=True,
):
    """
    raman_shift: (n_wavenumbers)
    peaks: (n_samples, n_wavenumbers)
    """
    if raman_shift is None or peaks is None or wave_number_in is None:
        raise ValueError("raman_shift, peaks, and wave_number_in must be provided")

    raman_data = np.concatenate((raman_shift[None, :], peaks), axis=0)
    raman_data = pp.clip_data_by_shift(raman_data, CLIP_RANGE)

    if apply_baseline:
        if parallel:
            raman_data = mp_run(
                pp.baseline_als,
                raman_data.T,
                n_jobs=n_jobs,
                lam=BASELINE_LAMBDA,
                p=BASELINE_P,
            ).T
        else:
            raman_data = pp.baseline_als(
                raman_data.T, lam=BASELINE_LAMBDA, p=BASELINE_P
            ).T

    shift = raman_data[0, :]
    value = raman_data[1:, :]

    if apply_savgol:
        value = scp.signal.savgol_filter(
            value, SAVGOL_WINDOW, SAVGOL_POLYORDER, axis=1
        )
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

    return y_cubics


def save_processed_dataset(
    output_dir,
    X_reference,
    X_finetune,
    X_test,
    y_reference,
    y_finetune,
    y_test,
):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_reference.npy"), X_reference)
    np.save(os.path.join(output_dir, "X_finetune.npy"), X_finetune)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_reference.npy"), y_reference)
    np.save(os.path.join(output_dir, "y_finetune.npy"), y_finetune)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    y_reference_grouped = np.array([B_GROUPINGS[i] for i in y_reference])
    y_finetune_grouped = np.array([B_GROUPINGS[i] for i in y_finetune])
    y_test_grouped = np.array([B_GROUPINGS[i] for i in y_test])

    def get_binary_subset(X, y_grouped):
        idx0 = np.flatnonzero(y_grouped == 0)
        idx1 = np.flatnonzero(y_grouped == 1)
        idx = np.concatenate([idx0, idx1], axis=0)
        return X[idx], y_grouped[idx]

    X_reference_binary, y_reference_binary = get_binary_subset(
        X_reference, y_reference_grouped
    )
    X_finetune_binary, y_finetune_binary = get_binary_subset(
        X_finetune, y_finetune_grouped
    )
    X_test_binary, y_test_binary = get_binary_subset(X_test, y_test_grouped)

    np.save(os.path.join(output_dir, "X_reference_binary.npy"), X_reference_binary)
    np.save(os.path.join(output_dir, "X_finetune_binary.npy"), X_finetune_binary)
    np.save(os.path.join(output_dir, "X_test_binary.npy"), X_test_binary)
    np.save(os.path.join(output_dir, "y_reference_binary.npy"), y_reference_binary)
    np.save(os.path.join(output_dir, "y_finetune_binary.npy"), y_finetune_binary)
    np.save(os.path.join(output_dir, "y_test_binary.npy"), y_test_binary)

    print(f"Saved processed arrays to {output_dir}")
    print(f"  X_reference: {X_reference.shape}")
    print(f"  X_finetune: {X_finetune.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  X_reference_binary: {X_reference_binary.shape}")
    print(f"  X_finetune_binary: {X_finetune_binary.shape}")
    print(f"  X_test_binary: {X_test_binary.shape}")


def resolve_org_dir():
    for org_dir in ORG_DIR_CANDIDATES:
        if all(os.path.exists(os.path.join(org_dir, fn)) for fn in REQUIRED_ORG_FILES):
            return org_dir

    searched = "\n".join(f"  - {path}" for path in ORG_DIR_CANDIDATES)
    required = ", ".join(REQUIRED_ORG_FILES)
    raise FileNotFoundError(
        "Could not find complete Bacteria-ID raw dataset.\n"
        f"Searched:\n{searched}\n"
        f"Required files: {required}"
    )


def preprocess_bacteria_split(
    wavenumber,
    X_reference,
    X_finetune,
    X_test,
    wave_number_in,
    apply_baseline=True,
    apply_savgol=True,
    parallel=True,
    n_jobs=20,
):
    """
    Bacteria-ID stores wavenumbers and spectra in descending order, so both are
    reversed before clipping and interpolation.
    """
    preprocess_kwargs = {
        "raman_shift": wavenumber[::-1],
        "wave_number_in": wave_number_in,
        "parallel": parallel,
        "n_jobs": n_jobs,
        "apply_baseline": apply_baseline,
        "apply_savgol": apply_savgol,
    }
    X_reference_processed = preprocess_data(
        peaks=X_reference[:, ::-1], **preprocess_kwargs
    )
    X_finetune_processed = preprocess_data(
        peaks=X_finetune[:, ::-1], **preprocess_kwargs
    )
    X_test_processed = preprocess_data(peaks=X_test[:, ::-1], **preprocess_kwargs)

    return X_reference_processed, X_finetune_processed, X_test_processed


def main():
    wave_number_in = np.linspace(WAVE_NUMBER_MIN, WAVE_NUMBER_MAX, N_WAVE_POINTS)
    org_dir = resolve_org_dir()
    print(f"Loading Bacteria-ID raw arrays from {org_dir}")

    wavenumber = np.load(os.path.join(org_dir, "wavenumbers.npy"))
    X_reference = np.load(os.path.join(org_dir, "X_reference.npy"))
    X_finetune = np.load(os.path.join(org_dir, "X_finetune.npy"))
    X_test = np.load(os.path.join(org_dir, "X_test.npy"))

    y_reference = np.load(os.path.join(org_dir, "y_reference.npy"))
    y_finetune = np.load(os.path.join(org_dir, "y_finetune.npy"))
    y_test = np.load(os.path.join(org_dir, "y_test.npy"))

    print(
        "Preprocessing Bacteria-ID with full pipeline: "
        "baseline correction + Savitzky-Golay + min-max + interpolation"
    )
    full_processed = preprocess_bacteria_split(
        wavenumber,
        X_reference,
        X_finetune,
        X_test,
        wave_number_in,
        apply_baseline=True,
        apply_savgol=True,
        parallel=True,
        n_jobs=20,
    )
    save_processed_dataset(
        FULL_PREPROCESSED_DIR,
        *full_processed,
        y_reference,
        y_finetune,
        y_test,
    )

    print(
        "Preprocessing Bacteria-ID with minimal pipeline: "
        "crop + min-max + interpolation"
    )
    minimal_processed = preprocess_bacteria_split(
        wavenumber,
        X_reference,
        X_finetune,
        X_test,
        wave_number_in,
        apply_baseline=False,
        apply_savgol=False,
        parallel=False,
        n_jobs=1,
    )
    save_processed_dataset(
        MINIMAL_PREPROCESSED_DIR,
        *minimal_processed,
        y_reference,
        y_finetune,
        y_test,
    )


if __name__ == "__main__":
    main()
