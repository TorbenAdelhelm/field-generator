import numpy as np
import os
import h5py
from typing import Optional, List

def save_field_h5(
    K: np.ndarray,
    fpath: str,
    dataset_name: str = "Permeability",
) -> None:
    """
    Save a single field 'K' to an HDF5 file with the exact structure:

        <dataset_name> : float64, flattened column-major (Fortran order)
        "Cell Ids"     : int32, values 1..N

    Parameters
    ----------
    K : np.ndarray
        2D field (Ny, Nx).
    fpath : str
        Output file path (.h5).
    dataset_name : str
        Name of the dataset for the permeability array.
    """
    # Flatten in Fortran order and cast to float64 (no copying if already float64)
    flat_perm = K.ravel(order="F").astype(np.float64, copy=False)
    # Note: the snippet you provided adds +1 (=> 1-based IDs).
    cell_ids = (np.arange(flat_perm.size, dtype=np.int32) + 1)

    os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
    with h5py.File(fpath, "w") as h5:
        # No compression/chunking to replicate the reference structure exactly
        h5.create_dataset(dataset_name, data=flat_perm)  # float64
        h5.create_dataset("Cell Ids",    data=cell_ids)  # int32

def save_ensemble_h5(
    K_all: np.ndarray,
    out_dir: str,
    *,
    filename_pattern: str = "field_{i:04d}.h5",
    dataset_name: str = "Permeability",
    seeds: Optional[List[int]] = None,
) -> List[str]:
    """
    Save an ensemble of fields as separate HDF5 files.

    Parameters
    ----------
    K_all : np.ndarray
        Array with shape (n_realizations, Ny, Nx).
    out_dir : str
        Output directory. Will be created if it does not exist.
    filename_pattern : str
        Pattern for filenames. Supports `{i}` (realization index) and,
        if `seeds` is provided, `{seed}`.
        Examples: "field_{i:04d}.h5", "perm_seed{seed}_{i:03d}.h5"
    dataset_name : str
        HDF5 dataset name for the permeability data.
    seeds : list[int], optional
        Per-realization seeds (e.g., from MasterRNG). If provided and the
        pattern contains `{seed}`, it will be used; otherwise `{seed}` is ignored.

    Returns
    -------
    paths : list[str]
        List of file paths written, in order of realizations.
    """
    if K_all.ndim != 3:
        raise ValueError("K_all must have shape (n_realizations, Ny, Nx).")
    n_realizations = K_all.shape[0]

    if ("{seed" in filename_pattern) and (seeds is None):
        raise ValueError("filename_pattern expects {seed}, but 'seeds' was not provided.")
    if (seeds is not None) and (len(seeds) != n_realizations):
        raise ValueError("Length of 'seeds' must match number of realizations in K_all.")

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    for i in range(n_realizations):
        fmt_kwargs = {"i": i}
        if seeds is not None:
            fmt_kwargs["seed"] = seeds[i]
        fname = filename_pattern.format(**fmt_kwargs)
        fpath = os.path.join(out_dir, fname)

        save_field_h5(K_all[i], fpath, dataset_name=dataset_name)
        paths.append(fpath)

    return paths