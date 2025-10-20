import numpy as np
import gstools as gs
import pandas as pd
from typing import Tuple, Optional, List, Literal
from generators import to_latent_scores_from_K


def ensemble_to_fields_for_vario(
    K_ens: np.ndarray,
    *,
    mode: Literal["latent", "log10", "lin"],
    generator=None,
) -> List[np.ndarray]:
    """
    Convert an ensemble to the domain we want to variogram-fit on.
    - 'latent':    K -> Z using generator.truncated_dist (for copula fields)
    - 'log10':     K -> log10(K) (for clipped baseline)
    - 'lin':       K as-is
    Returns a Python list of 2D arrays (one per realization) ready for gstools.vario_estimate.
    """
    fields: List[np.ndarray] = []
    if mode == "latent":
        if generator is None:
            raise ValueError("generator required for mode='latent'")
        # local import to avoid circular deps
        from generators import to_latent_scores_from_K
        for K in K_ens:
            fields.append(to_latent_scores_from_K(K, generator.truncated_dist))
    elif mode == "log10":
        for K in K_ens:
            fields.append(np.log10(K, where=(K > 0)))
    elif mode == "lin":
        fields = [K for K in K_ens]
    else:
        raise ValueError("mode must be one of {'latent','log10','lin'}")
    return fields

# -------- binning helpers ------------------------------------------------
def auto_linear_bin_edges(x, y, len_scale: float, max_lag: Optional[float]=None,
                          q_per_len: int=8, min_lags: int=15, max_lags: int=60):
    Lx, Ly = float(x[-1]-x[0]), float(y[-1]-y[0])
    if max_lag is None:
        max_lag = 0.5 * np.hypot(Lx, Ly)
    n_lags = int(np.clip(round(q_per_len * max_lag / max(len_scale, 1e-12)),
                         min_lags, max_lags))
    edges = np.linspace(0.0, max_lag, n_lags+1)
    return edges

# -------- structured variograms (single field) --------------------------
def variogram_structured_isotropic(
    field: np.ndarray,
    x: np.ndarray, y: np.ndarray,
    *, bin_edges
) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic empirical variogram on a structured grid."""
    bc, gamma = gs.vario_estimate(
        (x, y), field, bin_edges,
        mesh_type="structured",
    )
    return bc, gamma

def variogram_structured_directional(
    field: np.ndarray,
    x: np.ndarray, y: np.ndarray,
    *, bin_edges, angle: float,
    angles_tol: float = np.pi/16,
    bandwidth: Optional[float] = 8,
    return_counts: bool = True
):
    """
    Directional empirical variogram on a structured grid using the two
    rotated main axes at the given angle (radians). Returns stacked curves:
      dir_gamma.shape == (2, n_bins)  and counts same shape (if requested).
    """
    args = {
        "direction": gs.rotated_main_axes(dim=2, angles=angle),
        "angles_tol": angles_tol,
        "bandwidth": bandwidth,
        "mesh_type": "structured",
        "return_counts": return_counts,
    }
    out = gs.vario_estimate((x, y), field, bin_edges, **args)
    if return_counts:
        bc, dir_gamma, counts = out
        return bc, dir_gamma, counts
    else:
        bc, dir_gamma = out
        return bc, dir_gamma, None

# -------- ensemble fitting: isotropic (structured) ----------------------
def fit_isotropic_ensemble_structured(
    K_ens: np.ndarray, x: np.ndarray, y: np.ndarray, *,
    data_scale: Literal["latent","log10","lin"]="log10",
    generator=None,                  # required if data_scale="latent"
    model_cls=gs.Exponential, nugget: bool=False,
    len_scale_hint: Optional[float]=None,
    max_lag: Optional[float]=None, q_per_len: int=8,
    return_models: bool=False
):
    """
    Fit an isotropic model to each field, return per-realization params and
    the ensemble-mean empirical variogram (for a single summary plot).
    """
    if max_lag is None:
        Lx, Ly = float(x[-1]-x[0]), float(y[-1]-y[0])
        max_lag = 0.5*np.hypot(Lx, Ly)
    if len_scale_hint is None:
        len_scale_hint = max_lag/3.0
    edges = auto_linear_bin_edges(x, y, len_scale_hint, max_lag=max_lag, q_per_len=q_per_len)

    rows, models, gammas = [], [], []
    bc_ref = None

    for i, K in enumerate(K_ens):
        if data_scale == "latent":
            if generator is None:
                raise ValueError("generator required for data_scale='latent'")
            field = to_latent_scores_from_K(K, generator.truncated_dist)
        elif data_scale == "log10":
            field = np.log10(K, where=(K>0))
        else:
            field = K

        bc, gamma = variogram_structured_isotropic(field, x, y, bin_edges=edges)
        bc_ref = bc if bc_ref is None else bc_ref
        fit = model_cls(dim=2)
        fit.fit_variogram(bc, gamma, nugget=nugget)

        rows.append({
            "realization": i,
            "var": float(getattr(fit, "var", np.nan)),
            "len_scale": float(np.atleast_1d(getattr(fit, "len_scale", np.nan))[0]),
            "anis": float(getattr(fit, "anis", np.nan)) if hasattr(fit, "anis") else np.nan,
            "angles": float(getattr(fit, "angles", 0.0)),
            "nugget": float(getattr(fit, "nugget", 0.0)) if nugget else 0.0,
        })
        gammas.append(gamma)
        if return_models: models.append(fit)

    df = pd.DataFrame(rows)
    gamma_mean = np.mean(np.vstack(gammas), axis=0)
    return (df, (bc_ref, gamma_mean)) if not return_models else (df, models, (bc_ref, gamma_mean))

# -------- ensemble fitting: directional (structured) --------------------
def fit_directional_ensemble_structured(
    K_ens: np.ndarray, x: np.ndarray, y: np.ndarray, *,
    angle: float,
    data_scale: Literal["latent","log10","lin"]="log10",
    generator=None,                  # required if data_scale='latent'
    model_cls=gs.Exponential,
    len_scale_hint: Optional[float]=None,
    max_lag: Optional[float]=None, q_per_len: int=8,
    angles_tol: float = np.pi/16, bandwidth: Optional[float] = 8,
    return_per_realization: bool=False
):
    """
    Compute directional variograms (two main axes) for each realization on a
    structured grid and fit a single anisotropic model to the *ensemble-mean*
    stacked directional variogram.
    Optionally returns per-realization anisotropic fits as well.
    """
    if max_lag is None:
        Lx, Ly = float(x[-1]-x[0]), float(y[-1]-y[0])
        max_lag = 0.5*np.hypot(Lx, Ly)
    if len_scale_hint is None:
        len_scale_hint = max_lag/3.0
    edges = auto_linear_bin_edges(x, y, len_scale_hint, max_lag=max_lag, q_per_len=q_per_len)

    dir_gammas = []
    bc_ref = None
    per_rows = []

    for i, K in enumerate(K_ens):
        if data_scale == "latent":
            if generator is None:
                raise ValueError("generator required for data_scale='latent'")
            field = to_latent_scores_from_K(K, generator.truncated_dist)
        elif data_scale == "log10":
            field = np.log10(K, where=(K>0))
        else:
            field = K

        bc, dir_gamma, _ = variogram_structured_directional(
            field, x, y, bin_edges=edges, angle=angle,
            angles_tol=angles_tol, bandwidth=bandwidth, return_counts=True
        )
        bc_ref = bc if bc_ref is None else bc_ref
        dir_gammas.append(dir_gamma[np.newaxis, ...])  # shape (1, 2, n_bins)

        if return_per_realization:
            fit_i = model_cls(dim=2)
            fit_i.fit_variogram(bc, dir_gamma)
            per_rows.append({
                "realization": i,
                "var": float(getattr(fit_i, "var", np.nan)),
                # len_scale can be scalar (major) or list [Lx, Ly]; save both robustly
                "len_major": float(np.atleast_1d(getattr(fit_i, "len_scale", np.nan))[0]),
                "len_minor": float(np.atleast_1d(getattr(fit_i, "len_scale", np.array([np.nan, np.nan])))[1]) \
                              if np.ndim(getattr(fit_i, "len_scale", np.nan)) else np.nan,
                "anis": float(getattr(fit_i, "anis", np.nan)) if hasattr(fit_i, "anis") else np.nan,
                "angles": float(getattr(fit_i, "angles", 0.0)),
            })

    dir_gamma_stack = np.concatenate(dir_gammas, axis=0)   # (n_real, 2, n_bins)
    dir_gamma_mean  = np.nanmean(dir_gamma_stack, axis=0)  # (2, n_bins)

    # Fit a single anisotropic model to the mean stacked curve
    fitted = model_cls(dim=2)
    fitted.fit_variogram(bc_ref, dir_gamma_mean)

    if return_per_realization:
        df_per = pd.DataFrame(per_rows)
        return fitted, (bc_ref, dir_gamma_mean), df_per
    return fitted, (bc_ref, dir_gamma_mean)
