import numpy as np
import gstools as gs
def validate_spatial_params_ensemble(
    K_ens: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_cls = gs.Exponential,
    n_lags: int = 30,
    max_lag: float | None = None,
    ignore_clipped: bool = False,
    bounds: tuple[float, float] | None = None,
    check_anisotropy: bool = False,
    azimuths: tuple = (0, 45, 90, 135),
    half_angle_deg: float = 22.5,
):
    """
    Fit variograms to log10(K) for each realization to recover spatial params.

    Returns
    -------
    results : list of dict
        [{'len_scale': ..., 'var': ..., 'realization': i, ...}, ...]
        If check_anisotropy=True, also returns 'len_major','anis','angle_rad'.
    """
    results = []
    for i, K in enumerate(K_ens):
        
        h, gamma = _empirical_variogram_log10(
            K, x, y, n_lags=n_lags, max_lag=max_lag,
            ignore_clipped=ignore_clipped, bounds=bounds
        )
        fit_iso = model_cls(dim=2)
        fit_iso.fit_variogram(h, gamma, nugget=False)
        row = {
            "realization": i,
            "len_scale": float(getattr(fit_iso, "len_scale", np.nan)),
            "var": float(getattr(fit_iso, "var", np.nan)),
            "anis": float(getattr(fit_iso, "anis", np.nan)),
            "angle": float(getattr(fit_iso, "angle", np.nan))
        }

        results.append(row)
    return results

def _empirical_variogram_log10(
    K: np.ndarray, x: np.ndarray, y: np.ndarray,
    *, n_lags: int = 30, max_lag: float | None = None,
    angle: float | None = 0,
    ignore_clipped: bool = False, bounds: tuple[float, float] | None = None,
):
    """Empirical isotropic variogram of log10(K). Can ignore clipped pixels (recommended)."""
    Y = np.log10(K, where=(K > 0))
    Ny, Nx = K.shape
    if max_lag is None:
        max_lag = 0.5 * np.hypot(float(x[-1]-x[0]), float(y[-1]-y[0]))
    # geometric bins: more resolution near 0
    ratio = 1.12
    k = np.arange(n_lags + 1)
    edges = max_lag * (ratio**k - 1) / (ratio**n_lags - 1)

    if ignore_clipped and bounds is not None:
        a, b = bounds
        mask = (K > a) & (K < b)
        iy, ix = np.where(mask)
        pts = (x[ix], y[iy])
        vals = Y[iy, ix]
        try:
            h, gamma = gs.vario_estimate_unstructured(pts, vals, bin_edges=edges)
        except AttributeError:
            h, gamma = gs.vario_estimate(pts, vals, bin_edges=edges)
    else:
        # use full grid
        try:
            h, gamma = gs.vario_estimate((x, y), Y, edges)#, direction=gs.rotated_main_axes(dim=2, angles=angle))#[x, y], Y, bin_edges=edges)
        except AttributeError:
            Xg, Yg = np.meshgrid(x, y, indexing="xy")
            h, gamma = gs.vario_estimate((x, y), Y, bin_edges=edges)
    return h, gamma
