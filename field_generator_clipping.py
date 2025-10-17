# ---- Add to the clipped generator file ----
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gstools as gs
from gstools.random import MasterRNG


class ClippedLognormalFieldGenerator:
    """
    Minimal field generator (no copula):
      Z ~ SRF(kernel)          (mean 0, var 1 unless chosen otherwise)
      Y = mu10 + sigma10 * Z   (Gaussian in log10-space)
      K = 10**Y
      CLIP K into [a, b]
    """

    def __init__(self, bounds: tuple[float, float], mu10: float, sigma10: float):
        a, b = float(bounds[0]), float(bounds[1])
        if not (a > 0 and b > a):
            raise ValueError("bounds must satisfy 0 < a < b.")
        self.a, self.b = a, b
        self.mu10 = float(mu10)
        self.sigma10 = float(sigma10)

    def generate_field(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        *,
        len_scale: float,
        anis: float = 1.0,
        angles: float = 0.0,
        model_cls=gs.Exponential,
        seed: int | None = None,
    ) -> np.ndarray:
        model = model_cls(dim=2, var=self.sigma10**2, len_scale=len_scale, anis=anis, angles=angles)
        srf = gs.SRF(model, mean=self.mu10)
        srf.set_pos([grid_x, grid_y], "structured")
        Y = srf(seed=seed)

        K = np.power(10.0, Y)
        K = np.clip(K, self.a, self.b)
        return K

    def generate_ensemble(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        n_realizations: int,
        *,
        len_scale: float,
        anis: float = 1.0,
        angles: float = 0.0,
        model_cls=gs.Exponential,
        master_seed: int = 20170519,
    ) -> tuple[np.ndarray, list[int]]:
        master = MasterRNG(master_seed)
        Ks, seeds = [], []
        for _ in range(int(n_realizations)):
            s = master()
            seeds.append(s)
            K = self.generate_field(
                grid_x, grid_y,
                len_scale=len_scale, anis=anis, angles=angles,
                model_cls=model_cls, seed=s,
            )
            Ks.append(K)
        return np.stack(Ks, 0), seeds


# ---------- (1) Plot field + histograms (one page per realization) ----------
def save_field_and_histograms_pdf(
    K_ens: np.ndarray,
    *,
    bounds: tuple[float, float],
    pdf_path: str = "clipped_fields_with_hist.pdf",
    bins: int = 50,
    suptitle: str | None = "Clipped fields — map + histograms",
    dpi: int = 300,
    cmap: str = "viridis",
) -> str:
    """Each page: [spatial map]  [hist K]  [hist log10(K)]"""
    a, b = map(float, bounds)
    with PdfPages(pdf_path) as pdf:
        for i, K in enumerate(K_ens):
            vals = K.ravel()
            logvals = np.log10(vals, where=vals > 0)

            fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
            if suptitle:
                fig.suptitle(f"{suptitle} — realization {i}", fontsize=12)

            # (1) spatial map
            ax0 = axes[0]
            im = ax0.imshow(K, origin="lower", cmap=cmap, vmin=a, vmax=b, interpolation="nearest")
            ax0.set_title("Permeability field K")
            ax0.set_xticks([]); ax0.set_yticks([])
            cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
            cbar.set_label("K")

            # (2) histogram K
            ax1 = axes[1]
            ax1.hist(vals, bins=bins, density=True, alpha=0.85)
            ax1.axvline(a, color="r", ls="--", lw=1)
            ax1.axvline(b, color="r", ls="--", lw=1)
            ax1.set_title("Histogram of K (linear)")
            ax1.set_xlabel("K"); ax1.set_ylabel("Density")

            # (3) histogram log10(K)
            ax2 = axes[2]
            ax2.hist(logvals, bins=bins, density=True, alpha=0.85, color="darkorange")
            ax2.set_title("Histogram of log10(K)")
            ax2.set_xlabel("log10(K)"); ax2.set_ylabel("Density")

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
    return pdf_path


# ---------- (2) Check spatial parameters on log10(K) ----------
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

def main():
    # Grid
    x = y = np.linspace(0, 639, 640)  # unit spacing

    # Clipped generator (no copula)
    scaling = 1e-3/(1000*9.81)
    bounds = (1e-4*scaling, 5e-2*scaling)
    mu10 = 0.5 * (np.log10(bounds[0]) + np.log10(bounds[1]))
    sigma10 = 0.4
    clip_gen = ClippedLognormalFieldGenerator(bounds, mu10, sigma10)

   # Generate ensemble with known spatial params
    true_len, true_anis, true_angle = 70.0, 1.0, 0.0  # angle in radians here
    K_ens, seeds = clip_gen.generate_ensemble(
        x, y, n_realizations=20,
        len_scale=true_len, anis=true_anis, angles=true_angle,
        model_cls=gs.Exponential, master_seed=20251017,
    )

    # (1) Save per-realization pages: field + histograms
    pdf_path = save_field_and_histograms_pdf(
        K_ens, bounds=bounds, pdf_path="clipped_fields_with_hist.pdf", bins=60
    )
    print("Wrote:", pdf_path)

    # (2) Check spatial parameters on log10(K)
    results = validate_spatial_params_ensemble(
        K_ens, x, y,
        model_cls=gs.Exponential,
        n_lags=30,
        ignore_clipped=False,         # recommended for fairness
        bounds=bounds,
        check_anisotropy=False       # set False if you only care about isotropic len_scale
    )

    # Quick summary
    import pandas as pd
    df = pd.DataFrame(results)
    print(df.head(50))
    print(df[["len_scale","var","anis","angle"]].describe())
    print(f"Target len={true_len}, anis={true_anis}, angle(rad)={true_angle}")

if __name__ == "__main__":
    main()