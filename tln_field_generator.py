import os

from typing import Dict, Tuple, Iterable, Generator, Optional, Literal, List, Type
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gstools as gs
from scipy.stats import truncnorm, norm, t as student_t
from scipy.optimize import root

# ---------------------------------------------------------------------
# Utility: mean and variance of a truncated normal
# ---------------------------------------------------------------------
def truncnorm_moments(mu: float, sigma: float, a: float, b: float) -> Tuple[float, float]:
    """Return mean and variance of a truncated normal N(mu, sigma^2) truncated to [a,b]."""
    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma
    Z = norm.cdf(b_std) - norm.cdf(a_std)
    phi_a, phi_b = norm.pdf(a_std), norm.pdf(b_std)
    mean = mu + sigma * (phi_a - phi_b) / Z
    var = sigma**2 * (1 + (a_std * phi_a - b_std * phi_b) / Z - ((phi_a - phi_b) / Z)**2)
    return mean, var


def match_truncnorm_meanvar(a: float, b: float, target_mean: float, target_var: float) -> Tuple[float, float]:
    """
    Find (mu, sigma) such that the *truncated* N(mu, sigma^2) on [a,b]
    has mean=target_mean and var=target_var.
    """
    def equations(params):
        mu, sigma = params
        m, v = truncnorm_moments(mu, sigma, a, b)
        return [m - target_mean, v - target_var]
    sol = root(equations, x0=[target_mean, np.sqrt(target_var)])
    if not sol.success:
        raise RuntimeError("Moment-matching failed to converge.")
    return sol.x


class TruncatedLog10LognormalFieldGenerator:
    """
    Truncated base-10 lognormal permeability fields:

      Y = log10(K) ~ TruncNorm(mu10, sigma10^2, [log10(a), log10(b)])
      => K in [a, b]

    Supports:
      - Mean/variance matching for truncated law.
      - Gaussian or t-copula spatial dependence.
      - Adjustable spatial variance (var_kernel).
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        mu10: float | None = None,
        sigma10: float | None = None,
        *,
        match_moments: bool = False,
        target_mean10: float | None = None,
        target_var10: float | None = None,
    ):
        """
        Parameters
        ----------
        bounds : (a, b)
            Truncation limits for K (0 < a < b).
        mu10, sigma10 : float, optional
            Parameters of the *underlying normal* for log10(K).
        match_moments : bool
            If True, adjusts (mu10, sigma10) so that truncated
            mean/variance match target_mean10/target_var10.
        target_mean10, target_var10 : float, optional
            Desired mean and variance of the truncated log10(K).
            Required if match_moments=True.
        """
        self.a, self.b = bounds
        loga, logb = np.log10(self.a), np.log10(self.b)

        # Optional moment-matching step
        if match_moments:
            if target_mean10 is None or target_var10 is None:
                raise ValueError("Specify target_mean10 and target_var10 when match_moments=True.")
            mu10, sigma10 = match_truncnorm_meanvar(loga, logb, target_mean10, target_var10)

        if mu10 is None or sigma10 is None:
            raise ValueError("Provide mu10 and sigma10 or enable match_moments.")

        self.mu10 = float(mu10)
        self.sigma10 = float(sigma10)
        self.a_std = (loga - self.mu10) / self.sigma10
        self.b_std = (logb - self.mu10) / self.sigma10

        self.truncated_dist = truncnorm(
            a=self.a_std, b=self.b_std, loc=self.mu10, scale=self.sigma10
        )

    def generate_ensemble(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        n_realizations: int,
        *,
        len_scale: float,
        anis: float = 1.0,
        var_kernel: float = 1.0,
        copula: Literal["gaussian", "t"] = "gaussian",
        nu: float | None = 5,
        standardize_latent: bool = True,
        master_seed: int = 20170519,
        store_prefix: str | None = None,
        return_latent: bool = False,
    ) -> Tuple[np.ndarray, List[int]] | Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Generate an ensemble of permeability fields on a structured grid.

        Parameters
        ----------
        grid_x, grid_y : 1D arrays
            Structured grid coordinates (GSTools "structured" layout).
        n_realizations : int
            Number of fields to generate.
        len_scale : float
            Correlation length of the Exponential kernel.
        anis : float
            Anisotropy ratio (>0). Angle is fixed to 0.0 in this method.
        var_kernel : float
            Variance of the latent Gaussian field (typically keep at 1.0).
        copula : {'gaussian', 't'}
            Dependence family for the copula transform.
        nu : float | None
            Degrees of freedom for t-copula (ignored for Gaussian).
        standardize_latent : bool
            If True, standardize each latent draw (mean=0, std=1) before mapping.
        master_seed : int
            Seed for gstools.random.MasterRNG to make the ensemble reproducible.
        store_prefix : str | None
            If provided, also store each latent Z draw in the SRF with this prefix.
        return_latent : bool
            If True, also return the latent Gaussian fields (after standardization).

        Returns
        -------
        K_all : ndarray, shape (n_realizations, Ny, Nx)
            Ensemble of permeability fields.
        used_seeds : list[int]
            The per-draw seeds produced by MasterRNG.
        (optional) Z_all : ndarray, shape (n_realizations, Ny, Nx)
            The latent Gaussian fields used to generate the ensemble.
        """
        # 1) Build kernel
        model = gs.Exponential(dim=2, var=var_kernel, len_scale=len_scale,
                               angles=0.0, anis=anis)

        # 2) One SRF instance reused for all draws; fixed grid positions
        srf = gs.SRF(model, mean=0.0)
        srf.set_pos([grid_x, grid_y], "structured")

        # 3) Master RNG for reproducible per-realization seeds
        master = gs.random.MasterRNG(master_seed)

        # 4) Allocate lazily after first draw (shape depends on grid ordering)
        K_list: List[np.ndarray] = []
        Z_list: List[np.ndarray] = []
        used_seeds: List[int] = []

        for i in range(n_realizations):
            draw_seed = master()
            used_seeds.append(draw_seed)

            # Draw latent Gaussian field on the fixed grid
            if store_prefix is not None:
                Z = srf(seed=draw_seed, store=f"{store_prefix}{i}")
            else:
                Z = srf(seed=draw_seed)

            # Optional standardization (recommended for correct copula mapping)
            if standardize_latent:
                Z = (Z - Z.mean()) / Z.std()

            # Copula mapping to uniforms
            if copula == "gaussian":
                U = norm.cdf(Z)
            elif copula == "t":
                if nu is None or nu <= 0:
                    raise ValueError("For copula='t', provide nu > 0.")
                # Use a numpy RNG tied to the same master sequence for reproducibility
                rng = np.random.default_rng(draw_seed)
                W = rng.chisquare(df=nu) / nu  # global radial factor per realization
                T = Z / np.sqrt(W)
                U = student_t.cdf(T, df=nu)
            else:
                raise ValueError("copula must be 'gaussian' or 't'.")

            # Numerical safety and inverse-CDF to truncated normal in log10-space
            U = np.clip(U, 1e-12, 1 - 1e-12)
            Y = self.truncated_dist.ppf(U)
            K = np.power(10.0, Y)

            K_list.append(K)
            if return_latent:
                Z_list.append(Z)

        K_all = np.stack(K_list, axis=0)
        if return_latent:
            Z_all = np.stack(Z_list, axis=0)
            return K_all, Z_all, used_seeds
        return K_all, used_seeds

    @staticmethod
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
        self,
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

            self.save_field_h5(K_all[i], fpath, dataset_name=dataset_name)
            paths.append(fpath)

        return paths

def save_ensemble_diagnostics_single_page_pdf(
    K_ens: np.ndarray,
    generator,
    pdf_path: str,
    *,
    log_scale: bool = True,
    bins: int = 50,
    cmap: str = "viridis",
    titles: list[str] | None = None,
    seeds: list[int] | None = None,
    figsize: tuple[float, float] | None = None,
    per_block_size: tuple[float, float] = (3.6, 2.8),
    suptitle: str | None = None,
    dpi: int = 300,
) -> str:
    """
    Save a single-PAGE PDF showing, for each realization, the spatial map and histograms
    (linear K and log10(K) with truncated-normal PDF overlay).

    Parameters
    ----------
    K_ens : ndarray
        Shape (n_realizations, Ny, Nx).
    generator : TruncatedLog10LognormalFieldGenerator
        Used for marginal params and color limits (a, b; mu10, sigma10; truncated_dist).
    pdf_path : str
        Output file path (.pdf).
    log_scale : bool
        If True, include the log10(K) histogram with truncated-normal PDF overlay.
    bins : int
        Histogram bins.
    cmap : str
        Colormap for spatial plots.
    titles : list[str], optional
        Per-realization titles. If provided, overrides seeds-based titles.
    seeds : list[int], optional
        Per-realization seeds for titles, if desired.
    figsize : (W, H) in inches, optional
        Overall figure size. If None, computed from per_block_size and grid shape.
    per_block_size : (w, h)
        Size (inches) per realization row and per column set; used to auto-compute figsize.
    suptitle : str, optional
        Global title for the page.
    dpi : int
        Rendering DPI for the saved page.

    Returns
    -------
    pdf_path : str
        The written file path.
    """
    if K_ens.ndim != 3:
        raise ValueError("K_ens must have shape (n_realizations, Ny, Nx).")
    n = K_ens.shape[0]
    if titles is not None and len(titles) != n:
        raise ValueError("len(titles) must equal number of realizations.")
    if seeds is not None and len(seeds) != n:
        raise ValueError("len(seeds) must equal number of realizations.")

    # Columns: [spatial, hist(K), hist(log10 K) optional]
    n_cols = 3 if log_scale else 2
    n_rows = n

    # Auto figure size if not provided
    if figsize is None:
        w_per_col, h_per_row = per_block_size
        figsize = (w_per_col * n_cols, h_per_row * n_rows)

    a, b = generator.a, generator.b
    loga, logb = np.log10(a), np.log10(b)
    mu10, sigma10 = generator.mu10, generator.sigma10

    # Precompute theoretical truncated-normal PDF on log10 scale (for overlay)
    x_log = np.linspace(loga, logb, 200)
    pdf_log = generator.truncated_dist.pdf(x_log)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    im_last = None

    for i in range(n_rows):
        K = K_ens[i]
        vals = K.ravel()
        logvals = np.log10(vals)

        # --- (1) Spatial map ---
        ax0 = axes[i, 0]
        im_last = ax0.imshow(
            K, origin="lower", cmap=cmap, vmin=a, vmax=b, interpolation="nearest"
        )
        if titles is not None:
            ax0.set_title(titles[i], fontsize=9)
        elif seeds is not None:
            ax0.set_title(f"Realization {i} (seed={seeds[i]})", fontsize=9)
        else:
            ax0.set_title(f"Realization {i}", fontsize=9)
        ax0.set_xticks([]); ax0.set_yticks([])

        # --- (2) Histogram of K (linear) ---
        ax1 = axes[i, 1]
        ax1.hist(vals, bins=bins, density=True, color="steelblue", alpha=0.8)
        ax1.axvline(a, color="r", ls="--", lw=1)
        ax1.axvline(b, color="r", ls="--", lw=1)
        if i == n_rows - 1:
            ax1.set_xlabel("K")
        ax1.set_ylabel("Density" if i == 0 else "")
        if i == 0:
            ax1.set_title("Histogram of K (linear)", fontsize=9)

        # --- (3) Histogram of log10(K) with PDF overlay ---
        if log_scale:
            ax2 = axes[i, 2]
            ax2.hist(logvals, bins=bins, density=True, color="darkorange", alpha=0.8)
            ax2.plot(x_log, pdf_log, "k--", lw=1.2, label="Trunc. normal PDF")
            if i == 0:
                ax2.legend(fontsize=8, frameon=False)
                ax2.set_title("Histogram of log10(K)", fontsize=9)
            if i == n_rows - 1:
                ax2.set_xlabel("log10(K)")

        # Optionally, you could annotate summary stats compactly on the right,
        # but to keep the page readable for many realizations, we omit text here.

    # Shared colorbar for all spatial maps (leftmost column)
    if im_last is not None:
        cax = fig.add_axes([0.92, 0.1, 0.015, 0.8])  # manual thin colorbar on the right
        cbar = fig.colorbar(im_last, cax=cax)
        cbar.set_label("Permeability K", fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.995)

    # Spacing tuned for many rows
    fig.subplots_adjust(left=0.05, right=0.90, top=0.97, bottom=0.05, wspace=0.25, hspace=0.35)

    # Save a single page
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf_path

import pandas as pd

def fit_variogram_ensemble(
    K_ens: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    data_scale: str = "log10",         # "log10" (recommended) or "lin"
    model_cls: Type[gs.CovModel] = gs.Stable,
    nugget: bool = False,
    n_lags: int = 30,
    max_lag: Optional[float] = None,   # default: 0.5 * domain diagonal
    return_models: bool = False,
    len_scale: float = 70,
    angle: float = 0,
) -> Tuple[pd.DataFrame, Optional[List[gs.CovModel]], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Fit an isotropic variogram model to each field in an ensemble.

    Parameters
    ----------
    K_ens : ndarray
        Shape (n_realizations, Ny, Nx). Permeability fields.
    x, y : ndarray
        1D coordinate arrays defining the structured grid axes.
    data_scale : {"log10","lin"}
        Work on log10(K) (recommended for your pipeline) or raw K.
    model_cls : gstools covariance model class
        e.g., gs.Stable (default), gs.Exponential, gs.Matern, ...
    nugget : bool
        Fit a nugget (True) or not (False).
    n_lags : int
        Number of lag bins for empirical variogram.
    max_lag : float or None
        Maximum lag to consider. If None, uses 0.5 * domain diagonal.
    return_models : bool
        If True, also return the list of fitted model instances and
        the common (bin_center, ensemble_mean_gamma).

    Returns
    -------
    df : pandas.DataFrame
        One row per realization with fitted parameters:
        ["realization", "var", "len_scale", "nugget", "alpha"] (where available).
    models : list[gs.CovModel] or None
        The fitted model objects (if return_models=True).
    (bin_center, gamma_mean) : tuple or None
        The common bin centers and the ensemble-mean empirical gamma
        (if return_models=True).
    """
    if K_ens.ndim != 3:
        raise ValueError("K_ens must have shape (n_realizations, Ny, Nx).")
    n, Ny, Nx = K_ens.shape
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays (structured grid axes).")
    if Ny != y.size or Nx != x.size:
        raise ValueError("K_ens.shape must match len(y), len(x).")

    # Build one common set of lag bin edges (shared across ensemble)
    Lx = float(x[-1] - x[0])
    Ly = float(y[-1] - y[0])
    domain_diag = np.hypot(Lx, Ly)
    if max_lag is None:
        max_lag = 0.5 * domain_diag
    if len_scale is not None:
        bin_edges, n_lags = auto_linear_bin_edges(x, y, max_lag=max_lag, len_scale=len_scale, q_per_len=8)
    else:
        bin_edges = np.linspace(0.0, max_lag, n_lags + 1)


    # Pre-allocate containers
    rows = []
    models: List[gs.CovModel] = []
    gamma_stack = []

    # Coordinates for structured variogram estimation
    # Use the structured estimator for performance/consistency.
    pos_structured = [x, y]

    for i in range(n):
        K = K_ens[i]
        field = np.log10(K) if data_scale.lower() == "log10" else K

        bin_center, gamma = gs.vario_estimate(
            (x, y), field, bin_edges=bin_edges, #angles_tol=np.pi / 16,
        )

        gamma_stack.append(gamma)

        # Fit the chosen model class
        fit_model = model_cls(dim=2, len_scale=len_scale)
        fit_model.fit_variogram(bin_center, gamma, nugget=nugget)

        # Collect parameters (only those that exist on the chosen model)
        row = {
            "realization": i,
            "var": getattr(fit_model, "var", np.nan),
            "len_scale": getattr(fit_model, "len_scale", np.nan),
            "anis": float(getattr(fit_model, "anis", np.nan)),
            "angle": float(getattr(fit_model, "angles", np.nan)),
            "nugget": getattr(fit_model, "nugget", 0.0) if nugget else 0.0,
        }
        # Stable/Matern carry shape parameters; others may not.
        row["alpha"] = getattr(fit_model, "alpha", np.nan)
        row["nu"]    = getattr(fit_model, "nu",    np.nan)

        rows.append(row)
        if return_models:
            models.append(fit_model)

    df = pd.DataFrame(rows)
    gamma_stack = np.vstack(gamma_stack)  # (n, n_lags)

    if return_models:
        gamma_mean = gamma_stack.mean(axis=0)
        return df, models, (bin_center, gamma_mean)
    return df, None, None

def auto_linear_bin_edges(x, y, len_scale, max_lag=None, q_per_len=8,
                          min_lags=15, max_lags=60):
    """
    Linear binning with n_lags ≈ q_per_len * (max_lag / len_scale),
    clipped to [min_lags, max_lags].
    """
    Lx, Ly = float(x[-1]-x[0]), float(y[-1]-y[0])
    if max_lag is None:
        max_lag = 0.5 * np.hypot(Lx, Ly)
    n_lags = int(np.clip(np.round(q_per_len * max_lag / len_scale),
                         min_lags, max_lags))
    edges = np.linspace(0.0, max_lag, n_lags + 1)
    return edges, n_lags

def plot_ensemble_variogram_summary(
    bin_center: np.ndarray,
    gamma_mean: np.ndarray,
    df_params: pd.DataFrame,
    model_cls: Type[gs.CovModel] = gs.Stable,
    title: str = "Ensemble variogram: empirical mean vs. median-fit model",
):
    """
    Plot ensemble-mean empirical variogram and the model fitted with
    median parameters across realizations (robust representative).

    Parameters
    ----------
    bin_center : ndarray
        Common lag bin centers (from fit_variogram_ensemble return).
    gamma_mean : ndarray
        Ensemble-mean empirical semivariances.
    df_params : DataFrame
        Output from fit_variogram_ensemble (one row per realization).
    model_cls : gstools covariance model class
        Same type used in fitting.
    title : str
        Plot title.
    """
    # median parameters (robust against outliers)
    var_med      = np.nanmedian(df_params["var"].values)
    len_med      = np.nanmedian(df_params["len_scale"].values)
    anis_med     = np.nanmedian(df_params["anis"].values)
    angle_med    = np.nanmedian(df_params["angle"].values)
    nugget_med   = np.nanmedian(df_params["nugget"].values)
    alpha_med    = np.nanmedian(df_params["alpha"].values) if "alpha" in df_params else np.nan
    nu_med       = np.nanmedian(df_params["nu"].values)    if "nu" in df_params else np.nan

    # instantiate model with available parameters
    kwargs = {"dim": 2, "var": var_med, "len_scale": len_med, "anis": anis_med, "angle": angle_med}
    if not np.isnan(nugget_med):
        kwargs["nugget"] = nugget_med
    if not np.isnan(alpha_med) and hasattr(model_cls, "alpha"):
        kwargs["alpha"] = float(alpha_med)
    if not np.isnan(nu_med) and hasattr(model_cls, "nu"):
        kwargs["nu"] = float(nu_med)

    model = model_cls(**kwargs)

    # model variogram curve on a dense lag grid for smoothness
    h_dense = np.linspace(0, bin_center.max(), 400)
    gamma_model = model.variogram(h_dense)

    # plot
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(bin_center, gamma_mean, "o", label="Empirical (ensemble mean)")
    plt.plot(h_dense, gamma_model, "-", label=f"{model_cls.__name__} (median fit)")
    plt.xlabel("Lag h")
    plt.ylabel("Semivariance γ(h)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig()


def main():
    x = y = np.linspace(0, 639, 640)
    scaling = 1e-3/(1000*9.81)
    bounds = (1e-4*scaling, 5e-2*scaling)
    mu10 = 0.5 * (np.log10(bounds[0]) + np.log10(bounds[1]))
    sigma10 = 0.4
    len_scale = 70.0

    gen = TruncatedLog10LognormalFieldGenerator(
        bounds,
        mu10=mu10,
        sigma10=sigma10,
        match_moments=True,
        target_mean10=mu10,
        target_var10=sigma10**2,
    )

    K_ens, seeds = gen.generate_ensemble(
        x, y, n_realizations=20,
        len_scale=len_scale,
        anis=1.0,
        var_kernel=1.0,              # typically leave at 1.0 (standardize_latent=True)
        copula="gaussian",
        master_seed=20251017,
        store_prefix=None,  # optional: SRF will store the latent Z draws
        return_latent=False,
    )

    save_ensemble_diagnostics_single_page_pdf(
        K_ens,
        generator=gen,
        pdf_path="ensemble_diagnostics.pdf",
        log_scale=True,
        bins=50,
        titles=None,          # or provide custom titles per realization
        seeds=seeds,          # optional: shows seed in the left-panel title
        suptitle="Permeability Ensemble — Spatial Maps and Histograms",
        dpi=300,
    )

    # Assume: x, y (1D axes), and K_ens with shape (n_realizations, Ny, Nx)
    # Prefer log10(K) for variogram fitting in your pipeline:
    df, models, (h, gamma_mean) = fit_variogram_ensemble(
        K_ens, x, y,
        data_scale="log10",
        model_cls=gs.Exponential,   # or gs.Exponential, gs.Matern, ...
        nugget=False,
        n_lags=30,
        max_lag=None,          # defaults to 0.5 * domain diagonal
        return_models=True,
        len_scale=len_scale
    )

    print(df.head(100))  # fitted parameters per realization

    # Quick sanity plot: ensemble mean empirical vs. "median" fitted model
    plot_ensemble_variogram_summary(
        h, gamma_mean, df, model_cls=gs.Stable,
        title="Log10(K) variogram — empirical mean vs. median Stable fit"
    )

    print("done")


    # out_paths = gen.save_ensemble_h5(
    #     K_ens,
    #     seeds=seeds,
    #     out_dir="/scratch/adelhetn/data/vampireman/permeability-input-fields",
    #     filename_pattern="permeability_seed{seed}_{i:04}.h5",
    #     dataset_name="Permeability"
    # )

if __name__ == "__main__":
    main()