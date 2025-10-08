import os

from typing import Dict, Tuple, Iterable, Generator, Optional, Literal, List
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

    # -----------------------------------------------------------------
    # Main generation method
    # -----------------------------------------------------------------
    def generate_field(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        *,
        var_kernel: float = 1.0,
        len_scale: float,
        angles: float = 0.0,
        anis: float = 1.0,
        seed: int | None = None,
        copula: Literal["gaussian", "t"] = "gaussian",
        nu: float | None = 5,
        standardize_latent: bool = True,
    ) -> np.ndarray:
        """
        Generate a truncated base-10 lognormal field using the chosen copula.

        Parameters
        ----------
        var_kernel : float
            Variance of latent Gaussian field (spatial variability control).
        len_scale, angles, anis : float
            Covariance parameters for GSTools Exponential kernel.
        seed : int | None
            Random seed for reproducibility.
        copula : {'gaussian', 't'}
            Spatial dependence type.
        nu : float | None
            Degrees of freedom for t-copula.
        standardize_latent : bool
            If True, re-standardize Z to mean=0, var=1 before mapping.

        Returns
        -------
        K : 2D ndarray, values in [a, b]
        """
        # 1) Latent Gaussian field
        model = gs.Exponential(dim=2, var=var_kernel, len_scale=len_scale,
                               angles=angles, anis=anis)
        srf = gs.SRF(model, mean=0.0, seed=seed)
        Z = srf.structured([grid_x, grid_y])  # Gaussian field

        if standardize_latent:
            Z = (Z - np.mean(Z)) / np.std(Z)

        # 2) Map to uniform via copula
        if copula == "gaussian":
            U = norm.cdf(Z)
        elif copula == "t":
            if nu is None or nu <= 0:
                raise ValueError("For copula='t', provide nu > 0.")
            rng = np.random.default_rng(seed)
            W = rng.chisquare(df=nu) / nu
            T = Z / np.sqrt(W)
            U = student_t.cdf(T, df=nu)
        else:
            raise ValueError("copula must be 'gaussian' or 't'.")

        U = np.clip(U, 1e-12, 1 - 1e-12)  # numerical safety

        # 3) Apply truncated-normal quantile and exponentiate
        Y = self.truncated_dist.ppf(U)
        K = np.power(10.0, Y)
        return K

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

def plot_field_diagnostics(K: np.ndarray, generator: TruncatedLog10LognormalFieldGenerator, *,
                           figsize=(15,5), log_scale=True, title=None):
    """
    Visualize a generated permeability field and its marginal statistics.

    Parameters
    ----------
    K : np.ndarray
        Generated permeability field (2D).
    generator : TruncatedLog10LognormalFieldGenerator
        The generator instance used (for access to a,b,mu10,sigma10,...).
    figsize : tuple
        Figure size.
    log_scale : bool
        Whether to also plot log10(K) histogram.
    title : str, optional
        Title for the figure.
    """

    a, b = generator.a, generator.b
    loga, logb = np.log10(a), np.log10(b)
    mu10, sigma10 = generator.mu10, generator.sigma10

    # Flatten values
    vals = K.flatten()
    logvals = np.log10(vals)

    # Compute empirical stats
    emp_mean, emp_std = np.mean(logvals), np.std(logvals)
    theo_mean, theo_var = truncnorm_moments(mu10, sigma10, loga, logb)
    theo_std = np.sqrt(theo_var)

    # ---- Figure layout ----
    fig, axes = plt.subplots(1, 3 if log_scale else 2, figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=13)

    # --- (1) Spatial field map ---
    ax = axes[0]
    im = ax.imshow(K, origin="lower", cmap="viridis",
                   vmin=a, vmax=b, interpolation="nearest")
    ax.set_title("Permeability Field K")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="K")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # --- (2) Histogram of K ---
    ax = axes[1]
    ax.hist(vals, bins=50, color="steelblue", alpha=0.8, density=True)
    ax.axvline(a, color="r", ls="--", lw=1)
    ax.axvline(b, color="r", ls="--", lw=1)
    ax.set_title("Histogram of K (linear)")
    ax.set_xlabel("K"); ax.set_ylabel("Density")

    # --- (3) Histogram of log10(K) ---
    if log_scale:
        ax = axes[2]
        ax.hist(logvals, bins=50, color="darkorange", alpha=0.8, density=True)
        x = np.linspace(loga, logb, 200)
        pdf = generator.truncated_dist.pdf(x)
        ax.plot(x, pdf, "k--", lw=1.5, label="Truncated normal PDF")
        ax.legend()
        ax.set_xlabel("log10(K)")
        ax.set_title("Histogram of log10(K)")

    # ---- Print numerical summary ----
    print("\n--- Truncated log10(K) statistics ---")
    print(f"Theoretical mean:     {theo_mean: .4f}")
    print(f"Theoretical std:      {theo_std: .4f}")
    print(f"Empirical mean:       {emp_mean: .4f}")
    print(f"Empirical std:        {emp_std: .4f}")
    print(f"Relative error (mean/std): "
          f"{(emp_mean-theo_mean)/theo_mean:+.2%}, "
          f"{(emp_std-theo_std)/theo_std:+.2%}")
    print(f"Coverage within bounds [{a:.2e}, {b:.2e}]: "
          f"{np.mean((vals>=a)&(vals<=b))*100:.2f}%")

    plt.tight_layout()
    plt.show()


def main():
    x = y = np.linspace(0, 128, 128)
    scaling = 1e-3/(1000*9.81)
    bounds = (1e-4*scaling, 1e-2*scaling)
    mu10 = 0.5 * (np.log10(bounds[0]) + np.log10(bounds[1]))
    sigma10 = 0.3

    gen = TruncatedLog10LognormalFieldGenerator(
        bounds,
        mu10=mu10,
        sigma10=sigma10,
        match_moments=True,
        target_mean10=mu10,
        target_var10=sigma10**2,
    )

    K_ens, seeds = gen.generate_ensemble(
        x, y, n_realizations=100,
        len_scale=40.0,
        anis=1.0,
        var_kernel=1.0,              # typically leave at 1.0 (standardize_latent=True)
        copula="gaussian",
        master_seed=20170519,
        store_prefix=None,  # optional: SRF will store the latent Z draws
        return_latent=False,
    )

    #plot_field_diagnostics(K_ens[0], gen, title=f"Realization 0 (seed={seeds[0]})")

    out_paths = gen.save_ensemble_h5(
        K_ens,
        seeds=seeds,
        out_dir="/scratch/adelhetn/data/vampireman/permeability-input-fields",
        filename_pattern="permeability_seed{seed}_{i:04}.h5",
        dataset_name="Permeability"
    )

if __name__ == "__main__":
    main()