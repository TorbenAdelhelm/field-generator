import os
import itertools
import hashlib
from typing import Dict, Tuple, Iterable, Generator, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gstools as gs
from scipy.stats import truncnorm, norm, t as student_t


class TruncatedLog10LognormalFieldGenerator:
    """
    Truncated base-10 lognormal permeability fields:
      Y = log10(K) ~ N(mu10, sigma10^2), truncated to [log10(a), log10(b)]
      => K in [a, b]
    Includes: multi-realizations, plotting, histograms, flattened HDF5 saving.
    """

    def __init__(self, bounds: Tuple[float, float], mu10: float = None, sigma10: float = None, alpha: float = None):
        """
        Parameters
        ----------
        bounds : (a, b) with 0 < a < b
        mu10, sigma10 : parameters of log10(K). If not provided, they are inferred
                        from bounds treating [a,b] as (alpha, 1-alpha) quantiles.
        alpha : float in (0, 0.5), used to infer (mu10, sigma10) if not provided.
        """
        self.a, self.b = bounds
        if not (np.isfinite(self.a) and np.isfinite(self.b) and 0 < self.a < self.b):
            raise ValueError("bounds must satisfy 0 < a < b and be finite.")

        if mu10 is not None and sigma10 is not None:
            self.mu10, self.sigma10 = float(mu10), float(sigma10)
        else:
            if alpha is None:
                raise ValueError("Provide (mu10, sigma10) or alpha to infer them from bounds.")
            z_lo, z_hi = norm.ppf(alpha), norm.ppf(1 - alpha)
            self.mu10 = 0.5 * (np.log10(self.a) + np.log10(self.b))
            self.sigma10 = (np.log10(self.b) - np.log10(self.a)) / (z_hi - z_lo)

        if not (self.sigma10 > 0):
            raise ValueError("sigma10 must be positive.")

        # Calculate standardized truncation bounds
        # truncnorm uses standardized bounds: (x - mu)/sigma
        self.a_std = (np.log10(self.a) - self.mu10) / self.sigma10
        self.b_std = (np.log10(self.b) - self.mu10) / self.sigma10

        # Create the truncated normal distribution object
        self.truncated_dist = truncnorm(
            a=self.a_std,
            b=self.b_std,
            loc=self.mu10,
            scale=self.sigma10
        )

        trunc_var = self.truncated_dist.var()
        # Store CDF values at bounds (optional, for reference)
        self.Fa = self.truncated_dist.cdf(np.log10(self.a))
        self.Fb = self.truncated_dist.cdf(np.log10(self.b))

        # # Truncated CDF bounds in log10-space
        # self.Fa = norm.cdf((np.log10(self.a) - self.mu10) / self.sigma10)
        # self.Fb = norm.cdf((np.log10(self.b) - self.mu10) / self.sigma10)

        # margin = 1e-10
        # self.Fa = self.Fa + margin if self.Fa > 0 else margin
        # self.Fb = self.Fb - margin if self.Fb < 1 else 1 - margin

    # -------------------------------------------------
    # Helper: build/create per-configuration subfolder
    # -------------------------------------------------
    def _config_subdir(
        self,
        out_dir: str,
        var: float,
        ls: float,
        anis: float,
        ang: float,
        subdir_pattern: str = "cfg_var{var:g}_ls{ls:g}_anis{anis:g}_ang{ang:g}",
    ) -> str:
        subdir = os.path.join(
            out_dir, subdir_pattern.format(var=var, ls=ls, anis=anis, ang=ang)
        )
        os.makedirs(subdir, exist_ok=True)
        return subdir

    # -----------------------
    # Core field generation
    # -----------------------
    def _stable_seed(self, key: Tuple, base_seed: Optional[int]) -> Optional[int]:
        if base_seed is None:
            return None
        s = f"{base_seed}|" + "|".join(map(str, key))
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little") % (2**31 - 1)

    def generate_field(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        var: float,
        len_scale: float,
        angles: float = 0.0,
        anis: float = 1.0,
        seed: int | None = None,
        *,
        copula: str = "gaussian",
        nu: float | None = 5,
    ) -> np.ndarray:
        """
        Generate a single truncated base-10 lognormal field using the chosen copula.

        Parameters
        ----------
        var, len_scale, angles, anis : covariance params for the latent field Z
        seed : int | None
            Reproducible seed (used for Z and, if copula='t', also for chi-square draw).
        copula : {'gaussian', 't'}
            - 'gaussian': U = Phi(Z/sqrt(var))
            - 't'      : U = F_t( Z / sqrt(W/nu) ), W ~ Chi^2_nu  (elliptical t-copula)
        nu : float | None
            Degrees of freedom for the t-copula (required if copula='t').

        Returns
        -------
        K : 2D ndarray with values in [a, b]
        """
        # 1) Latent Gaussian field with desired spatial covariance
        model = gs.Exponential(dim=2, var=1.0, len_scale=len_scale, angles=angles, anis=anis)
        srf = gs.SRF(model, mean=0.0, seed=seed)
        Z = srf.structured([grid_x, grid_y])   # Z ~ N(0, var) pointwise

        # 2) Map to uniforms according to the chosen copula
        if copula == "gaussian":
            # Probability integral transform via Gaussian CDF
            U = norm.cdf(Z, loc=0.0)
        elif copula == "t":
            if nu is None or nu <= 0:
                raise ValueError("For copula='t', provide nu > 0.")
            # Elliptical t-copula: T = Z / sqrt(W/nu), W ~ Chi^2_nu (single radial mix)
            # Use a deterministic RNG tied to 'seed' for reproducibility.
            rng = np.random.default_rng(seed if seed is not None else None)
            W = rng.chisquare(df=nu)/nu  # scalar radial mixing variable
            T = Z / np.sqrt(W)   # pointwise Student-t nu with scale sqrt(var)
            U = student_t.cdf(T, df=nu, loc=0.0)
        else:
            raise ValueError("copula must be 'gaussian' or 't'.")

        # Numerical safety: keep strictly inside (0,1)
        U = np.clip(U, 1e-12, 1 - 1e-12)
        
        # 3) Apply inverse CDF (quantile function) of truncated normal
        # This directly gives Y ~ TruncatedNormal(mu10, sigma10^2, log10(a), log10(b))
        Y = self.truncated_dist.ppf(U)

        # # 3) Truncate mass to [Fa, Fb] in CDF space (for Y = log10 K)
        # U_trunc = self.Fa + U * (self.Fb - self.Fa)
        # U_trunc = np.clip(U_trunc, np.nextafter(self.Fa, 1), np.nextafter(self.Fb, 0))

        # # 4) Invert to Y ~ truncated N(mu10, sigma10^2) in log10-space, then exponentiate
        # Y = self.mu10 + self.sigma10 * norm.ppf(U_trunc)

        K = np.power(10.0, Y)  # enforce K in [a, b]
        return K

    # -----------------------
    # Multiple realizations
    # -----------------------
    def iter_generate_combinations_multi(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        var_list: Iterable[float],
        len_scale_list: Iterable[float],
        anis_list: Iterable[float],
        angles_list: Iterable[float],
        n_realizations: int = 1,
        seed: int = None,
    ) -> Generator[Tuple[Tuple[float, float, float, float, int], np.ndarray], None, None]:
        combos = itertools.product(var_list, len_scale_list, anis_list, angles_list)
        for var, ls, anis, angle in combos:
            for real in range(n_realizations):
                key = (var, ls, anis, angle, real)
                s = self._stable_seed(key, seed)
                field = self.generate_field(grid_x, grid_y, var, ls, anis=anis, angles=angle, seed=s)
                yield key, field

    def generate_combinations_multi(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        var_list: Iterable[float],
        len_scale_list: Iterable[float],
        anis_list: Iterable[float],
        angles_list: Iterable[float],
        n_realizations: int = 1,
        seed: int = None,
    ) -> Dict[Tuple[float, float, float, float, int], np.ndarray]:
        out: Dict[Tuple[float, float, float, float, int], np.ndarray] = {}
        for key, field in self.iter_generate_combinations_multi(
            grid_x, grid_y, var_list, len_scale_list, anis_list, angles_list, n_realizations, seed
        ):
            out[key] = field
        return out

# inside TruncatedLog10LognormalFieldGenerator
    def save_combined_pdf_per_config(
        self,
        fields,
        grid_x,
        grid_y,
        out_dir: str,
        *,
        subdir_pattern: str = "cfg_var{var:g}_ls{ls:g}_anis{anis:g}_ang{ang:g}",
        pdf_name_pattern: str = "figs_var{var:g}_ls{ls:g}_anis{anis:g}_ang{ang:g}.pdf",
        bins_linear="auto",
        bins_log="auto",
        density: bool = True,
        cmap: str = "viridis",
        dpi: int = 200,
        sort_keys: bool = True,
        title_pattern: str = "var={var:g}, ls={ls:g}, anis={anis:g}, ang={ang:g}, real={real:d}",
        one_page_per_config: bool = True,   # << NEW: put all realizations on a single page
    ) -> None:
        """
        For each (var, ls, anis, ang), write ONE PDF. If `one_page_per_config=True`,
        all realizations are shown on a single page (columns = realizations):
            Row 0: Field (linear K)       [one per realization]
            Row 1: Field (log10 K)        [one per realization]
            Row 2: Histogram (linear K)   [one per realization]
            Row 3: Histogram (log10 K)    [one per realization]
        Otherwise, it falls back to one page per realization (previous behavior).
        """
        os.makedirs(out_dir, exist_ok=True)

        # group realizations by configuration (ignore 'real' in the key)
        grouped = defaultdict(list)
        for (var, ls, anis, ang, real), arr in fields.items():
            grouped[(var, ls, anis, ang)].append((real, arr))

        cfg_keys = list(grouped.keys())
        if sort_keys:
            cfg_keys = sorted(cfg_keys, key=lambda t: (t[0], t[1], t[2], t[3]))

        extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
        vmin_lin, vmax_lin = self.a, self.b
        vmin_log, vmax_log = np.log10(self.a), np.log10(self.b)

        for (var, ls, anis, ang) in cfg_keys:
            items = grouped[(var, ls, anis, ang)]
            if sort_keys:
                items = sorted(items, key=lambda r: r[0])  # sort by realization id

            # Prepare output path
            subdir = self._config_subdir(out_dir, var, ls, anis, ang, subdir_pattern)
            pdf_path = os.path.join(subdir, pdf_name_pattern.format(var=var, ls=ls, anis=anis, ang=ang))

            with PdfPages(pdf_path) as pdf:
                if not one_page_per_config:
                    # --- Previous behavior: one page per realization ---
                    for real, field in items:
                        data_lin = field
                        data_log = np.log10(field)
                        data_flat = data_lin.ravel()
                        data_log_flat = data_log.ravel()

                        fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
                        # Field (linear)
                        im = axes[0, 0].imshow(
                            data_lin, origin="lower", extent=extent,
                            vmin=vmin_lin, vmax=vmax_lin, cmap=cmap, aspect="equal"
                        )
                        fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04).set_label("Permeability")
                        axes[0, 0].set_title("Field (linear K)")
                        axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")

                        # Field (log10)
                        im2 = axes[0, 1].imshow(
                            data_log, origin="lower", extent=extent,
                            vmin=vmin_log, vmax=vmax_log, cmap=cmap, aspect="equal"
                        )
                        fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04).set_label(r"$\log_{10}(K)$")
                        axes[0, 1].set_title(r"Field ($\log_{10}$ K)")
                        axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")

                        # Hist (linear)
                        axes[1, 0].hist(
                            data_flat, bins=bins_linear, density=density, edgecolor="black", linewidth=0.5
                        )
                        axes[1, 0].set_xlim(vmin_lin, vmax_lin)
                        axes[1, 0].axvline(self.a, color="red", ls="--", lw=0.9)
                        axes[1, 0].axvline(self.b, color="red", ls="--", lw=0.9)
                        axes[1, 0].set_title("Histogram (linear)"); axes[1, 0].set_xlabel("K")
                        axes[1, 0].set_ylabel("Density" if density else "Count")

                        # Hist (log10)
                        axes[1, 1].hist(
                            data_log_flat, bins=bins_log, density=density, edgecolor="black", linewidth=0.5
                        )
                        axes[1, 1].set_xlim(vmin_log, vmax_log)
                        axes[1, 1].axvline(vmin_log, color="red", ls="--", lw=0.9)
                        axes[1, 1].axvline(vmax_log, color="red", ls="--", lw=0.9)
                        axes[1, 1].set_title(r"Histogram ($\log_{10}$)"); axes[1, 1].set_xlabel(r"$\log_{10} K$")
                        axes[1, 1].set_ylabel("Density" if density else "Count")

                        fig.suptitle(f"var={var:g}, ls={ls:g}, anis={anis:g}, ang={ang:g}, real={real}", y=0.995, fontsize=11)
                        fig.tight_layout(rect=[0, 0, 1, 0.97])
                        pdf.savefig(fig, dpi=dpi)
                        plt.close(fig)

                else:
                    # --- NEW: single page per configuration, multiple realizations as columns ---
                    n = len(items)

                    # build consistent histogram bins across all realizations (so columns comparable)
                    all_lin = np.concatenate([arr.ravel() for _, arr in items])
                    all_log = np.log10(all_lin)
                    if bins_linear == "auto":
                        bins_linear_edges = np.histogram_bin_edges(all_lin, bins="auto", range=(vmin_lin, vmax_lin))
                    else:
                        bins_linear_edges = bins_linear
                    if bins_log == "auto":
                        bins_log_edges = np.histogram_bin_edges(all_log, bins="auto", range=(vmin_log, vmax_log))
                    else:
                        bins_log_edges = bins_log

                    # figure size: ~3.0 in per column, 4 rows
                    fig_w = max(3.0 * n, 6.0)
                    fig_h = 10.0
                    fig, axes = plt.subplots(
                        4, n, figsize=(fig_w, fig_h),
                        gridspec_kw=dict(hspace=0.35, wspace=0.25, height_ratios=[1.2, 1.2, 0.9, 0.9])
                    )

                    # if only one realization, axes dims come as 1D; make them 2D for uniform code
                    if n == 1:
                        axes = np.array([axes]).reshape(4, 1)

                    # plot each realization as a column
                    for col, (real, field) in enumerate(items):
                        data_lin = field
                        data_log = np.log10(field)
                        data_flat = data_lin.ravel()
                        data_log_flat = data_log.ravel()

                        # Column header
                        axes[0, col].set_title(f"real={real}", fontsize=10, pad=6)

                        # Row 0: Field (linear)
                        im_lin = axes[0, col].imshow(
                            data_lin, origin="lower", extent=extent,
                            vmin=vmin_lin, vmax=vmax_lin, cmap=cmap, aspect="equal"
                        )
                        axes[0, col].set_xlabel("x"); axes[0, col].set_ylabel("y")

                        # Row 1: Field (log10)
                        im_log = axes[1, col].imshow(
                            data_log, origin="lower", extent=extent,
                            vmin=vmin_log, vmax=vmax_log, cmap=cmap, aspect="equal"
                        )
                        axes[1, col].set_xlabel("x"); axes[1, col].set_ylabel("y")

                        # Row 2: Histogram (linear)
                        axes[2, col].hist(
                            data_flat, bins=bins_linear_edges, density=density,
                            edgecolor="black", linewidth=0.4
                        )
                        axes[2, col].set_xlim(vmin_lin, vmax_lin)
                        axes[2, col].axvline(self.a, color="red", ls="--", lw=0.8)
                        axes[2, col].axvline(self.b, color="red", ls="--", lw=0.8)
                        axes[2, col].set_xlabel("K"); axes[2, col].set_ylabel("Density" if density else "Count")
                        axes[2, col].set_title("Hist (linear)", fontsize=9)

                        # Row 3: Histogram (log10)
                        axes[3, col].hist(
                            data_log_flat, bins=bins_log_edges, density=density,
                            edgecolor="black", linewidth=0.4
                        )
                        axes[3, col].set_xlim(vmin_log, vmax_log)
                        axes[3, col].axvline(vmin_log, color="red", ls="--", lw=0.8)
                        axes[3, col].axvline(vmax_log, color="red", ls="--", lw=0.8)
                        axes[3, col].set_xlabel(r"$\log_{10} K$")
                        axes[3, col].set_ylabel("Density" if density else "Count")
                        axes[3, col].set_title(r"Hist ($\log_{10}$)", fontsize=9)

                    # Shared colorbars for the two field rows (saves space)
                    # Use the last plotted image handles for each row to define colorbars
                    cbar0 = fig.colorbar(im_lin, ax=list(axes[0, :]), fraction=0.02, pad=0.01)
                    cbar0.set_label("Permeability")
                    cbar1 = fig.colorbar(im_log, ax=list(axes[1, :]), fraction=0.02, pad=0.01)
                    cbar1.set_label(r"$\log_{10}(K)$")
                    fig.suptitle(title_pattern.format(ls=ls, anis=anis, ang=ang, real=n), y=0.995, fontsize=12)
                    fig.tight_layout(rect=[0, 0, 1, 0.97])

                    pdf.savefig(fig, dpi=dpi)
                    plt.close(fig)

   
    # -----------------------
    # Plotting (single field)
    # -----------------------
    def plot_and_save(
        self,
        field: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        filename: str,
        *,
        log10: bool = False,
        cmap: str = "viridis",
        dpi: int = 200,
        vmin: float = None,
        vmax: float = None,
        add_contours: bool = False,
        n_contours: int = 10,
        title: str = None,
        tight_layout: bool = True,
        show: bool = False,
    ) -> None:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        if log10:
            data = np.log10(field)
            if vmin is None:
                vmin = np.log10(self.a)
            if vmax is None:
                vmax = np.log10(self.b)
            cbar_label = r"$\log_{10}(\mathrm{Permeability})$"
        else:
            data = field
            if vmin is None:
                vmin = self.a
            if vmax is None:
                vmax = self.b
            cbar_label = "Permeability"

        fig, ax = plt.subplots(figsize=(6, 5))
        extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
        im = ax.imshow(data, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if title:
            ax.set_title(title)

        if add_contours:
            try:
                levels = np.linspace(vmin, vmax, n_contours)
                cs = ax.contour(
                    np.linspace(extent[0], extent[1], data.shape[1]),
                    np.linspace(extent[2], extent[3], data.shape[0]),
                    data,
                    levels=levels,
                    linewidths=0.6,
                )
                ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
            except Exception:
                pass

        if tight_layout:
            fig.tight_layout()
        fig.savefig(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)

    # -----------------------
    # Batch plotting (multi)
    # -----------------------
    def save_all_plots_multi(
        self,
        fields,
        grid_x,
        grid_y,
        out_dir: str,
        *,
        log10: bool = False,
        fmt: str = "png",
        cmap: str = "viridis",
        dpi: int = 200,
        file_stem: str = "plot",  # stem used inside each config folder
        subdir_pattern: str = "cfg_var{var:g}_ls{ls:g}_anis{anis:g}_ang{ang:g}",
        add_contours: bool = False,
        n_contours: int = 10,
        title_pattern: str = "K (var={var:g}, ls={ls:g}, anis={anis:g}, ang={ang:g})",
        sort_keys: bool = True,
    ) -> None:
        """
        Save field plots, grouped by configuration:
            out_dir/<cfg_subdir>/<file_stem>_realXXX.<fmt>
        """
        os.makedirs(out_dir, exist_ok=True)
        keys = list(fields.keys())
        if sort_keys:
            keys = sorted(keys, key=lambda t: (t[0], t[1], t[2], t[3], t[4]))

        for (var, ls, anis, ang, real) in keys:
            subdir = self._config_subdir(out_dir, var, ls, anis, ang, subdir_pattern)
            fname = f"{file_stem}_real{real:03d}.{fmt}"
            fpath = os.path.join(subdir, fname)

            title = title_pattern.format(var=var, ls=ls, anis=anis, ang=ang)
            field = fields[(var, ls, anis, ang, real)]
            self.plot_and_save(
                field,
                grid_x,
                grid_y,
                fpath,
                log10=log10,
                cmap=cmap,
                dpi=dpi,
                add_contours=add_contours,
                n_contours=n_contours,
                title=title,
            )

    # -----------------------
    # Histograms (single + batch)
    # -----------------------
    def plot_histograms(
        self,
        field: np.ndarray,
        filename: str,
        *,
        bins_linear="auto",
        bins_log="auto",
        density: bool = True,
        dpi: int = 200,
        title_linear: str = "Histogram (linear scale)",
        title_log: str = r"Histogram ($\log_{10}$ scale)",
        show: bool = False,
    ) -> None:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        data = field.ravel()
        data_log10 = np.log10(data)
        a, b = self.a, self.b
        la, lb = np.log10(a), np.log10(b)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        ax.hist(data, bins=bins_linear, density=density, edgecolor="black", linewidth=0.5)
        ax.set_xlim(a, b)
        ax.axvline(a, color="red", linestyle="--", linewidth=0.9)
        ax.axvline(b, color="red", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Permeability")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(title_linear)

        ax = axes[1]
        ax.hist(data_log10, bins=bins_log, density=density, edgecolor="black", linewidth=0.5)
        ax.set_xlim(la, lb)
        ax.axvline(la, color="red", linestyle="--", linewidth=0.9)
        ax.axvline(lb, color="red", linestyle="--", linewidth=0.9)
        ax.set_xlabel(r"$\log_{10}(\mathrm{Permeability})$")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(title_log)

        fig.tight_layout()
        fig.savefig(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)

    def save_all_histograms_multi(
        self,
        fields,
        out_dir: str,
        *,
        bins_linear="auto",
        bins_log="auto",
        density: bool = True,
        fmt: str = "png",
        dpi: int = 200,
        file_stem: str = "hist",  # stem used inside each config folder
        subdir_pattern: str = "cfg_var{var:g}_ls{ls:g}_anis{anis:g}_ang{ang:g}",
        title_linear_pattern: str = "Histogram (linear) var={var:g}, ls={ls:g}, anis={anis:g}, ang={ang:g}, real={real:d}",
        title_log_pattern: str = "Histogram (log10) var={var:g}, ls={ls:g}, anis={anis:g}, ang={ang:g}, real={real:d}",
        sort_keys: bool = True,
        show: bool = False,
    ) -> None:
        """
        Save histogram figures, grouped by configuration:
            out_dir/<cfg_subdir>/<file_stem>_realXXX.<fmt>
        """
        os.makedirs(out_dir, exist_ok=True)
        keys = list(fields.keys())
        if sort_keys:
            keys = sorted(keys, key=lambda t: (t[0], t[1], t[2], t[3], t[4]))

        for (var, ls, anis, ang, real) in keys:
            subdir = self._config_subdir(out_dir, var, ls, anis, ang, subdir_pattern)
            fname = f"{file_stem}_real{real:03d}.{fmt}"
            fpath = os.path.join(subdir, fname)

            field = fields[(var, ls, anis, ang, real)]
            self.plot_histograms(
                field,
                fpath,
                bins_linear=bins_linear,
                bins_log=bins_log,
                density=density,
                dpi=dpi,
                title_linear=title_linear_pattern.format(var=var, ls=ls, anis=anis, ang=ang, real=real),
                title_log=title_log_pattern.format(var=var, ls=ls, anis=anis, ang=ang, real=real),
                show=show,
            )

    # -------------------------------------------------
    # Save plots and histograms together per config
    # -------------------------------------------------
    def save_all_figures_grouped(
        self,
        fields,
        grid_x,
        grid_y,
        out_dir: str,
        *,
        plot_log10: bool = False,
        plot_fmt: str = "png",
        hist_fmt: str = "png",
        cmap: str = "viridis",
        dpi: int = 200,
        subdir_pattern: str = "cfg_var{var:g}_ls{ls:g}_anis{anis:g}_ang{ang:g}",
        plot_stem: str = "plot",
        hist_stem: str = "hist",
        add_contours: bool = False,
        n_contours: int = 10,
        bins_linear="auto",
        bins_log="auto",
        density: bool = True,
        sort_keys: bool = True,
    ) -> None:
        """
        Convenience wrapper: in a single pass, for each (config, realization),
        save the field plot and the histogram side-by-side into the same folder:
            out_dir/<cfg_subdir>/<plot_stem>_realXXX.<plot_fmt>
            out_dir/<cfg_subdir>/<hist_stem>_realXXX.<hist_fmt>
        """
        os.makedirs(out_dir, exist_ok=True)
        keys = list(fields.keys())
        if sort_keys:
            keys = sorted(keys, key=lambda t: (t[0], t[1], t[2], t[3], t[4]))

        for (var, ls, anis, ang, real) in keys:
            subdir = self._config_subdir(out_dir, var, ls, anis, ang, subdir_pattern)

            # --- plot ---
            plot_name = f"{plot_stem}_real{real:03d}.{plot_fmt}"
            plot_path = os.path.join(subdir, plot_name)
            field = fields[(var, ls, anis, ang, real)]
            title = f"K (var={var:g}, ls={ls:g}, anis={anis:g}, ang={ang:g}, real={real:d})"
            self.plot_and_save(
                field,
                grid_x,
                grid_y,
                plot_path,
                log10=plot_log10,
                cmap=cmap,
                dpi=dpi,
                add_contours=add_contours,
                n_contours=n_contours,
                title=title,
            )

            # --- histogram ---
            hist_name = f"{hist_stem}_real{real:03d}.{hist_fmt}"
            hist_path = os.path.join(subdir, hist_name)
            self.plot_histograms(
                field,
                hist_path,
                bins_linear=bins_linear,
                bins_log=bins_log,
                density=density,
                dpi=dpi,
                title_linear=f"Histogram (linear) var={var:g}, ls={ls:g}, anis={anis:g}, ang={ang:g}, real={real:d}",
                title_log=rf"Histogram ($\log_{{10}}$) var={var:g}, ls={ls:g}, anis={anis:g}, ang={ang:g}, real={real:d}",
            )

    # -----------------------
    # HDF5 saving (flattened only, per realization)
    # -----------------------
    def save_fields_h5(
        self,
        fields: Dict[Tuple[float, float, float, float, int], np.ndarray],
        out_dir: str,
        *,
        dataset_name: str = "Permeability",
        filename_pattern: str = "K_var{var:g}_ls{ls:g}_anis{anis:g}_ang{ang:g}_real{real:03d}.h5",
        sort_keys: bool = True,
        overwrite: bool = True
    ) -> None:
        """
        Save each field realization into its own HDF5 file containing exactly:
        - dataset 'Permeability' (float64), flattened row-major
        - dataset 'Cell Ids' (int32), 0-based indices aligned with 'Permeability'
        No compression or chunking is used to mirror 'permeability_field(3).h5'.
        """
        import os, h5py
        os.makedirs(out_dir, exist_ok=True)

        keys = list(fields.keys())
        if sort_keys:
            keys = sorted(keys, key=lambda t: (t[0], t[1], t[2], t[3], t[4]))

        for (var, ls, anis, ang, real) in keys:
            field = np.asarray(fields[(var, ls, anis, ang, real)])

            # Force exact dtypes to match the reference file
            flat_perm = field.ravel(order="F").astype(np.float64, copy=False)  # Permeability as float64
            cell_ids  = (np.arange(flat_perm.size, dtype=np.int32) + 1)              # Cell Ids as int32 (0-based)

            fname = filename_pattern.format(var=var, ls=ls, anis=anis, ang=ang, real=real)
            fpath = os.path.join(out_dir, fname)

            if not overwrite and os.path.exists(fpath):
                raise FileExistsError(f"File exists and overwrite=False: {fpath}")

            with h5py.File(fpath, "w") as h5:
                # No compression/chunking to replicate the reference structure exactly
                h5.create_dataset(dataset_name, data=flat_perm)   # float64
                h5.create_dataset("Cell Ids",    data=cell_ids)   # int32



def main():
    x = y = np.linspace(0, 320, 320)
    scaling = 1e-3/(1000*9.81)
    bounds = (1e-4*scaling, 1e-2*scaling)
    mu10 = 0.5 * (np.log10(bounds[0]) + np.log10(bounds[1]))

    var_list_marginal = [0.1] #[0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0]

    for var_m in var_list_marginal: 
        sigma10 = np.sqrt(var_m)
        gen = TruncatedLog10LognormalFieldGenerator(bounds, mu10=mu10, sigma10=sigma10, alpha=0.05)  # or give mu10, sigma10 directly

        var_s = [1.0]
        len_scale_list = [50]
        anis_list      = [1.0]
        angles_list    = [0.0]
        n_realizations = 1
        seed_base      = 123

        fields = gen.generate_combinations_multi(
            x, y, var_s, len_scale_list, anis_list, angles_list, n_realizations=n_realizations, seed=seed_base
        )

        # gen.save_combined_pdf_per_config(
        #     fields, x, y, out_dir=f"out_pdf_50_{var_m:g}",
        #     bins_linear="auto", bins_log="auto", density=True, dpi=200,
        #     title_pattern=f"var={var_m:g}" + ", ls={ls:g}, anis={anis:g}, ang={ang:g}, real={real:d}",
        #     pdf_name_pattern="figs_var{var:g}_ls{ls:g}_anis{anis:g}_ang{ang:g}.pdf",
        # )
        # gen.save_all_plots_multi(fields, x, y, out_dir=f"out_grouped_{var_m:g}", file_stem="plot")
        # gen.save_all_plots_multi(fields, x, y, out_dir=f"out_grouped_{var_m:g}", file_stem="plot_log", log10=True)
        # gen.save_all_histograms_multi(fields, out_dir=f"out_grouped_{var_m:g}", file_stem="hist", bins_linear="auto", bins_log="auto")

        gen.save_fields_h5(fields, out_dir=f"out_h5_{var_m:g}")


if __name__ == "__main__":
    main()