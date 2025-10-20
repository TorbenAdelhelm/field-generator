# plotting.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import gstools as gs
import os

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt

def save_ensemble_diagnostics_single_page_pdf(
    K_ens: np.ndarray,
    generator,
    pdf_path: str,
    *,
    log_scale: bool = True,
    bins: int = 50,
    cmap: str = "viridis",
    titles=None,
    seeds=None,
    per_block_size=(3.6, 2.8),
    dpi: int = 300,
    suptitle: str | None = None,
    overlay: str = "auto",   # NEW: 'auto' | 'normal' | 'trunc' | 'none'
) -> str:
    """
    Works with BOTH generators. If `overlay='auto'`:
      - uses generator.log10_pdf(x) when available (copula -> truncated normal),
      - otherwise falls back to normal(mu10, sigma10) if attributes exist,
      - or draws no overlay if neither is available.

    For the clipped generator, the default overlay is the *untruncated normal* in log10.
    """
    if K_ens.ndim != 3:
        raise ValueError("K_ens must have shape (n_realizations, Ny, Nx).")
    n = K_ens.shape[0]
    n_cols = 3 if log_scale else 2
    figsize = (per_block_size[0] * n_cols, per_block_size[1] * n)

    # Color limits and log-axis limits from generator bounds
    a, b = float(generator.a), float(generator.b)
    loga, logb = np.log10(a), np.log10(b)

    # Decide the log10 overlay PDF to use
    x_log = np.linspace(loga, logb, 200)
    pdf_log = None
    if overlay != "none":
        # Prefer a generator-provided pdf
        if hasattr(generator, "log10_pdf") and callable(getattr(generator, "log10_pdf")):
            # overlay='auto' -> let generator decide; otherwise pass explicit kind
            kind = None if overlay == "auto" else overlay
            try:
                pdf_log = generator.log10_pdf(x_log, kind=overlay if kind else "auto")
            except TypeError:
                # Backward compatibility: some implementations may not accept 'kind'
                pdf_log = generator.log10_pdf(x_log)
        elif overlay in ("normal", "auto") and hasattr(generator, "mu10") and hasattr(generator, "sigma10"):
            from scipy.stats import norm as _norm
            pdf_log = _norm(loc=generator.mu10, scale=generator.sigma10).pdf(x_log)
        elif overlay == "trunc" and hasattr(generator, "mu10") and hasattr(generator, "sigma10"):
            from scipy.stats import truncnorm as _truncnorm
            a_std = (loga - generator.mu10) / generator.sigma10
            b_std = (logb - generator.mu10) / generator.sigma10
            tn = _truncnorm(a=a_std, b=b_std, loc=generator.mu10, scale=generator.sigma10)
            pdf_log = tn.pdf(x_log)
        # else: leave pdf_log=None -> no overlay

    fig, axes = plt.subplots(n, n_cols, figsize=figsize, squeeze=False)
    im_last = None

    for i in range(n):
        K = K_ens[i]
        vals = K.ravel()
        logvals = np.log10(vals, where=(vals > 0))

        # (1) Spatial map
        ax0 = axes[i, 0]
        im_last = ax0.imshow(
            K, origin="lower", cmap=cmap, vmin=a, vmax=b, interpolation="nearest"
        )
        ax0.set_title(
            titles[i] if titles
            else (f"Realization {i} (seed={seeds[i]})" if seeds is not None else f"Realization {i}")
        )
        ax0.set_xticks([]); ax0.set_yticks([])

        # (2) Histogram of K (linear)
        ax1 = axes[i, 1]
        ax1.hist(vals, bins=bins, density=True, color="steelblue", alpha=0.8)
        ax1.axvline(a, color="r", ls="--", lw=1)
        ax1.axvline(b, color="r", ls="--", lw=1)
        if i == 0:
            ax1.set_title("Histogram of K (linear)")
        if i == n - 1:
            ax1.set_xlabel("K")

        # (3) Histogram of log10(K) + optional overlay
        if log_scale:
            ax2 = axes[i, 2]
            ax2.hist(logvals, bins=bins, density=True, color="darkorange", alpha=0.8)
            if pdf_log is not None:
                ax2.plot(x_log, pdf_log, "k--", lw=1.2, label="log10 marginal (overlay)")
                if i == 0:
                    ax2.legend(frameon=False)
            if i == 0:
                ax2.set_title("Histogram of log10(K)")
            if i == n - 1:
                ax2.set_xlabel("log10(K)")

    # Shared colorbar
    if im_last is not None:
        cax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
        cbar = fig.colorbar(im_last, cax=cax)
        cbar.set_label("Permeability K")

    if suptitle:
        fig.suptitle(suptitle, y=0.997)

    fig.subplots_adjust(left=0.05, right=0.90, top=0.97, bottom=0.05, wspace=0.25, hspace=0.35)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf_path

def save_variogram_data(
    bin_center: np.ndarray,
    gamma_emp: np.ndarray,
    *,
    out_path: str,
    gamma_model_h: np.ndarray | None = None,
    gamma_model: np.ndarray | None = None,
    metadata: dict | None = None,
):
    """
    Save variogram arrays to disk. Format inferred from file extension:
      - '.csv' -> CSV with columns (h, gamma_emp, gamma_model) if provided
      - '.npz' -> NumPy npz with arrays + metadata dict
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if out_path.lower().endswith(".csv"):
        df = pd.DataFrame({"h": bin_center, "gamma_emp": gamma_emp})
        if gamma_model_h is not None and gamma_model is not None:
            # align model to empirical h for convenience
            gm_interp = np.interp(bin_center, gamma_model_h, gamma_model)
            df["gamma_model"] = gm_interp
        df.to_csv(out_path, index=False)
    elif out_path.lower().endswith(".npz"):
        np.savez(
            out_path,
            h=bin_center,
            gamma_emp=gamma_emp,
            h_model=(gamma_model_h if gamma_model_h is not None else np.array([])),
            gamma_model=(gamma_model if gamma_model is not None else np.array([])),
            metadata=(metadata if metadata is not None else {}),
        )
    else:
        raise ValueError("Unsupported variogram data format. Use .csv or .npz")
    return out_path


def plot_ensemble_variogram_summary(
    bin_center: np.ndarray,
    gamma_mean: np.ndarray,
    df_params: pd.DataFrame,
    *,
    model_cls=gs.Exponential,
    title: str = "Ensemble variogram",
    # NEW: optional outputs
    out_plot_path: str | None = None,
    out_data_path: str | None = None,
    dpi: int = 300,
):
    """
    Plot ensemble-mean empirical variogram with a model curve (median params),
    and optionally SAVE:
      - the figure (PNG/PDF) via `out_plot_path`
      - the variogram arrays (CSV/NPZ) via `out_data_path`
    """
    # median params → model curve
    var_med   = float(np.nanmedian(df_params["var"])) if "var" in df_params else 1.0
    len_med   = float(np.nanmedian(df_params["len_scale"])) if "len_scale" in df_params else 1.0
    anis_med  = float(np.nanmedian(df_params["anis"])) if "anis" in df_params else 1.0
    angles_md = float(np.nanmedian(df_params["angles"])) if "angles" in df_params else 0.0

    model = model_cls(dim=2, var=var_med, len_scale=len_med, anis=anis_med, angles=angles_md)
    h_dense = np.linspace(0.0, float(np.nanmax(bin_center)), 400)
    gamma_model = model.variogram(h_dense)

    # ---- plot
    plt.figure(figsize=(6.6, 4.2))
    plt.plot(bin_center, gamma_mean, "o", label="Empirical (ensemble mean)")
    plt.plot(h_dense, gamma_model, "-", label=f"{model_cls.__name__} (median params)")
    plt.xlabel("Lag h"); plt.ylabel("Semivariance γ(h)")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()

    # save figure if requested
    if out_plot_path is not None:
        os.makedirs(os.path.dirname(out_plot_path) or ".", exist_ok=True)
        plt.savefig(out_plot_path, dpi=dpi, bbox_inches="tight")

    # save data if requested
    if out_data_path is not None:
        meta = {
            "model": model_cls.__name__,
            "var_med": var_med,
            "len_scale_med": len_med,
            "anis_med": anis_med,
            "angles_med": angles_md,
            "title": title,
        }
        save_variogram_data(
            bin_center,
            gamma_mean,
            out_path=out_data_path,
            gamma_model_h=h_dense,
            gamma_model=gamma_model,
            metadata=meta,
        )

    plt.close()

def plot_directional_variogram_summary(
    bin_center: np.ndarray,
    dir_gamma: np.ndarray,     # shape (2, n_bins)
    fitted_model,
    *,
    title: str = "Directional variogram (two main axes)",
    out_plot_path: str | None = None,
    out_data_path: str | None = None,
    dpi: int = 300,
):
    """
    Plot the two stacked directional curves (major/minor) and the fitted, anisotropic model.
    """
    h_dense = np.linspace(0.0, float(np.nanmax(bin_center)), 400)
    gamma_model = fitted_model.variogram(h_dense)

    plt.figure(figsize=(6.6, 4.2))
    plt.plot(bin_center, dir_gamma[0], "o", label="dir 0 (major-axis band)")
    plt.plot(bin_center, dir_gamma[1], "s", label="dir 1 (minor-axis band)")
    plt.plot(h_dense, gamma_model, "-", label="Fitted anisotropic model")
    plt.xlabel("Lag h"); plt.ylabel("Semivariance γ(h)")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()

    if out_plot_path is not None:
        os.makedirs(os.path.dirname(out_plot_path) or ".", exist_ok=True)
        plt.savefig(out_plot_path, dpi=dpi, bbox_inches="tight")

    if out_data_path is not None:
        # Save the mean of the two curves for a single-line CSV, but also drop an NPZ with both
        from plotting import save_variogram_data
        save_variogram_data(
            bin_center, dir_gamma.mean(axis=0),
            out_path=out_data_path if out_data_path.lower().endswith(".csv") else out_data_path + ".csv",
            gamma_model_h=h_dense, gamma_model=gamma_model,
            metadata={"note": "mean over two rotated main axes"}
        )
        # Full two-direction arrays in NPZ (same stem)
        np.savez(
            (out_data_path[:-4] if out_data_path.lower().endswith(".csv") else out_data_path) + ".npz",
            h=bin_center, dir0=dir_gamma[0], dir1=dir_gamma[1],
            h_model=h_dense, gamma_model=gamma_model,
            model_params={
                "len_scale": getattr(fitted_model, "len_scale", None),
                "anis": getattr(fitted_model, "anis", None),
                "angles": getattr(fitted_model, "angles", None),
                "var": getattr(fitted_model, "var", None),
            }
        )
    plt.close()
