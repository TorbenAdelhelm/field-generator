# main.py  — REPLACE CONTENTS
import numpy as np
import matplotlib.pyplot as plt
import gstools as gs

from generators import (
    TruncatedLog10LognormalFieldGenerator,
    ClippedLognormalFieldGenerator,
)
from vario import ensemble_to_fields_for_vario
from plotting import (
    save_ensemble_diagnostics_single_page_pdf,
    plot_ensemble_variogram_summary,
    plot_directional_variogram_summary,
    save_variogram_data,
)

# ---------------------------------------------------------------------
# Grid (structured only)
# ---------------------------------------------------------------------
x = y = np.arange(640)  # unit spacing

# Kernel (anisotropic to demonstrate rotated_main_axes)
L_major = 30           # major-axis range
anis = 1.0               # minor/major ratio
theta = 0 #np.pi / 6        # rotation [rad]

# Marginal (bounds, moments)
scaling = 1e-3/(1000*9.81)
bounds = (1e-4*scaling, 5e-2*scaling)
mu10 = 0.5 * (np.log10(bounds[0]) + np.log10(bounds[1]))
sigma10 = 0.4

n_real = 10

# Output folder
import os; os.makedirs("out", exist_ok=True)

# =====================================================================
# A) Copula generator (Option-C) — variograms on LATENT Z
# =====================================================================
cop = TruncatedLog10LognormalFieldGenerator(
    bounds, mu10, sigma10,
    match_moments=True, target_mean10=mu10, target_var10=sigma10**2
)

Kc_ens, cop_seeds = cop.generate_ensemble(
    x, y, n_realizations=n_real,
    len_scale=[L_major, anis * L_major], angles=theta,
    var_kernel=1.0, copula="gaussian",
    master_seed=42
)
from saving import save_ensemble_h5
out_paths = save_ensemble_h5(
        Kc_ens,
        seeds=cop_seeds,
        out_dir="/scratch/adelhetn/data/vampireman/permeability-input-fields",
        filename_pattern="permeability_seed{seed}_{i:04}.h5",
        dataset_name="Permeability"
)
# Optional: diagnostics page
# save_ensemble_diagnostics_single_page_pdf(
#     Kc_ens, generator=cop, pdf_path="out/copula_ensemble_diag.pdf",
#     suptitle="Copula fields: maps + histograms", seeds=cop_seeds
# )

# ---- Build 'fields' for GSTools: LATENT domain ----
#fields_latent = ensemble_to_fields_for_vario(Kc_ens, mode="log10", generator=cop)

# ---- Isotropic joined variogram (CONVENIENT API) ----
#bins = np.arange(0, 60, 2)
# bc_iso, gamma_iso = gs.vario_estimate((x, y), fields_latent, bins, mesh_type="structured")

# Quick iso fit (median overlay plot)
# fit_iso = gs.Exponential(dim=2)
# fit_iso.fit_variogram(bc_iso, gamma_iso)
# plot_ensemble_variogram_summary(
#     bc_iso, gamma_iso,  # passing empirical only is enough for the plotter
#     # Supply a tiny DataFrame-like dict for median params; or just ignore model curve
#     df_params=__import__("pandas").DataFrame([{"var": fit_iso.var, "len_scale": fit_iso.len_scale}]),
#     model_cls=gs.Exponential,
#     title="Isotropic joined variogram (latent, copula)",
#     out_plot_path="out/copula_latent_iso_vario.png",
#     out_data_path="out/copula_latent_iso_vario.csv"
# )

# print(fit_iso)

# ---- Directional (two main axes, CONVENIENT API) ----
# dirs = gs.rotated_main_axes(dim=2, angles=theta)  # returns two unit vectors
# bc_dir, dir_gamma, counts = gs.vario_estimate(
#     (x, y), fields_latent, bins,
#     direction=dirs, angles_tol=np.pi/16, bandwidth=8,
#     mesh_type="structured", return_counts=True,
# )

# Fit anisotropic model to the stacked directional curve
# fit_aniso = gs.Exponential(dim=2)
# fit_aniso.fit_variogram(bc_dir, dir_gamma)

# print(fit_aniso)

# Plot & save
# plot_directional_variogram_summary(
#     bc_dir, dir_gamma, fit_aniso,
#     title="Directional variogram (latent, copula)",
#     out_plot_path="out/copula_latent_dirvario.png",
#     out_data_path="out/copula_latent_dirvario.csv",
# )

# =====================================================================
# B) Clipped generator — variograms on log10(K)
# =====================================================================
# clip = ClippedLognormalFieldGenerator(bounds, mu10, sigma10)

# K_ens_clip, seeds_clipped = clip.generate_ensemble(
#     x, y, n_realizations=n_real,
#     len_scale=[L_major, anis * L_major], angles=theta,
#     master_seed=42
# )

# from saving import save_ensemble_h5
# out_paths = save_ensemble_h5(
#         K_ens_clip,
#         seeds=seeds_clipped,
#         out_dir="/scratch/adelhetn/data/vampireman/permeability-input-fields",
#         filename_pattern="permeability_seed{seed}_{i:04}.h5",
#         dataset_name="Permeability"
# )

# save_ensemble_diagnostics_single_page_pdf(
#     K_ens_clip, generator=clip, pdf_path="out/clipped_ensemble_diag.pdf",
#     overlay="trunc",
#     suptitle="Copula fields: maps + histograms", seeds=cop_seeds
# )

# ---- Build 'fields' for GSTools: LOG10 domain ----
#fields_log = ensemble_to_fields_for_vario(K_ens_clip, mode="log10")

# ---- Isotropic joined variogram (CONVENIENT API) ----
#bc_iso_c, gamma_iso_c = gs.vario_estimate((x, y), fields_log, bins, mesh_type="structured")

# fit_iso_c = gs.Exponential(dim=2)
# fit_iso_c.fit_variogram(bc_iso_c, gamma_iso_c)
# plot_ensemble_variogram_summary(
#     bc_iso_c, gamma_iso_c,
#     df_params=__import__("pandas").DataFrame([{"var": fit_iso_c.var, "len_scale": fit_iso_c.len_scale}]),
#     model_cls=gs.Exponential,
#     title="Isotropic joined variogram (log10, clipped)",
#     out_plot_path="out/clipped_log_iso_vario.png",
#     out_data_path="out/clipped_log_iso_vario.csv"
# )

# print(fit_iso_c)

# ---- Directional (two main axes, CONVENIENT API) ----
# bc_dir_c, dir_gamma_c, counts_c = gs.vario_estimate(
#     (x, y), fields_log, bins,
#     direction=dirs, angles_tol=np.pi/16, bandwidth=8,
#     mesh_type="structured", return_counts=True,
# )

# fit_aniso_c = gs.Exponential(dim=2)
# fit_aniso_c.fit_variogram(bc_dir_c, dir_gamma_c)

# print(fit_aniso_c)

# plot_directional_variogram_summary(
#     bc_dir_c, dir_gamma_c, fit_aniso_c,
#     title="Directional variogram (log10, clipped)",
#     out_plot_path="out/clipped_log_dirvario.png",
#     out_data_path="out/clipped_log_dirvario.csv",
# )

print("Done. Plots & CSVs written to ./out/")
