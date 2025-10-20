# generators.py
import numpy as np
import gstools as gs
from typing import Tuple, List, Literal, Optional
from scipy.stats import truncnorm, norm, t as student_t
from gstools.random import MasterRNG

# ---- utilities (marginal transforms) ---------------------------------
def to_latent_scores_from_K(K: np.ndarray, truncated_dist: truncnorm) -> np.ndarray:
    Y = np.log10(K, where=(K > 0))
    U = truncated_dist.cdf(Y).clip(1e-12, 1-1e-12)
    return norm.ppf(U)

def from_latent_scores_to_K(Z: np.ndarray, truncated_dist: truncnorm) -> np.ndarray:
    U = norm.cdf(Z).clip(1e-12, 1-1e-12)
    Y = truncated_dist.ppf(U)
    return 10**Y

# ---- Gaussian-copula, truncated-log10 generator ----------------------
class TruncatedLog10LognormalFieldGenerator:
    def __init__(self, bounds: Tuple[float,float], mu10: float, sigma10: float,
                 *, match_moments: bool=False, target_mean10: float=None, target_var10: float=None):
        a, b = bounds
        loga, logb = np.log10(a), np.log10(b)
        if match_moments:
            if target_mean10 is None or target_var10 is None:
                raise ValueError("Provide target_mean10/var10 with match_moments=True.")
            # simple 2-eq solve (reuse your version)
            from scipy.optimize import root
            def trunc_moments(mu,sigma):
                a_std, b_std = (loga-mu)/sigma, (logb-mu)/sigma
                Z = norm.cdf(b_std)-norm.cdf(a_std)
                phi_a,phi_b = norm.pdf(a_std), norm.pdf(b_std)
                mean = mu + sigma*(phi_a-phi_b)/Z
                var  = sigma**2*(1+(a_std*phi_a-b_std*phi_b)/Z - ((phi_a-phi_b)/Z)**2)
                return mean, var
            def fun(p):
                m,v = trunc_moments(p[0], p[1])
                return [m-target_mean10, v-target_var10]
            sol = root(fun, x0=[target_mean10, np.sqrt(target_var10)])
            if not sol.success:
                raise RuntimeError("Moment match failed.")
            mu10, sigma10 = float(sol.x[0]), float(sol.x[1])

        self.a, self.b = a, b
        self.mu10, self.sigma10 = float(mu10), float(sigma10)
        a_std, b_std = (loga-self.mu10)/self.sigma10, (logb-self.mu10)/self.sigma10
        self.truncated_dist = truncnorm(a=a_std, b=b_std, loc=self.mu10, scale=self.sigma10)
    
    def log10_pdf(self, x: np.ndarray, *, kind: str = "auto") -> np.ndarray:
        """
        PDF of log10(K) for plotting. For this generator, it's the truncated normal
        we actually use. 'kind' is accepted for API symmetry and ignored.
        """
        mean = self.truncated_dist.mean()
        var = self.truncated_dist.var()
        return self.truncated_dist.pdf(x)

    def generate_ensemble(self, x: np.ndarray, y: np.ndarray, n_realizations: int, *,
                          len_scale: float, angles: float=0.0,
                          var_kernel: float=1.0, copula: Literal["gaussian","t"]="gaussian",
                          nu: Optional[float]=5.0, standardize_latent: bool=True,
                          master_seed: int=20170519, return_latent: bool=False):
        model = gs.Exponential(dim=2, var=var_kernel, len_scale=len_scale, angles=angles)
        #srf = gs.SRF(model, mean=0.0); srf.set_pos([x,y], "structured")
        master = MasterRNG(master_seed)

        Ks, Zs, seeds = [], [], []
        for i in range(int(n_realizations)):
            seed = master(); seeds.append(seed)
            srf = gs.SRF(model, mean=0.0, seed=seed)
            Z = srf((x, y), mesh_type="structured")
            if standardize_latent:
                Z = (Z - Z.mean()) / Z.std()

            if copula == "gaussian":
                K = from_latent_scores_to_K(Z, self.truncated_dist)
            elif copula == "t":
                if nu is None or nu <= 0: raise ValueError("nu>0 required for t-copula")
                # radial factor per-realization (t-copula): keep dependence
                rng = np.random.default_rng(seed)
                W = rng.chisquare(df=nu) / nu
                T = Z/np.sqrt(W)
                U = student_t.cdf(T, df=nu).clip(1e-12, 1-1e-12)
                Y = self.truncated_dist.ppf(U)
                K = np.power(Y, 10.0)
            else:
                raise ValueError("copula ∈ {'gaussian','t'}.")

            Ks.append(K)
            if return_latent: Zs.append(Z)

        K_ens = np.stack(Ks, 0)
        if return_latent:
            Z_ens = np.stack(Zs, 0)
            return K_ens, Z_ens, seeds
        return K_ens, seeds

# ---- Clipped baseline (no copula) ------------------------------------
class ClippedLognormalFieldGenerator:
    def __init__(self, bounds: Tuple[float,float], mu10: float, sigma10: float):
        self.a, self.b = float(bounds[0]), float(bounds[1])
        self.mu10, self.sigma10 = float(mu10), float(sigma10)

    def log10_pdf(self, x: np.ndarray, *, kind: str = "auto") -> np.ndarray:
        """
        PDF of log10(K) for plotting overlays.

        Default ('auto') returns the *untruncated* normal N(mu10, sigma10^2),
        because this generator actually CLIPS in K-space (which adds point masses
        at the bounds). If you prefer a truncated-normal overlay in log10 anyway,
        pass kind='trunc' (note: that won't match the true clipped marginal either).
        """
        if kind == "trunc":
            loga, logb = np.log10(self.a), np.log10(self.b)
            a_std = (loga - self.mu10) / self.sigma10
            b_std = (logb - self.mu10) / self.sigma10
            tn = truncnorm(a=a_std, b=b_std, loc=self.mu10, scale=self.sigma10)
            mean = tn.mean()
            var = tn.var()
            return tn.pdf(x)
        # default: simple normal overlay in log10
        mean = norm(loc=self.mu10, scale=self.sigma10).mean()
        var = norm(loc=self.mu10, scale=self.sigma10).var()
        return norm(loc=self.mu10, scale=self.sigma10).pdf(x)

    def generate_ensemble(self, x: np.ndarray, y: np.ndarray, n_realizations: int, *,
                          len_scale: float, angles: float=0.0,
                          master_seed: int=20170519):
        model = gs.Exponential(dim=2, var=self.sigma10**2, len_scale=len_scale, angles=angles)
        #srf = gs.SRF(model, mean=self.mu10); srf.set_pos([x,y], "structured")

        master = MasterRNG(master_seed)
        Ks, seeds = [], []
        for _ in range(int(n_realizations)):
            s = master(); seeds.append(s)
            srf = gs.SRF(model, mean=self.mu10, seed=s)
            Y = srf((x, y), mesh_type="structured")
            K = 10**Y
            K = np.clip(K, self.a, self.b)
            Ks.append(K)
        return np.stack(Ks, 0), seeds
