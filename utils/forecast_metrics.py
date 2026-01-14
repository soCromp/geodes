import numpy as np
from dataclasses import dataclass

# Outline:
# - RMSE metric (stats class and accumulator object)
# - Power Spectral Density (PSD)


def _nan_to_num(x):
    return np.where(np.isfinite(x), x, 0.0)

@dataclass
class RmseStats:
    # Weighted sum of squared errors and weights
    sse_overall: float
    w_overall: float
    sse_lead: np.ndarray   # (T,)
    w_lead:   np.ndarray   # (T,)
    sse_var:  np.ndarray   # (V,)
    w_var:    np.ndarray   # (V,)
    sse_lv:   np.ndarray   # (T,V)
    w_lv:     np.ndarray   # (T,V)
    # Baseline (persistence) totals
    base_sse_overall: float
    base_w_overall: float

    @staticmethod
    def init(T, V):
        z = np.zeros
        return RmseStats(
            sse_overall=0.0, w_overall=0.0,
            sse_lead=z(T, dtype=np.float64), w_lead=z(T, dtype=np.float64),
            sse_var=z(V, dtype=np.float64),  w_var=z(V, dtype=np.float64),
            sse_lv=z((T, V), dtype=np.float64), w_lv=z((T, V), dtype=np.float64),
            base_sse_overall=0.0, base_w_overall=0.0
        )

    def merge(self, other: "RmseStats"):
        self.sse_overall += other.sse_overall; self.w_overall += other.w_overall
        self.sse_lead += other.sse_lead; self.w_lead += other.w_lead
        self.sse_var  += other.sse_var;  self.w_var  += other.w_var
        self.sse_lv   += other.sse_lv;   self.w_lv   += other.w_lv
        self.base_sse_overall += other.base_sse_overall
        self.base_w_overall   += other.base_w_overall
        return self

    def finalize(self):
        def safe_div(a, b):
            return np.divide(a, np.maximum(b, 1e-12), dtype=np.float64)
        overall_rmse = float(np.sqrt(safe_div(self.sse_overall, self.w_overall)))
        rmse_by_lead = np.sqrt(safe_div(self.sse_lead, self.w_lead))
        rmse_by_var  = np.sqrt(safe_div(self.sse_var,  self.w_var))
        rmse_by_lv   = np.sqrt(safe_div(self.sse_lv,   self.w_lv))
        baseline_rmse = float(np.sqrt(safe_div(self.base_sse_overall, self.base_w_overall)))
        skill_vs_persistence = 1.0 - overall_rmse / (baseline_rmse + 1e-12)
        return dict(
            overall_rmse=overall_rmse,
            rmse_by_lead=rmse_by_lead,
            rmse_by_var=rmse_by_var,
            rmse_by_var_and_lead=rmse_by_lv,
            baseline_rmse=baseline_rmse,
            skill_vs_persistence=skill_vs_persistence,
        )

class RmseAccumulator:
    """
    Streaming RMSE for batches of forecasts.
    y_pred, y_true : (B, T, H, W, V) per batch (float32 ok)
    standardize: z-score using training-set mean/std per variable (V,)
    var_weights: (V,), 
    mask: broadcastable to (B,T,H,W,1)
    """
    def __init__(self, T, H, W, V,
                 mean_per_var=None, std_per_var=None, 
                 standardize=False, var_weights=None, ):
        self.T, self.H, self.W, self.V = T, H, W, V
        self.standardize = standardize

        # per-var z-score params
        if standardize:
            assert mean_per_var is not None and std_per_var is not None
            m = np.asarray(mean_per_var, dtype=np.float64).reshape(1,1,1,1,V)
            s = np.asarray(std_per_var,  dtype=np.float64).reshape(1,1,1,1,V)
            self.mean = m
            self.std  = np.where(s==0.0, 1.0, s)
        else:
            self.mean = None; self.std = None

        # weights, pre-broadcasted
        self.vw = np.ones((1,1,1,1,V), dtype=np.float64) if var_weights is None \
                  else np.asarray(var_weights, dtype=np.float64).reshape(1,1,1,1,V)

        self.stats = RmseStats.init(T, V)

    def _prep(self, arr):  # cast + standardize
        x = arr.astype(np.float64, copy=False)
        if self.standardize:
            x = (x - self.mean) / self.std
        return x

    def update(self, y_pred, y_true, mask=None):
        # shapes
        B, T, H, W, V = y_true.shape
        assert (T,H,W,V) == (self.T,self.H,self.W,self.V), "shape mismatch"
        # cast / standardize
        yp = self._prep(y_pred)
        yt = self._prep(y_true)

        # weights (B,T,H,W,V)
        base_w = self.vw
        if mask is not None:
            mk = np.asarray(mask, dtype=np.float64)
            # allow (B,H,W), (B,1,H,W), or (B,T,H,W)
            if mk.ndim == 3: mk = mk[:,None,...]  # -> (B,1,H,W)
            if mk.shape[1] == 1: mk = np.repeat(mk, T, axis=1)
            mk = mk[..., None]  # add var dim
            base_w = base_w * mk
        w = np.broadcast_to(base_w, (B,T,H,W,V))

        # ignore NaNs
        finite = np.isfinite(yp) & np.isfinite(yt)
        w = np.where(finite, w, 0.0)

        # squared errors
        se = (yp - yt)**2
        
        se = np.where(finite, se, 0.0) #from nans

        # overall
        se_sum = _nan_to_num((se * w).sum(dtype=np.float64))
        w_sum  = w.sum(dtype=np.float64)
        self.stats.sse_overall += se_sum
        self.stats.w_overall   += w_sum

        # by lead
        se_t = (se * w).sum(axis=(0,2,3,4), dtype=np.float64)   # (T,)
        w_t  = w.sum(axis=(0,2,3,4), dtype=np.float64)
        self.stats.sse_lead += se_t; self.stats.w_lead += w_t

        # by var
        se_v = (se * w).sum(axis=(0,1,2,3), dtype=np.float64)   # (V,)
        w_v  = w.sum(axis=(0,1,2,3), dtype=np.float64)
        self.stats.sse_var += se_v; self.stats.w_var += w_v

        # by (lead, var)
        se_tv = (se * w).sum(axis=(0,2,3), dtype=np.float64)    # (T,V)
        w_tv  = w.sum(axis=(0,2,3), dtype=np.float64)
        self.stats.sse_lv += se_tv; self.stats.w_lv += w_tv

        # persistence baseline: predict frame 0 for all leads
        baseline = np.broadcast_to(yt[:, :1], yt.shape)          # (B,T,H,W,V)
        base_se  = (baseline - yt)**2
        base_sse_sum = _nan_to_num((base_se * w).sum(dtype=np.float64))
        self.stats.base_sse_overall += base_sse_sum
        self.stats.base_w_overall   += w_sum

    def results(self):
        return self.stats.finalize()


# Power Spectral Density (PSD)
@dataclass
class PsdStats:
    # Accumulated power per frequency bin for Pred and True
    # Shape: (V, n_bins)
    power_pred: np.ndarray
    power_true: np.ndarray
    count: int
    
    # The wavenumbers (x-axis for plotting)
    k_freqs: np.ndarray

    @staticmethod
    def init(V, n_bins, k_freqs):
        z = np.zeros
        return PsdStats(
            power_pred=z((V, n_bins), dtype=np.float64),
            power_true=z((V, n_bins), dtype=np.float64),
            count=0,
            k_freqs=k_freqs
        )

    def finalize(self):
        # Average over batch/time samples
        # resulting shape: (V, n_bins)
        avg_pred = self.power_pred / max(self.count, 1)
        avg_true = self.power_true / max(self.count, 1)
        
        return dict(
            k_wavenumbers=self.k_freqs,
            psd_pred=avg_pred,
            psd_true=avg_true,
        )

class PsdAccumulator:
    """
    Streaming Power Spectral Density (1D Azimuthal Average).
    Computes the energy at different spatial scales.
    
    y_pred, y_true : (B, T, H, W, V)
    """
    def __init__(self, H, W, V):
        self.H, self.W, self.V = H, W, V
        
        # --- Pre-compute Radial Wavenumber Bins ---
        # 1. Get frequencies corresponding to FFT
        ky = np.fft.fftfreq(H)
        kx = np.fft.fftfreq(W)
        
        # 2. Create 2D grid of radial frequencies (k = sqrt(kx^2 + ky^2))
        k_grid_x, k_grid_y = np.meshgrid(kx, ky)
        k_radial = np.sqrt(k_grid_x**2 + k_grid_y**2)
        
        # 3. Flatten and sort to create bins
        # We bin pixels based on their distance from the DC component (0,0)
        self.k_flat = k_radial.flatten()
        
        # Quantize k into integer bins for accumulation
        # We scale up so bins are distinct enough, or just use unique values
        # A simple robust way is to use indices of unique values
        k_unique, self.bin_indices = np.unique(self.k_flat, return_inverse=True)
        self.n_bins = len(k_unique)
        self.k_vals = k_unique
        
        self.stats = PsdStats.init(V, self.n_bins, self.k_vals)

    def _compute_1d_spectrum(self, img_batch):
        """
        Input: (N, H, W, V) where N = B*T
        Output: (V, n_bins) accumulated power spectrum
        """
        N, H, W, V = img_batch.shape
        
        # 1. 2D FFT over H and W axes
        # Result is complex64/128
        fft_out = np.fft.fft2(img_batch, axes=(1, 2))
        
        # 2. Power Spectrum (Amplitude squared)
        # Normalization: / (H*W)**2 is standard for physical consistency,
        # but for relative comparison (sharp vs blur), raw magnitude is fine.
        # We will normalize by pixel count to keep numbers sane.
        power_2d = (np.abs(fft_out) ** 2) / (H * W)
        
        # 3. Azimuthal Average (Summing rings)
        # Reshape to (N, H*W, V) to match flattened bin indices
        power_flat = power_2d.reshape(N, H * W, V)
        
        # We need to sum over the spatial pixels (H*W) based on bin_indices
        # Output shape should be (V, n_bins)
        # Since np.bincount is 1D, we loop over V (usually small) or use add.at
        
        accum_power = np.zeros((V, self.n_bins), dtype=np.float64)
        
        for v in range(V):
            # Average over the batch dim (N) first to save memory/loops
            # (H*W,) mean power for this variable
            mean_power_v = power_flat[:, :, v].mean(axis=0) 
            
            # Sum into radial bins
            # bincount sums weights (power) for each bin index
            accum_power[v] = np.bincount(self.bin_indices, weights=mean_power_v, minlength=self.n_bins)
            
        return accum_power

    def update(self, y_pred, y_true):
        # Flatten B and T into a single batch dimension
        B, T, H, W, V = y_true.shape
        
        # Reshape to (N, H, W, V)
        yp = y_pred.reshape(-1, H, W, V)
        yt = y_true.reshape(-1, H, W, V)
        
        # Handle NaNs: FFT cannot handle NaNs. 
        # Simple strategy: replace with 0.0 (or mean). 
        # If your data has land masks (NaNs), this might introduce artifacts, 
        # but it's better than crashing.
        yp = np.nan_to_num(yp, nan=0.0)
        yt = np.nan_to_num(yt, nan=0.0)

        # Compute spectra
        spec_p = self._compute_1d_spectrum(yp)
        spec_t = self._compute_1d_spectrum(yt)
        
        # Accumulate
        self.stats.power_pred += spec_p
        self.stats.power_true += spec_t
        self.stats.count += 1

    def results(self):
        return self.stats.finalize()
