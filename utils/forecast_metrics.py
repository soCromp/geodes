import numpy as np
from dataclasses import dataclass

# Outline:
# - RMSE metric (stats class and accumulator object)
# - Anomaly correlation

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



