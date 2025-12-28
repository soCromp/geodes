from typing import List, Tuple, Optional, Callable
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*np.trace(covmean))


def _temporal_resample(video, T):
    # duplicates frames at evenly-spaced intervals to reach T frames
    cur_T = video.shape[1]
    if cur_T == T or T is None:
        return video 
    
    idx = np.linspace(0, cur_T-1, num=T).round().astype(int)
    return video[:, idx]


class PCAEncoder:
    def __init__(self, n_components=256, whiten=False):
        self.scaler = StandardScaler(with_mean=True, with_std=False)
        self.pca = PCA(n_components=n_components, whiten=whiten)
        self.fitted = False 
        self.target_T = None 
        
    def fit(self, videos, target_T=None):
        # videos should be a numpy matrix (N, T, C, H, W)
        assert videos.ndim == 4 or videos.ndim == 5, \
            "Input videos should be a numpy matrix (N, T, C, H, W) or (N, T, H, W)" 
        self.target_T = target_T 
        videos = _temporal_resample(videos, target_T)
        videos_flat = videos.reshape(videos.shape[0], -1)
        X = self.scaler.fit_transform(videos_flat)
        self.pca.fit(X)
        self.fitted = True 
        return self 
    
    def encode(self, videos):
        assert self.fitted, "PCAEncoder not fitted yet"
        assert videos.ndim == 4 or videos.ndim == 5, \
            "Input videos should be a numpy matrix (N, T, C, H, W) or (N, T, H, W)" 
        videos = _temporal_resample(videos, self.target_T)
        videos_flat = videos.reshape(videos.shape[0], -1)
        X = self.scaler.transform(videos_flat)
        X_pca = self.pca.transform(X)
        return X_pca
    
    
def compute_fvd(videos_real, videos_synth, target_T=None, 
                n_components=256, encoder=None):
    """videos should be numpy matrices (N, T, C, H, W) or (N, T, H, W).
    Can pass pre-fitted encoder"""
    assert videos_real.ndim == videos_synth.ndim, f'ndim mismatch: {videos_real.ndim} vs {videos_synth.ndim}'
    assert videos_real.ndim == 4 or videos_real.ndim == 5
    
    if encoder is None:
        encoder = PCAEncoder(n_components=n_components, whiten=True)
        encoder.fit(videos_real, target_T=target_T)
    
    feats_real = encoder.encode(videos_real)
    feats_synth= encoder.encode(videos_synth)
    
    mu_real = feats_real.mean(axis=0)
    cov_real = np.cov(feats_real, rowvar=False)
    mu_synth = feats_synth.mean(axis=0)
    cov_synth = np.cov(feats_synth, rowvar=False)
    return _frechet_distance(mu_real, cov_real, mu_synth, cov_synth), encoder


def _pairwise_rbf_sum(A, B, sigmas, chunk=1024):
    """
    Sum_{i,j} mean_{sigma in sigmas} exp(-||A_i - B_j||^2 / (2*sigma^2))
    computed in memory-friendly blocks. Returns a scalar (float64).
    """
    A = A.astype(np.float64, copy=False)
    B = B.astype(np.float64, copy=False)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    inv_2sigma2 = 1.0 / (2.0 * sigmas**2)  # shape (S,)
    S = float(len(sigmas))
    
    total = 0.0
    # precompute squared norms for efficiency
    An = (A*A).sum(axis=1)[:, None]  # (a,1)
    Bn = (B*B).sum(axis=1)[None, :]  # (1,b)
    
    for i in range(0, len(A), chunk):
        Ai = A[i:i+chunk]
        Ani = An[i:i+chunk]
        for j in range(0, len(B), chunk):
            Bj = B[j:j+chunk]
            Bnj = Bn[:, j:j+chunk]
            # pairwise squared distances (i-block x j-block)
            # d2_{p,q} = ||Ai_p||^2 + ||Bj_q||^2 - 2 Ai_p · Bj_q
            d2 = Ani + Bnj - 2.0 * (Ai @ Bj.T)  # shape (bi, bj)
            # accumulate multi-kernel average
            # exp(-d2 * inv_2sigma2) for each sigma, average over sigmas
            # vectorized over sigmas via broadcasting
            # Result per (p,q) = mean_s exp(-d2 * inv_2sigma2[s])
            K_block = np.exp(-d2[..., None] * inv_2sigma2).mean(axis=2)
            total += K_block.sum()
            
    return float(total)


def compute_kvd(videos_real, videos_synth, target_T=None,
                n_components=256, encoder=None, sigmas=None,
                chunk=1024, unbiased=True):
    """videos should be numpy matrices (N, T, C, H, W) or (N, T, H, W).
    Can pass pre-fitted encoder"""
    assert videos_real.ndim == videos_synth.ndim
    assert videos_real.ndim == 4 or videos_real.ndim == 5
    
    if encoder is None:
        encoder = PCAEncoder(n_components=n_components, whiten=True)
        encoder.fit(videos_real, target_T=target_T)
        
    feats_real = encoder.encode(videos_real)
    feats_synth= encoder.encode(videos_synth)
    
    mu_real  = feats_real.mean(axis=0)
    cov_real = np.cov(feats_real, rowvar=False)
    mu_synth = feats_synth.mean(axis=0)
    cov_synth= np.cov(feats_synth, rowvar=False)
    
    m, n = len(feats_real), len(feats_synth)
    if sigmas is None:
        feats_all = np.vstack([feats_real, feats_synth])
        rng = np.random.default_rng(0)
        max_pairs = 20000
        if len(feats_all) <= 1:
            med = 1.0
        else:
            i = rng.integers(0, len(feats_all), size=max_pairs)
            j = rng.integers(0, len(feats_all), size=max_pairs)
            mask = i != j
            i, j = i[mask], j[mask]
            diffs = feats_all[i] - feats_all[j]
            dists = np.sqrt((diffs**2).sum(axis=1))
            med = float(max(np.median(dists), 1e-6))
            sigmas = [med / 2, med, med * 2]
    else:
        sigmas = list(sigmas)
        
    # Kxx, Kyy (include diagonal), Kxy
    Kxx = _pairwise_rbf_sum(feats_real, feats_real, sigmas, chunk=chunk)
    Kyy = _pairwise_rbf_sum(feats_synth, feats_synth, sigmas, chunk=chunk)
    Kxy = _pairwise_rbf_sum(feats_real, feats_synth, sigmas, chunk=chunk)
    
    if unbiased:
        # For RBF, k(x,x)=1 for every sigma; with averaging it’s still 1
        Kxx_off = Kxx - m  # remove diagonal ones
        Kyy_off = Kyy - n
        mmd2 = (Kxx_off / (m*(m-1) + 1e-12)
                + Kyy_off / (n*(n-1) + 1e-12)
                - 2.0 * Kxy / (m*n))
    else:
        mmd2 = (Kxx / (m*m) + Kyy / (n*n) - 2.0 * Kxy / (m*n))

    return max(float(mmd2), 0.0), med, sigmas, encoder
    
        