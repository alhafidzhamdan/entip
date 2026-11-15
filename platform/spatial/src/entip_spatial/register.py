"""Register an assay to its H&E scan.

Step 1, landmarks: fit an affine (6 dof) or similarity (4 dof: rotation,
uniform scale, shift) map from assay micrometres to H&E micrometres using
>= 3 landmark pairs picked by a person. Errors are reported in micrometres:
fit RMSE, leave-one-out RMSE and the worst landmark.

Step 2, optional refinement for single-cell assays (Xenium, Slide-tags):
iterate mutual nearest neighbours between transformed assay cells and H&E
nuclei, refitting on the matched pairs (trimmed ICP). The landmark fit must
already be close (within a few cell diameters) for this to help.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.spatial import cKDTree

from .ontology import Provenance, Transform, new_id

Model = Literal["affine", "similarity"]


def _as_pts(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("points must be (n, 2)")
    return a


def apply(matrix: np.ndarray, pts) -> np.ndarray:
    pts = _as_pts(pts)
    M = np.asarray(matrix, dtype=np.float64)
    return pts @ M[:2, :2].T + M[:2, 2]


def invert(matrix) -> np.ndarray:
    return np.linalg.inv(np.asarray(matrix, dtype=np.float64))


def _fit_affine(src, dst) -> np.ndarray:
    A = np.hstack([src, np.ones((len(src), 1))])
    X, *_ = np.linalg.lstsq(A, dst, rcond=None)  # (3, 2)
    M = np.eye(3)
    M[:2, :] = X.T
    return M


def _fit_similarity(src, dst, allow_reflection: bool = False) -> np.ndarray:
    """Umeyama (1991) least-squares similarity."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if not allow_reflection and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    var_s = (s ** 2).sum() / len(src)
    scale = np.trace(np.diag(S) @ D) / var_s
    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2] = mu_d - scale * R @ mu_s
    return M


MIN_PTS = {"affine": 3, "similarity": 2}


def fit(src, dst, model: Model = "affine", allow_reflection: bool = False) -> np.ndarray:
    src, dst = _as_pts(src), _as_pts(dst)
    if len(src) != len(dst):
        raise ValueError("src and dst need the same number of points")
    if len(src) < MIN_PTS[model]:
        raise ValueError(f"{model} needs at least {MIN_PTS[model]} point pairs, got {len(src)}")
    if model == "affine":
        M = _fit_affine(src, dst)
        if np.linalg.matrix_rank(np.hstack([src, np.ones((len(src), 1))])) < 3:
            raise ValueError("landmarks are collinear; affine is undetermined")
        return M
    return _fit_similarity(src, dst, allow_reflection)


def residuals_um(M, src, dst) -> np.ndarray:
    return np.linalg.norm(apply(M, src) - _as_pts(dst), axis=1)


def loo_rmse_um(src, dst, model: Model = "affine") -> Optional[float]:
    """Leave-one-out RMSE: refit without each landmark and measure its error."""
    src, dst = _as_pts(src), _as_pts(dst)
    n = len(src)
    if n <= MIN_PTS[model]:
        return None
    errs = []
    for i in range(n):
        keep = np.arange(n) != i
        try:
            M = fit(src[keep], dst[keep], model)
        except ValueError:
            return None
        errs.append(residuals_um(M, src[i:i + 1], dst[i:i + 1])[0])
    return float(np.sqrt(np.mean(np.square(errs))))


@dataclass
class IcpResult:
    matrix: np.ndarray
    n_matched: int
    median_residual_um: float
    iterations: int


def refine_icp(M0, assay_pts, he_pts, model: Model = "affine", max_dist_um: float = 10.0,
               iters: int = 30, trim: float = 0.9, tol_um: float = 1e-3) -> IcpResult:
    """Trimmed ICP on mutual nearest neighbours (assay cells vs H&E nuclei)."""
    a, h = _as_pts(assay_pts), _as_pts(he_pts)
    tree_h = cKDTree(h)
    M = np.asarray(M0, dtype=np.float64).copy()
    med, n, it = float("nan"), 0, 0
    for it in range(1, iters + 1):
        ta = apply(M, a)
        d_ah, j = tree_h.query(ta, distance_upper_bound=max_dist_um)
        ok = np.isfinite(d_ah)
        if ok.sum() < 3 * MIN_PTS[model]:
            break
        # mutual: the H&E nucleus's nearest transformed assay cell is this one
        _, back = cKDTree(ta).query(h[j[ok]])
        idx = np.flatnonzero(ok)[back == np.flatnonzero(ok)]
        if len(idx) < 3 * MIN_PTS[model]:
            break
        d = d_ah[idx]
        keep = idx[d <= np.quantile(d, trim)]
        M_new = fit(a[keep], h[j[keep]], model)
        shift = float(np.max(np.linalg.norm(apply(M_new, a[keep]) - apply(M, a[keep]), axis=1)))
        M, n, med = M_new, len(keep), float(np.median(residuals_um(M_new, a[keep], h[j[keep]])))
        if shift < tol_um:
            break
    return IcpResult(matrix=M, n_matched=n, median_residual_um=med, iterations=it)


def register(assay_id: str, scan_id: str, assay_landmarks, he_landmarks, model: Model = "affine",
             refine_assay_pts=None, refine_he_pts=None, max_dist_um: float = 10.0,
             provenance: Optional[Provenance] = None) -> Transform:
    """Fit a stored Transform assay:<id> -> scan:<id>, with errors in micrometres.

    Landmark errors are always measured with the final matrix, so a bad
    refinement shows up as a larger landmark RMSE rather than hiding.
    """
    src, dst = _as_pts(assay_landmarks), _as_pts(he_landmarks)
    M = fit(src, dst, model)
    method, meta = "landmarks", {}
    if refine_assay_pts is not None and refine_he_pts is not None:
        res = refine_icp(M, refine_assay_pts, refine_he_pts, model=model, max_dist_um=max_dist_um)
        M, method = res.matrix, "landmarks+icp"
        meta = {"icp_matched": res.n_matched, "icp_median_residual_um": res.median_residual_um,
                "icp_iterations": res.iterations}
    r = residuals_um(M, src, dst)
    det = float(np.linalg.det(M[:2, :2]))
    meta.update({"scale": float(np.sqrt(abs(det))), "reflected": det < 0})
    return Transform(
        transform_id=new_id("tf"), src_frame=f"assay:{assay_id}", dst_frame=f"scan:{scan_id}",
        model=model, matrix=M.tolist(), rmse_um=float(np.sqrt(np.mean(r ** 2))),
        loo_rmse_um=loo_rmse_um(src, dst, model), max_err_um=float(r.max()), n_landmarks=len(src),
        method=method, provenance=provenance or Provenance.program("register"), meta=meta,
    )


def to_he(adata, tf: Transform, key: str = "spatial_he_um"):
    """Write H&E-frame micrometre positions into ``adata.obsm[key]``."""
    adata.obsm[key] = apply(np.array(tf.matrix), adata.obsm["spatial_um"])
    adata.uns.setdefault("entip", {})["transform_id"] = tf.transform_id
    return adata
