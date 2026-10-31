"""Minimal reader/writer for the 10x Genomics feature-barcode HDF5 matrix
(``filtered_feature_bc_matrix.h5`` in Space Ranger, ``cell_feature_matrix.h5``
in Xenium). Layout: group ``matrix`` with ``barcodes``, ``data``, ``indices``,
``indptr``, ``shape`` (features x barcodes, CSC) and ``features/{id,name,feature_type}``.
Written against 10x's documented layout; tested here on synthetic files only.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def _str(a) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def read_10x_h5(path: str | Path, feature_types: tuple[str, ...] | None = ("Gene Expression",)) -> ad.AnnData:
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = tuple(int(x) for x in m["shape"][:])  # (n_features, n_barcodes)
        mat = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
        barcodes = _str(m["barcodes"][:])
        feats = m["features"]
        var = pd.DataFrame({
            "gene_ids": _str(feats["id"][:]),
            "feature_types": _str(feats["feature_type"][:]) if "feature_type" in feats else "Gene Expression",
        }, index=_str(feats["name"][:]))
    X = mat.T.tocsr().astype(np.float32)
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=barcodes), var=var)
    if feature_types is not None:
        adata = adata[:, adata.var["feature_types"].isin(feature_types).to_numpy()].copy()
    adata.var_names_make_unique()
    return adata


def write_10x_h5(path: str | Path, X, barcodes, genes, feature_types=None) -> Path:
    """Write a cells x genes matrix in the 10x HDF5 layout (used for synthetic data)."""
    X = sparse.csr_matrix(X)
    csc = X.T.tocsc()  # features x barcodes
    genes = list(genes)
    feature_types = feature_types or ["Gene Expression"] * len(genes)
    path = Path(path)
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        m.create_dataset("data", data=csc.data.astype(np.int32))
        m.create_dataset("indices", data=csc.indices.astype(np.int64))
        m.create_dataset("indptr", data=csc.indptr.astype(np.int64))
        m.create_dataset("shape", data=np.array(csc.shape, dtype=np.int32))
        g = m.create_group("features")
        g.create_dataset("id", data=np.array([f"SYN{i:05d}" for i in range(len(genes))], dtype="S"))
        g.create_dataset("name", data=np.array(genes, dtype="S"))
        g.create_dataset("feature_type", data=np.array(feature_types, dtype="S"))
        g.create_dataset("genome", data=np.array(["synthetic"] * len(genes), dtype="S"))
    return path
