from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from rectanglepy.pp.create_signature import (
    build_rectangle_signatures,
)
from rectanglepy.tl.deconvolution import (
    _calculate_dwls,
    _scale_weights,
    correct_for_unknown_cell_content,
    deconvolution,
    solve_qp,
)


@pytest.fixture
def quantiseq_data(data_dir):
    signature = pd.read_csv(data_dir / "TIL10_signature.txt", index_col=0, sep="\t")
    bulk = pd.read_csv(data_dir / "quanTIseq_SimRNAseq_mixture_smaller.csv", index_col=0)
    fractions = pd.read_csv(data_dir / "quanTIseq_SimRNAseq_read_fractions_small.txt", index_col=0, sep="\t")
    fractions = fractions.iloc[:, :-1]
    return bulk, fractions, signature


@pytest.fixture
def data_dir():
    return Path(__file__).resolve().parent / "data"


@pytest.fixture
def small_data(data_dir):
    sc_data = pd.read_csv(data_dir / "sc_object_small.csv", index_col=0)
    annotations = list(pd.read_csv(data_dir / "cell_annotations_small.txt", header=None, index_col=0).index)
    annotations = pd.Series(annotations, index=sc_data.columns)
    bulk = pd.read_csv(data_dir / "bulk_small.csv", index_col=0)
    return sc_data, annotations, bulk


@pytest.fixture
def finotello_bulk(data_dir):
    bulk = pd.read_csv(data_dir / "finotello_bulk.csv", index_col=0, header=0)
    return bulk


def test_scale_weigths():
    weights = [1, 0]
    result = _scale_weights(weights)
    assert (result == [np.Inf, 0]).all()


def test_simple_weighted_dampened_deconvolution(quantiseq_data):
    bulk, real_fractions, signature = quantiseq_data
    j = 5
    bulk = bulk.iloc[:, j]
    expected = real_fractions.T.iloc[:, j]

    result = _calculate_dwls(signature, bulk)
    corr = np.corrcoef(result, expected)[0, 1]
    rsme = np.sqrt(np.mean((result - expected) ** 2))

    assert corr > 0.815 and rsme < 0.011


def test_correct_for_unknown_cell_content(small_data, quantiseq_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    adata = AnnData(sc_counts.T, obs=annotations.to_frame(name="cell_type"))
    signature = build_rectangle_signatures(adata, "cell_type", p=0.9, lfc=0.1, optimize_cutoffs=False)
    bulk, _, _ = quantiseq_data
    bulk = bulk.iloc[:, 5]
    pseudo_signature = signature.pseudobulk_sig_cpm
    sig = pseudo_signature.loc[signature.signature_genes]
    fractions = _calculate_dwls(sig, bulk)
    biasfact = (pseudo_signature > 0).sum(axis=0)
    biasfact = biasfact / biasfact.min()
    result, bulk_err = correct_for_unknown_cell_content(bulk, pseudo_signature, fractions, biasfact)
    assert len(fractions) + 1 == len(result)
    assert len(bulk_err) > 0


def test_solve_dampened_wsl(quantiseq_data):
    bulk, real_fractions, signature = quantiseq_data
    j = 5
    bulk = bulk.iloc[:, j]
    expected = real_fractions.T.iloc[:, j]

    genes = list(set(signature.index) & set(bulk.index))
    signature = signature.loc[genes].sort_index()
    bulk = bulk.loc[genes].sort_index().astype("double")

    result = solve_qp(signature, bulk)
    # evaluation metrics
    corr = np.corrcoef(result, expected)[0, 1]
    rsme = np.sqrt(np.mean((result - expected) ** 2))

    assert corr > 0.73 and rsme < 0.012


def test_deconvolute_no_hierarchy(small_data, quantiseq_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    adata = AnnData(sc_counts.T, obs=annotations.to_frame(name="cell_type"))
    signature = build_rectangle_signatures(adata, "cell_type", p=0.9, lfc=0.1, optimize_cutoffs=False)
    bulk, _, _ = quantiseq_data

    estimations, bulk_err = deconvolution(signature, bulk.T)
    assert np.allclose(estimations.sum(axis=1), 1)
    assert estimations.shape == (8, 4)
    assert len(bulk_err) == 8


def test_deconvolute_sparse_no_hierarchy(small_data, quantiseq_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    adata = AnnData(sc_counts.T, obs=annotations.to_frame(name="cell_type"))
    signature = build_rectangle_signatures(adata, "cell_type", p=0.9, lfc=0.1, optimize_cutoffs=False)
    bulk, _, _ = quantiseq_data

    expected, bulk_err_exp = deconvolution(signature, bulk.T)

    sc_counts = sc_counts.astype(pd.SparseDtype("int"))
    csr_sparse_matrix = sc_counts.sparse.to_coo().tocsr()
    adata_sparse = AnnData(
        csr_sparse_matrix.T, obs=annotations.to_frame(name="cell_type"), var=sc_counts.index.to_frame(name="gene")
    )
    signature_sparse = build_rectangle_signatures(adata_sparse, "cell_type", p=0.9, lfc=0.1, optimize_cutoffs=False)

    estimations, bulk_err = deconvolution(signature_sparse, bulk.T)
    assert expected.equals(estimations)
    assert bulk_err_exp.equals(bulk_err)
