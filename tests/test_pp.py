from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import rectanglepy as rectangle
from rectanglepy.pp.create_signature import (
    _assess_parameter_fit,
    _calculate_cluster_range,
    _create_annotations_from_cluster_labels,
    _create_bias_factors,
    _create_bootstrap_signature,
    _create_fclusters,
    _create_linkage_matrix,
    _create_pseudo_count_sig,
    _de_analysis,
    _generate_pseudo_bulks,
    _get_fcluster_assignments,
    _run_deseq2,
    build_rectangle_signatures,
)


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
def hao_signature(data_dir):
    hao_signature = pd.read_csv(data_dir / "signature_hao1.csv", index_col=0)
    return hao_signature


def test_create_linkage_matrix(hao_signature):
    linkage_matrix = _create_linkage_matrix(hao_signature)
    assert len(linkage_matrix) == 10


def test_create_fclusters(hao_signature):
    linkage_matrix = _create_linkage_matrix(hao_signature)
    clusters = rectangle.pp.create_signature._create_fclusters(hao_signature, linkage_matrix)
    assert clusters == [3, 4, 4, 6, 1, 7, 4, 2, 3, 8, 5]


# Define the test function with parameters
@pytest.mark.parametrize("test_input,expected", [(5, (3, 4)), (20, (16, 19))])
def test_calculate_cluster_range(test_input, expected):
    result = _calculate_cluster_range(test_input)
    assert result == expected


def test_get_fcluster_assignments(hao_signature):
    linkage_matrix = _create_linkage_matrix(hao_signature)
    clusters = _create_fclusters(hao_signature, linkage_matrix)
    assignments = _get_fcluster_assignments(clusters, hao_signature.columns)
    assert assignments == [3, 4, 4, "NK cells", "B cells", "pDC", 4, "Plasma cells", 3, "Platelet", "ILC"]


def test_create_annotations_from_cluster_labels(hao_signature):
    annotations = pd.Series(
        [
            "NK cells",
            "pDC",
            "Plasma cells",
            "ILC",
            "T cells CD8",
            "Platelet",
            "B cells",
            "mDC",
            "T cells CD4 conv",
            "Tregs",
            "Monocytes",
        ]
    )
    linkage_matrix = _create_linkage_matrix(hao_signature)
    clusters = _create_fclusters(hao_signature, linkage_matrix)
    assignments = _get_fcluster_assignments(clusters, hao_signature.columns)
    annotations_from_cluster = _create_annotations_from_cluster_labels(assignments, annotations, hao_signature)

    assert list(annotations_from_cluster) == [
        "NK cells",
        "pDC",
        "Plasma cells",
        "ILC",
        "4",
        "Platelet",
        "B cells",
        "3",
        "4",
        "4",
        "3",
    ]


def test_create_pseudo_count_signature(small_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    expected = sc_counts.groupby(annotations.values, axis=1).sum()

    adata = AnnData(sc_counts.T, obs=annotations.to_frame(name="cell_type"))
    pseudo_sig_count = _create_pseudo_count_sig(adata.X.T, adata.obs.cell_type, adata.var_names)

    sc_counts = sc_counts.astype(pd.SparseDtype("int"))
    adata_sparse = AnnData(sc_counts.T, obs=annotations.to_frame(name="cell_type"))
    pseudo_sig_count_sparse = _create_pseudo_count_sig(adata_sparse.X.T, adata_sparse.obs.cell_type, adata.var_names)

    assert (expected == pseudo_sig_count).all().all()
    assert (expected == pseudo_sig_count_sparse).all().all()


def test_create_bias_factors(small_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    adata = AnnData(sc_counts.T, obs=annotations.to_frame(name="cell_type"))

    expected = [1.290858550761261, 1.056939659633611, 1.0]

    sig_counts = _create_pseudo_count_sig(adata.X.T, adata.obs.cell_type, adata.var_names)
    bias_factors = _create_bias_factors(sig_counts, adata.X.T, annotations)

    sc_counts = sc_counts.astype(pd.SparseDtype("int"))
    csr_sparse_matrix = sc_counts.sparse.to_coo().tocsr()
    adata_sparse = AnnData(csr_sparse_matrix.T, obs=annotations.to_frame(name="cell_type"))
    sig_counts_sparse = _create_pseudo_count_sig(adata_sparse.X.T, adata_sparse.obs.cell_type, adata.var_names)
    bias_factors_sparse = _create_bias_factors(sig_counts_sparse, adata_sparse.X.T, annotations)

    assert np.allclose(expected, bias_factors)
    assert np.allclose(expected, bias_factors_sparse)


def test_build_rectangle_signatures(small_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    adata = AnnData(sc_counts.T, obs=annotations.to_frame(name="cell_type"))
    results = build_rectangle_signatures(adata, "cell_type", bulks=bulk.T, p=0.5, lfc=0.1, optimize_cutoffs=False)
    assert results.assignments is None  # not enough cells to cluster
    assert 10 < len(results.signature_genes) < 400


def test_generate_pseudo_bulks(small_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    adata = AnnData(sc_counts.T, obs=annotations.to_frame(name="cell_type"))
    result, _ = _generate_pseudo_bulks(adata.X.T, annotations, adata.var_names)

    sc_data = sc_counts.astype(pd.SparseDtype("int"))
    csr_sparse_matrix = sc_data.sparse.to_coo().tocsr()
    adata_sparse = AnnData(csr_sparse_matrix.T, obs=annotations.to_frame(name="cell_type"))

    result_sparse, _ = _generate_pseudo_bulks(adata_sparse.X.T, annotations, adata_sparse.var_names)

    assert len(result) == 1000 and len(result.columns) == 50
    # first gene should have all 0s
    assert result.iloc[0, :].sum() == 0
    assert np.allclose(result, result_sparse)


def test_asses_fit(small_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    sc_pseudo = sc_counts.groupby(annotations.values, axis=1).sum()
    de_result = _run_deseq2(sc_pseudo, sc_counts.values, annotations)

    adata = AnnData(sc_counts.T, obs=annotations.to_frame(name="cell_type"))
    bulks, real_fractions = _generate_pseudo_bulks(adata.X.T, annotations, adata.var_names)
    result = _assess_parameter_fit(0.1, 0.9, bulks, real_fractions, sc_pseudo, de_result)

    sc_data = sc_counts.astype(pd.SparseDtype("int"))
    csr_sparse_matrix = sc_data.sparse.to_coo().tocsr()
    adata_sparse = AnnData(
        csr_sparse_matrix.T, obs=annotations.to_frame(name="cell_type"), var=adata.var_names.to_frame(name="gene")
    )
    bulks_sparse, real_fractions_sparse = _generate_pseudo_bulks(adata_sparse.X.T, annotations, adata_sparse.var_names)
    result_sparse = _assess_parameter_fit(0.1, 0.9, bulks_sparse, real_fractions_sparse, sc_pseudo, de_result)

    assert len(result) == 2
    assert 0.10 < result[0] < 0.15
    assert np.allclose(result, result_sparse)


def test_de_analysis(small_data):
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    sc_pseudo = sc_counts.groupby(annotations.values, axis=1).sum()

    adata = AnnData(sc_counts.T, obs=annotations.to_frame(name="cell_type"))
    r1, r2, r3 = _de_analysis(sc_pseudo, adata.X.T, annotations, 0.4, 0.1, False, None, adata.var_names)

    sc_counts = sc_counts.astype(pd.SparseDtype("int"))
    csr_sparse_matrix = sc_counts.sparse.to_coo().tocsr()
    adata_sparse = AnnData(csr_sparse_matrix.T, obs=annotations.to_frame(name="cell_type"))
    # test with sparse matrix
    _ = _de_analysis(sc_pseudo, adata_sparse.X.T, annotations, 0.4, 0.1, False, None, adata.var_names)

    assert 5 < len(r1) < 50
    assert len(r2) == 3


def test_create_bootstrap_signature(small_data):
    bootstraps_per_cell = 7
    sc_counts, annotations, bulk = small_data
    sc_counts = sc_counts.astype("int")
    sc_pseudo = sc_counts.groupby(annotations.values, axis=1).sum()
    adata = AnnData(sc_counts.T, obs=annotations.to_frame(name="cell_type"))
    bootstrap = _create_bootstrap_signature(sc_pseudo, adata.X.T, annotations)

    assert len(bootstrap.columns) == len(sc_pseudo.columns) * bootstraps_per_cell
