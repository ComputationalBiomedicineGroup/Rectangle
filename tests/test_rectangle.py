import pandas as pd
from anndata import AnnData

import rectanglepy.rectangle
from rectanglepy import ConsensusResult
from rectanglepy.pp import RectangleSignatureResult


def test_load_tutorial_data():
    sc_data, annotations, bulks = rectanglepy.load_tutorial_data()
    assert isinstance(sc_data, pd.DataFrame)
    assert isinstance(annotations, pd.Series)
    assert isinstance(bulks, pd.DataFrame)


def test_rectangle():
    sc_data, annotations, bulks = rectanglepy.load_tutorial_data()
    sc_data = sc_data.iloc[:, :2000]
    sc_data_adata = AnnData(sc_data, obs=annotations.to_frame(name="cell_type"))

    result = rectanglepy.rectangle(sc_data_adata, bulks)
    estimations, signatures = result

    assert isinstance(estimations, pd.DataFrame)
    assert isinstance(signatures, RectangleSignatureResult)
    assert isinstance(signatures.unkn_gene_corr, pd.Series)


def test_rectangle_consensus():
    sc_data, annotations, bulks = rectanglepy.load_tutorial_data()
    sc_data = sc_data.iloc[:, :500]
    sc_data_adata = AnnData(sc_data, obs=annotations.to_frame(name="cell_type"))

    result = rectanglepy.rectangle_consens(
        sc_data_adata, bulks, optimize_cutoffs=False, p=0.2, lfc=0.0, consensus_runs=3, sample_size=50
    )

    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], RectangleSignatureResult)
    assert isinstance(result[2], ConsensusResult)

    estimations = result[0]
    consensus_results = result[2]
    signature_results = consensus_results.rectangle_signature_results
    bias_factors = [result.bias_factors for result in signature_results]
    # there was a problem with the random sampling, the bias factors should differ
    assert bias_factors[0]["Monocytes"] != bias_factors[1]["Monocytes"]
    assert bias_factors[0]["Monocytes"] != bias_factors[2]["Monocytes"]
    assert bias_factors[1]["Monocytes"] != bias_factors[2]["Monocytes"]
    # in the consensus we have to make sure that the estimations are again normalized to 1
    assert estimations.sum(axis=1).all() == 1.0
