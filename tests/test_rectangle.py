import pandas as pd
from anndata import AnnData

import rectanglepy.rectangle
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
