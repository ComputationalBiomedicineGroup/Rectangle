import pandas as pd
from anndata import AnnData
from loguru import logger
from pandas import DataFrame
from pkg_resources import resource_stream

from .pp import RectangleSignatureResult, build_rectangle_signatures
from .tl import deconvolution


def rectangle(
    adata: AnnData,
    bulks: DataFrame,
    cell_type_col: str = "cell_type",
    *,
    layer: str = None,
    raw: bool = False,
    correct_mrna_bias: bool = True,
    optimize_cutoffs=True,
    p=0.015,
    lfc=1.5,
    n_cpus: int = None,
    gene_expression_threshold=0.5,
) -> tuple[DataFrame, RectangleSignatureResult]:
    r"""All in one deconvolution method. Creates signatures and deconvolutes the bulk data. Has options for subsampling and consensus runs.

    Parameters
    ----------
    adata
        The single-cell count data as a DataFrame. DataFrame must have the genes as index and cell identifier as columns. Each entry should be in raw counts.
    bulks
        The bulk data as a DataFrame. DataFrame must have the bulk identifier as index and the genes as columns. Each entry should be in transcripts per million (TPM).
    cell_type_col
        The annotations corresponding to the single-cell count data. Series data should have the cell identifier as index and the annotations as values.
    layer
        The Anndata layer to use for the single-cell data.
    raw
        A flag indicating whether to use the raw Anndata data.
    optimize_cutoffs
        Indicates whether to optimize the p-value and log fold change cutoffs using gridsearch.
    p
        The p-value threshold for the DE analysis (only used if optimize_cutoffs is False).
    lfc
        The log fold change threshold for the DE analysis (only used if optimize_cutoffs is False).
    n_cpus
        The number of cpus to use for the DE analysis. None value takes all cpus available.
    correct_mrna_bias : bool
        A flag indicating whether to correct for mRNA bias. Defaults to True.
    gene_expression_threshold : float
        The threshold for gene expression. Genes with expression below this threshold are removed from the analysis.

    Returns
    -------
    DataFrame : The estimated cell fractions.
    RectangleSignatureResult : The result of the rectangle signature analysis.
    """
    assert isinstance(adata, AnnData), "adata must be an AnnData object"
    assert isinstance(bulks, DataFrame), "bulks must be a DataFrame"

    signatures = build_rectangle_signatures(
        adata,
        cell_type_col,
        bulks=bulks,
        optimize_cutoffs=optimize_cutoffs,
        layer=layer,
        raw=raw,
        p=p,
        lfc=lfc,
        n_cpus=n_cpus,
        gene_expression_threshold=gene_expression_threshold,
    )

    estimations, bulk_err = deconvolution(signatures, bulks, correct_mrna_bias, n_cpus)
    signatures.unkn_bulk_err = bulk_err
    if "Unknown" in estimations.columns:
        try:
            unkn_gene_corr = _genes_linked_to_unkn(bulks, estimations["Unknown"], bulk_err)
        except Exception as e:
            logger.warning(f"Could not calculate gene correlation with unknown cell type: {e}")
            unkn_gene_corr = None
    else:
        unkn_gene_corr = None

    signatures.unkn_gene_corr = unkn_gene_corr

    return estimations, signatures


def load_tutorial_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the single-cell count data, annotations, and bulk data from the tutorial.

    Returns
    -------
    The single-cell count data, annotations, and bulk data.
    """
    with resource_stream(__name__, "data/hao1_annotations_small.zip") as annotations_file:
        annotations = pd.read_csv(annotations_file, index_col=0, compression="zip")["0"]

    with resource_stream(__name__, "data/hao1_counts_small.zip") as counts_file:
        sc_counts = pd.read_csv(counts_file, index_col=0, compression="zip").astype("int")

    with resource_stream(__name__, "data/small_fino_bulks.zip") as bulks_file:
        bulks = pd.read_csv(bulks_file, index_col=0, compression="zip")

    return sc_counts.T, annotations, bulks.T


def _genes_linked_to_unkn(bulks: DataFrame, unkn_fractions: pd.Series, bulk_err: DataFrame) -> pd.DataFrame:
    genes = bulk_err.columns.drop_duplicates()
    corr_expr = []
    for gene in genes:
        corr_expr.append(unkn_fractions.corr(bulks.loc[:, gene]))

    corr_err = []
    for gene in genes:
        corr_err.append(unkn_fractions.corr(bulk_err.loc[:, gene]))

    corr_expr = pd.Series(corr_expr, index=genes)
    corr_err = pd.Series(corr_err, index=genes)

    df = pd.DataFrame({"corr_expr": corr_expr, "corr_err": corr_err})

    return df
