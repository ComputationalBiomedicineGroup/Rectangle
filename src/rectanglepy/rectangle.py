import pandas as pd
from anndata import AnnData
from loguru import logger
from pandas import DataFrame
from pkg_resources import resource_stream

from .pp import RectangleSignatureResult, build_rectangle_signatures
from .tl import deconvolution


class ConsensusResult:
    """A class used to represent the consensus result of the rectangle_consens function.

    Parameters
    ----------
    estimations
        A list of DataFrame objects representing the estimations from each consensus run.
    rectangle_signature_results
        A list of RectangleSignatureResult objects representing the rectangle signature results from each consensus run.
    """

    def __init__(self, estimations: list[DataFrame], rectangle_signature_results: list[RectangleSignatureResult]):
        self.estimations = estimations
        self.rectangle_signature_results = rectangle_signature_results


def rectangle_consens(
    adata: AnnData,
    bulks: DataFrame,
    cell_type_col: str = "cell_type",
    *,
    layer: str = None,
    raw: bool = False,
    subsample: bool = True,
    sample_size: int = 1500,
    consensus_runs: int = 5,
    correct_mrna_bias: bool = True,
    optimize_cutoffs=True,
    p=0.015,
    lfc=1.5,
    n_cpus: int = None,
    gene_expression_threshold=0.5,
) -> tuple[DataFrame, RectangleSignatureResult, ConsensusResult]:
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
        The Anndata layer to use for the single-cell data. Defaults to None.
    raw
        A flag indicating whether to use the raw Anndata data. Defaults to False.
    subsample : bool
        A flag indicating whether to balance the single-cell data.
    sample_size : int
        The number of cells to balance the single-cell data to. If cell number is less than this number it takes the original number of cells.
    consensus_runs : int
        The number of consensus runs to perform. Consensus runs are performed by subsampling the single-cell data and running the analysis multiple times. The results are then aggregated.
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
    DataFrame : The estimated cell fractions consens.
    RectangleSignatureResult : The result of the last consensus run.
    ConsensusResult : Estimations and rectangle signature results for each consensus run.
    """
    assert isinstance(adata, AnnData), "adata must be an AnnData object"
    assert isinstance(bulks, DataFrame), "bulks must be a DataFrame"

    if consensus_runs > 1:
        logger.info(f"Running {consensus_runs} consensus runs with subsample size {sample_size}")
        subsample = True

    estimations = []
    rectangle_signature_results = []
    most_recent_signatures = None

    for i in range(consensus_runs):
        logger.info(f"Running consensus run {i + 1} of {consensus_runs}")
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
            subsample=subsample,
            sample_size=sample_size,
            run=i,
            gene_expression_threshold=gene_expression_threshold,
        )
        most_recent_signatures = signatures
        cell_fractions = deconvolution(signatures, bulks, correct_mrna_bias, n_cpus)
        estimations.append(cell_fractions)
        unkn_gene_corr = _genes_linked_to_unkn(bulks, cell_fractions["Unknown"])
        signatures.unkn_gene_corr = unkn_gene_corr
        rectangle_signature_results.append(signatures)
    consensus_estimations = pd.concat(estimations).groupby(level=0).median()

    # normalize the estimations to 1, needed for the consensus
    consensus_estimations = consensus_estimations.div(consensus_estimations.sum(axis=1), axis=0)
    consensus_results = ConsensusResult(estimations, rectangle_signature_results)
    return consensus_estimations, most_recent_signatures, consensus_results


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
    estimations, signatures, _ = rectangle_consens(
        adata,
        bulks,
        cell_type_col,
        layer=layer,
        raw=raw,
        subsample=False,
        sample_size=-1,
        consensus_runs=1,
        correct_mrna_bias=correct_mrna_bias,
        optimize_cutoffs=optimize_cutoffs,
        p=p,
        lfc=lfc,
        n_cpus=n_cpus,
        gene_expression_threshold=gene_expression_threshold,
    )
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


def _genes_linked_to_unkn(bulks: DataFrame, unkn_fractions: pd.Series):
    genes = bulks.columns
    corr = []
    for gene in genes:
        corr.append(unkn_fractions.corr(bulks.loc[:, gene]))
    return pd.Series(corr, index=genes).sort_values(ascending=False)
