import pandas as pd


class RectangleSignatureResult:
    """Represents the result of a rectangle signature analysis (Created by the method pp.build_rectangle_signatures).

    Parameters
    ----------
    signature_genes
        The signature genes as a pd.Series.
    bias_factors
        The mRNA bias factors associated with each cell type.
    pseudobulk_sig_cpm
        The pseudo bulk signature build from the single cell data, contains all genes. Normalized to CPM. Columns are cell types, rows are genes.
    clustered_pseudobulk_sig_cpm
        The  clustered pseudo bulk signature build from the single cell data, contains all genes. Normalized to CPM. Columns are cell types, rows are genes.
    clustered_bias_factors
        The bias factors associated with each cell type cluster.
    cluster_assignments
        The assignments of signature cell-types to clusters, as a list of ints or strings. In the same order as the pseudobulk_sig_cpm columns.
    marker_genes_per_cell_type
        The number of marker genes per cell type, as a dictionary. Keys are cell type names, values are the number of marker genes.
    optimization_result
        The result of the p lfc cut off optimization, as a pd.DataFrame. Contains the following columns: p, lfc, pearson_r, rsme
    unkn_gene_corr
        The correlation between the unknown cell type and the genes linked to the unknown cell type.
    """

    def __init__(
        self,
        signature_genes: pd.Series,
        bias_factors: pd.Series,
        pseudobulk_sig_cpm: pd.DataFrame,
        marker_genes_per_cell_type: dict[str, [str]],
        optimization_result: pd.DataFrame = None,
        clustered_pseudobulk_sig_cpm: pd.DataFrame = None,
        clustered_bias_factors: pd.Series = None,
        marker_genes_per_cluster: dict[str, [str]] = None,
        clustered_signature_genes: pd.Series = None,
        cluster_assignments: list[int or str] = None,
        unkn_gene_corr: pd.Series = None,
    ):
        self.signature_genes = signature_genes
        self.bias_factors = bias_factors
        self.pseudobulk_sig_cpm = pseudobulk_sig_cpm
        self.marker_genes_per_cluster = marker_genes_per_cluster
        self.marker_genes_per_cell_type = marker_genes_per_cell_type
        self.optimization_result = optimization_result
        self.clustered_pseudobulk_sig_cpm = clustered_pseudobulk_sig_cpm
        self.clustered_bias_factors = clustered_bias_factors
        self.clustered_signature_genes = clustered_signature_genes
        self.assignments = cluster_assignments
        self.unkn_gene_corr = unkn_gene_corr

    def get_signature_matrix(self, include_mrna_bias=True) -> pd.DataFrame:
        """Calculates the signature matrix by multiplying the pseudobulk_sig_cpm DataFrame subset by signature_genes and the bias_factors Series.

        Parameters
        ----------
        include_mrna_bias
            If True, the method includes mRNA bias in the calculation. Defaults to True.

        Returns
        -------
        pandas.DataFrame: The signature matrix. Where columns are cell types and rows are genes.

        """
        if include_mrna_bias:
            return self.pseudobulk_sig_cpm.loc[self.signature_genes] * self.bias_factors
        else:
            return self.pseudobulk_sig_cpm.loc[self.signature_genes]
