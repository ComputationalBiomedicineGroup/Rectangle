import math
import multiprocessing

import numpy as np
import pandas as pd
import quadprog
import statsmodels.api as sm
from joblib import Parallel, delayed, parallel_backend
from loguru import logger

from rectanglepy.pp.rectangle_signature import RectangleSignatureResult


def _scale_weights(weights: np.ndarray) -> np.ndarray:
    min_weight = np.nextafter(min(weights), np.float64(1.0))  # prevent division by zero
    return weights / min_weight


def solve_qp(
    signature: pd.DataFrame,
    bulk: pd.Series,
    prev_assignments: list[int or str] = None,
    prev_solution: pd.Series = None,
    gld: np.ndarray = None,
    multiplier: int = None,
) -> np.ndarray:
    """Performs quadratic programming optimization to solve the deconvolution problem.

    Parameters
    ----------
    signature : pd.DataFrame
        The signature matrix for deconvolution. Each row represents a gene and each column represents a cell type.
    bulk : pd.Series
        The bulk data for deconvolution. Each entry represents the expression level of a gene.
    prev_assignments : default is None
        A list of previous assignments of cell types to genes. If provided, the function will use these assignments as additional QP constraints.
    prev_solution : default is None
        A series of previous solution for each cell type. If provided, the function will use the prev solution as additional QP constraints..
    gld :  default is None
        An array of gene length data. If provided, the function will use this data to adjust the weights.
    multiplier :  default is None
        A multiplier for the weights. If provided, the function will use this multiplier to adjust the weights.

    Returns
    -------
    np.ndarray
        An array of corrected cell fractions. Each entry represents the fraction of a cell type in the bulk data.

    Notes
    -----
    This function uses quadratic programming to solve the deconvolution problem. The objective is to minimize the difference between the observed bulk data and the data predicted by the signature matrix and the cell fractions. The function also includes constraints to ensure that the cell fractions are non-negative and sum to 1, and to make the solution similar to the previous assignments and weights if they are provided.
    """
    # ------------------ QP-based deconvolution
    # Minimize     1/2 x^T G x - a^T x
    # Subject to   C.T x >= b
    if multiplier is None:
        a = np.dot(signature.T, bulk).astype("double")
        G = np.dot(signature.T, signature).astype("double")
    else:
        weights = np.square(1 / (signature @ gld))
        weights_dampened = np.clip(_scale_weights(weights), None, multiplier)
        W = np.diag(weights_dampened)
        G = np.dot(signature.T, np.dot(W, signature))
        a = np.dot(signature.T, np.dot(W, bulk))

    # Compute the constraints matrices
    n_genes = G.shape[0]
    # Constraint 1: sum of all fractions is below or equal to 1
    C1 = -np.ones((n_genes, 1))
    b1 = -np.ones(1)
    # Constraint 2: every fraction is greater or equal to 0
    C2 = np.eye(n_genes)
    b2 = np.zeros(n_genes)
    # Constraint 3: if a previous solution is provided, the new solution should be similar to the previous one
    prev_constraints, prev_b = [], []
    if prev_solution is not None:
        for cluster in prev_solution.index:
            C_upper = [-np.ones(1) if str(x) == str(cluster) else np.zeros(1) for x in prev_assignments]
            C_lower = [np.ones(1) if str(x) == str(cluster) else np.zeros(1) for x in prev_assignments]
            prev_constraints.extend([C_upper, C_lower])
            prev_weight = prev_solution.loc[cluster]
            # allow a 3% variation in the previous weights
            prev_weight_upper = min(1, prev_weight + 0.03)
            prev_weight_lower = max(0, prev_weight - 0.03)
            prev_b.extend([prev_weight_upper * np.ones(1) * -1, prev_weight_lower * np.ones(1)])

    C = np.concatenate((C1, C2, *prev_constraints), axis=1)
    b = np.concatenate((b1, b2, *prev_b), axis=0)

    scale = np.linalg.norm(G)
    solution = quadprog.solve_qp(G / scale, a / scale, C, b, factorized=False)
    return solution[0]


def _find_dampening_constant(signature: pd.DataFrame, bulk: pd.Series, qp_gld: np.ndarray) -> int:
    solutions_std = []
    np.random.seed(1)
    weights = np.square(1 / (np.dot(signature, qp_gld)))
    weights_scaled = _scale_weights(weights)
    weights_scaled_no_inf = weights_scaled[weights_scaled != np.inf]
    qp_gld_sum = sum(qp_gld)
    # try multiple values of the dampening constant (multiplier)
    # for each, calculate the variance of the dampened weighted solution for a subset of genes
    max_range = 40
    multiplier_range = min(max_range, math.ceil(np.log2(max(weights_scaled_no_inf))))
    for i in range(multiplier_range):
        solutions = []
        multiplier = 2**i
        weights_dampened = np.array([multiplier if multiplier <= x else x for x in weights_scaled]).astype("double")
        for _ in range(100):
            subset = np.random.choice(len(signature), size=len(signature) // 2, replace=False)
            bulk_subset = bulk.iloc[list(subset)]
            signature_subset = signature.iloc[subset, :]
            fit = sm.WLS(bulk_subset, -1 + signature_subset, weights=weights_dampened[subset]).fit()
            solution = fit.params * qp_gld_sum / sum(fit.params)
            solutions.append(solution)
        solutions_df = pd.DataFrame(solutions)

        solutions_std.append(solutions_df.std(axis=0))
    solutions_std_df = pd.DataFrame(solutions_std)
    means = solutions_std_df.apply(lambda x: np.mean(x**2), axis=1)
    best_dampening_constant = means.idxmin()
    return best_dampening_constant


def _calculate_dwls(
    signature: pd.DataFrame,
    bulk: pd.Series,
    prev_assignments: list[int or str] = None,
    prev_weights: pd.Series = None,
) -> pd.Series:
    genes = list(set(signature.index) & set(bulk.index))
    signature = signature.loc[genes].sort_index()
    bulk = bulk.loc[genes].sort_index().astype("double")

    approximate_solution = solve_qp(signature, bulk, prev_assignments, prev_weights)
    dampening_constant = _find_dampening_constant(signature, bulk, approximate_solution)
    multiplier = 2**dampening_constant

    max_iterations = 1000
    convergence_threshold = 0.002
    change = 1
    iterations = 2
    solutions_sum = approximate_solution
    while (change > convergence_threshold) and (iterations < max_iterations):
        dampened_solution = solve_qp(signature, bulk, prev_assignments, prev_weights, approximate_solution, multiplier)
        solutions_sum += dampened_solution
        solution_averages = solutions_sum / iterations
        change = np.linalg.norm(solution_averages - approximate_solution, 1)
        approximate_solution = solution_averages
        iterations += 1

    if iterations == max_iterations:
        logger.warning("Dampened weighted least squares did not converge, using last solution")

    return pd.Series(approximate_solution, index=signature.columns)


def deconvolution(
    signatures: RectangleSignatureResult,
    bulks: pd.DataFrame,
    correct_mrna_bias: bool = True,
    n_cpus: int = None,
) -> pd.DataFrame:
    """Performs recursive deconvolution using rectangle signatures and bulk data.

    Parameters
    ----------
    signatures : RectangleSignatureResult
        The rectangle signature result containing the signature data and results.
    bulks : pandas.DataFrame
        The tpm normalized bulk data for deconvolution. Rows are samples and columns are genes.
    correct_mrna_bias : bool
        A flag indicating whether to correct for mRNA bias. Defaults to True.
    n_cpus : int
        The number of CPUs to use for parallel processing. If not provided, the function will use all available CPUs. Each CPU will process a bulk sample.

    Returns
    -------
        A DataFrame containing the estimated cell fractions resulting from deconvolution. Each row represents a sample and each column represents a cell type.

    """
    bulks = bulks.div(bulks.sum(axis=1), axis=0) * 1e6

    if n_cpus is not None:
        num_processes = n_cpus
    else:
        num_processes = multiprocessing.cpu_count()

    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(
            n_jobs=num_processes,
        )(
            delayed(_process_bulk)(signatures, i, bulk, bulks.columns, correct_mrna_bias)
            for i, bulk in enumerate(bulks.values)
        )

    result_df = pd.DataFrame(results, index=bulks.index)

    # Return the result DataFrame
    return result_df


def _process_bulk(
    signatures: RectangleSignatureResult, i: int, bulk: pd.Series, var_names: pd.Index, correct_mrna_bias: bool
) -> pd.Series:
    try:
        logger.info(f"Deconvolute fractions for bulk: {i}")
        bulk = pd.Series(bulk, index=var_names)
        result = _deconvolute(signatures, bulk, correct_mrna_bias=correct_mrna_bias)
        logger.info(f"Finished deconvolution for bulk: {i}")
        return result
    except Exception as e:
        logger.warning(f"Deconvolution failed for bulk: {i} with error: {e}")
        return pd.Series(index=signatures.pseudobulk_sig_cpm.columns)


def _deconvolute(signatures: RectangleSignatureResult, bulk: pd.Series, correct_mrna_bias: bool = True) -> pd.Series:
    bulk_direct_reduced = bulk[bulk.index.isin(signatures.signature_genes)]
    signature_genes_direct_reduced = signatures.signature_genes[
        signatures.signature_genes.isin(bulk_direct_reduced.index)
    ]
    pseudobulk_sig_cpm = signatures.pseudobulk_sig_cpm
    clustered_pseudobulk_sig_cpm = signatures.clustered_pseudobulk_sig_cpm
    bias_factors = signatures.bias_factors

    if not correct_mrna_bias:
        bias_factors = bias_factors * 0 + 1  # set all bias factors to 1

    signature = pseudobulk_sig_cpm.loc[signature_genes_direct_reduced] * bias_factors
    start_fractions = _calculate_dwls(signature, bulk)

    if clustered_pseudobulk_sig_cpm is None:
        start_fractions = correct_for_unknown_cell_content(bulk, pseudobulk_sig_cpm, start_fractions, bias_factors)
        return start_fractions

    cluster_bias_factors = signatures.clustered_bias_factors
    if not correct_mrna_bias:
        cluster_bias_factors = cluster_bias_factors * 0 + 1

    bulk_rec_reduced = bulk[bulk.index.isin(clustered_pseudobulk_sig_cpm.index)]
    clustered_signature_genes = signatures.clustered_signature_genes[
        signatures.clustered_signature_genes.isin(bulk_rec_reduced.index)
    ]
    clustered_signature = clustered_pseudobulk_sig_cpm.loc[clustered_signature_genes] * cluster_bias_factors

    try:
        clustered_fractions = _calculate_dwls(clustered_signature, bulk)
        recursive_fractions = _calculate_dwls(signature, bulk, signatures.assignments, clustered_fractions)
    except Exception as e:
        logger.warning(f"Recursive deconvolution failed with error: {e}")
        start_fractions = correct_for_unknown_cell_content(bulk, pseudobulk_sig_cpm, start_fractions, bias_factors)
        return start_fractions

    final_fractions = []

    low_number_threshold = 20
    cell_types_with_low_number_of_marker_genes = [
        cell_type
        for cell_type, num_marker_genes in signatures.marker_genes_per_cell_type.items()
        if len(num_marker_genes) < low_number_threshold
    ]

    for cell_type in list(start_fractions.index):
        if cell_type in cell_types_with_low_number_of_marker_genes:
            final_fractions.append(recursive_fractions[cell_type])
        else:
            final_fractions.append(start_fractions[cell_type])

    final_fractions = pd.Series(final_fractions, index=start_fractions.index)

    final_fractions = correct_for_unknown_cell_content(bulk, pseudobulk_sig_cpm, final_fractions, bias_factors)
    return final_fractions


def correct_for_unknown_cell_content(
    bulk: pd.Series, pseudo_signature_cpm: pd.DataFrame, estimates: pd.Series, bias_factors: pd.Series
) -> pd.Series:
    r"""Performs correction for unknown cell content using the pseudo signature and bulk data.

    Reconstructs the bulk expression profiles through  the estimated cell fractions (weighted by the mRNA bias factors) and cell-type-specific expression profiles (i.e. signature).

    .. math::
        \text{bulk_est} = \text{pseudo_signature} \times (\text{estimates}^T \times \text{bias_factors})

    The unknown cellular content is then calculated as the difference of per-sample overall expression levels in the true vs. reconstructed bulk, divided by the overall expression in the true bulk.

    .. math::
         \text{ukn_cc} = \frac{\text{bulk} - \text{bulk_est}}{\sum_{i=1}^{n} x_i}

    Finally, the methods corrects (i.e. scales) the cell fraction estimates so that their sum equals 1 - the unknown cellular content and returns this corrected value to the user.

    .. math::
        \text{estimates_fix} = \frac{\text{estimates}}{\sum_{i=1}^{n} x_i} \times (1 - \text{ukn_cc})

    Parameters
    ----------
    bulk
        The bulk count data for deconvolution, indexed by gene. Normalized by TPM.
    pseudo_signature_cpm
        Averaged sc data, indexed by gene. Normalized by CPM. Contains all genes.
    estimates
        The estimated cell fractions resulting from the deconvolution. Indexed by cell type.
    bias_factors
        The mRNA bias factors of the sc data atlas.

    Returns
    -------
    pd.Series: The corrected cell fractions, indexed by cell type. Adds an "Unknown" cell type.
    """
    if estimates.sum() == 0:
        estimates_fix = estimates
        # analysis fails if all cell fractions are zero, so we set the unknown cell content to ß
        estimates_fix["Unknown"] = 0
        return estimates_fix

    signature_genes = pseudo_signature_cpm.index
    bulk = bulk.loc[signature_genes]
    signature = pseudo_signature_cpm.sort_index()
    bulk = bulk.sort_index()

    # Reconstruct the bulk expression profiles through matrix multiplication
    # of the estimated cell fractions (weighted by the scaling factors) and
    # cell-type-specific expression profiles (i.e. signature)
    bulk_est = pd.Series(np.dot(signature, (estimates.T * bias_factors).T))
    bulk_est.index = signature.index

    # Calculate the unknown cellular content ad the difference of
    # per-sample overall expression levels in the true vs. reconstructed
    # bulk RNA-seq data, divided by the overall expression in the true bulk
    ukn_cc = (bulk.sum() - bulk_est.sum()) / (bulk.sum())
    ukn_cc = max(0, ukn_cc)
    # Correct (i.e. scale) the cell fraction estimates so that their sum
    # equals 1 - the unknown cellular content estimated above
    estimates_fix = estimates / estimates.sum() * (1 - ukn_cc)
    estimates_fix["Unknown"] = abs(ukn_cc)

    return estimates_fix
