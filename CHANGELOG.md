# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Changed

-   The cutoff grid search now scores each candidate on 100 pseudo-bulks instead of 50,
    halving the variance of the objective it selects on. The count is exposed as
    `RectangleAdvancedParameters.grid_search_bulks`; it was previously hard-coded.
    `grid_search_split_size` is deliberately left at 50: it caps how many cells go into
    each pseudo-bulk rather than how many are drawn, and raising it makes the pseudo-bulks
    easier to deconvolve, which compresses the differences between cutoffs instead of
    resolving them.
-   Dependencies are no longer pinned to exact or capped versions. Rectangle now runs on
    the current releases of numpy (2.x), pandas (3.x), scipy, anndata and pydeseq2.
-   `pydeseq2>=0.5.4` is required: 0.5 moved to formulaic designs and made
    `DeseqStats(contrast=...)` mandatory, and 0.5.0-0.5.3 crash on numpy>=2.5.
-   Minimum Python is now 3.11 (pydeseq2 0.5.4 dropped 3.10); Python 3.13 and 3.14 are
    supported and tested in CI.
-   `scikit-learn` and `joblib` are now declared explicitly; both were already imported.

-   The cutoff grid search no longer takes a strict argmax over the Pearson r scores.
    Scores within `CUTOFF_SELECTION_TOLERANCE` (0.01) of the best are treated as
    indistinguishable, and the strictest (highest) logFC among them is chosen. The scores
    come from 50 random pseudo-bulks, so margins below that are noise: previously the
    selected cutoff — and with it the whole marker set — could flip on float-level
    differences between numpy or BLAS builds. When the near-optimal band reaches the top
    of the logFC search space the strictest indistinguishable cutoff is set by the grid
    rather than by the data, so the best-scoring cutoff is used instead and a warning is
    logged.

### Fixed

-   Grid-search results were ranked with pandas' default unstable sort, so equal scores
    were ordered arbitrarily.
-   CPM normalisation of the pseudo-bulk signature used `np.sum(dataframe)`, which reduces
    over both axes from pandas 3 on instead of per column. Left unfixed this silently
    changed cell-fraction estimates; the per-column reduction is now explicit.
-   Sparse single-cell matrices could not be subset with a pandas boolean mask on
    scipy>=1.15.

## [1.4.2] - 2026-05-24

### Changed

-   Changed the open-source license from GPLv3 to BSD-3-Clause while retaining the commercial licensing option.

## [1.4.1] - 2026-04-23

### Changed

-   Wider QP tolerances

## [1.4.0] - 2026-03-10

### Changed

-   **Licensing: Rectangle is available under a dual license.**
    -   Non-commercial use: GNU General Public License v3 (GPLv3).
    -   Commercial use: A proprietary license is available for companies wishing to use Rectangle in closed-source or proprietary applications. For enquiries, contact innovation-psb@uibk.ac.at.
    -   See the commercial license terms for details.
-   Added Contributor License Agreement (CLA) for external contributions.

## [1.2.0] - 2026-01-31

### Changed

-   Switched the quadratic programming solver from quadprog to OSQP.

## [1.0.0] - 2025-08-14

### Added

-   Public release of Rectangle with full core functionalities and documentation for cell-type deconvolution.

## [0.1.0] - 2024-04-25

### Added

-   Initial public release of Rectangle.
