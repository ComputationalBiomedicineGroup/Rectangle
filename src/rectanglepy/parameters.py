# SPDX-License-Identifier: BSD-3-Clause OR LicenseRef-Rectangle-Commercial

from dataclasses import dataclass


@dataclass(frozen=True)
class RectangleAdvancedParameters:
    """Advanced parameters for Rectangle internals."""

    number_of_bootstraps: int = 7
    #: Largest number of cells drawn per cell type when building one pseudo-bulk.
    grid_search_split_size: int = 50
    #: How many pseudo-bulks each cutoff is scored on. This is the sample size of the
    #: grid-search objective, so raising it reduces the noise in the selected cutoff.
    grid_search_bulks: int = 100
