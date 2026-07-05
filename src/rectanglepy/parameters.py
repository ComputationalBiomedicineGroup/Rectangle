# SPDX-License-Identifier: BSD-3-Clause OR LicenseRef-Rectangle-Commercial

from dataclasses import dataclass


@dataclass(frozen=True)
class RectangleAdvancedParameters:
    """Advanced parameters for Rectangle internals."""

    number_of_bootstraps: int = 7
    grid_search_split_size: int = 50
