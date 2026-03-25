# Rectangle

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/ComputationalBiomedicineGroup/Rectangle/build.yaml?branch=main
[link-tests]: https://github.com/ComputationalBiomedicineGroup/Rectangle/actions/workflows/build.yaml
[badge-docs]: https://img.shields.io/readthedocs/rectanglepy

Rectangle is an open-source Python package for single-cell-informed cell-type deconvolution of bulk and spatial transcriptomic data, which is part of the [scverse ecosystem](https://scverse.org/packages/).

Rectangle presents a novel approach to second-generation deconvolution, characterized by hierarchical signature building for fine-grained cell-type deconvolution, estimation and correction of unknown cellular content, and efficient handling of large-scale single-cell data during signature matrix computation.

Rectangle was developed to overcome the current challenges in cell-type deconvolution, providing a robust and accurate methodology while ensuring a low computational profile.

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [Tutorials][link-docs/tutorials] for a step-by-step guide on how to use Rectangle, and the

-   [API documentation][link-api].

## Installation

You need Python 3.10–3.12 installed on your system.

How to install Rectangle:

Install the latest release of `Rectangle` from `PyPI` <https://pypi.org/project/rectanglepy/>:

```bash
pip install rectanglepy
```

## Licence

Rectangle is available under a **dual licence**:

-   **Open-source licence:** [GNU General Public License v3.0 (GPLv3)](LICENSE)
    → Free to use, modify, and redistribute as long as modifications and redistributions are also under GPLv3.

-   **Commercial licence:** For companies and individuals who wish to use Rectangle in proprietary or closed-source applications, a separate commercial licence is available. See [LICENCE_COM](LICENCE_COM) for details.

For commercial licensing enquiries, please contact: **innovation-psb@uibk.ac.at**

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

For commercial licensing: **innovation-psb@uibk.ac.at**

## Citation

> If you use Rectangle in your project, please cite: (TBA)

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/ComputationalBiomedicineGroup/Rectangle/issues
[changelog]: https://rectanglepy.readthedocs.io/changelog.html
[link-docs]: https://Rectanglepy.readthedocs.io
[link-api]: https://rectanglepy.readthedocs.io/api.html
[link-docs/tutorials]: https://rectanglepy.readthedocs.io/notebooks/example.html
