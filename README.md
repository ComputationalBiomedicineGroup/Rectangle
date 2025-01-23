# Rectangle

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/ComputationalBiomedicineGroup/Rectangle/build.yaml?branch=main
[link-tests]: https://github.com/ComputationalBiomedicineGroup/Rectangle/actions/workflows/build.yaml
[badge-docs]: https://img.shields.io/readthedocs/rectanglepy

Rectangle is an open-source Python package developed for computational deconvolution.

Rectangle presents a novel approach to second-generation deconvolution, characterized by hierarchical processing,
an estimation of unknown cellular content and a significant reduction in data volume during signature matrix computation.

Rectangle was developed to outperform existing deconvolution solutions by introducing methods that promise
improvements in cell-type fraction estimation accuracy while keeping a low computational profile.

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.10 or higher installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

How to install Rectangle:

Install the latest release of `Rectangle` from `PyPI` <https://pypi.org/project/rectanglepy/>:

```bash
pip install rectanglepy
```

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/ComputationalBiomedicineGroup/Rectangle/issues
[changelog]: https://rectanglepy.readthedocs.io/changelog.html
[link-docs]: https://Rectanglepy.readthedocs.io
[link-api]: https://rectanglepy.readthedocs.io/api.html
