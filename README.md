# AODF

GPU-accelerated python implementation of ODF filtering algorithm for asymmetric ODF estimation in diffusion MRI. The method is presented in [Poirier et al. (2024)](https://doi.org/10.1016/j.neuroimage.2024.120516).

> [!IMPORTANT]
> AODF filtering is now available as a [**Scilpy**](https://github.com/scilus/scilpy) script (`scil_sh_to_aodf.py`)! In addition to the CPU and GPU OpenCL implementations, **Scilpy** also implements a pure-python version of the filtering. We recommend new and current `aodf-toolkit` users to migrate to **Scilpy** where it will be more easily maintainable.

> [!NOTE]
>  Filter calibration is not yet available through **Scilpy** but should be integrated in the near future.

For installing `aodf-toolkit`, see [Installation  section](https://github.com/CHrlS98/aodf-toolkit/tree/main?tab=readme-ov-file#installation) below.

## Citing
If you use the method, please cite. A BibTeX entry is given below.
```
@article{poirier_unified_2024,
    title = {A unified filtering method for estimating asymmetric orientation distribution functions},
    journal = {NeuroImage},
    pages = {120516},
    year = {2024},
    issn = {1053-8119},
    doi = {https://doi.org/10.1016/j.neuroimage.2024.120516},
    url = {https://www.sciencedirect.com/science/article/pii/S1053811924000119},
    author = {Charles Poirier and Maxime Descoteaux},
}
```

## Installation
The application can be installed using `pip`. The application is tested with `python 3.10`. We recommend installing the application inside a virtual environment (see python `venv` instructions [here](https://docs.python.org/3/library/venv.html)). To install, run the following command from the project root:

```
pip install -e .
```

## Usage
Once installed, the application can be called from the command line:
```
aodf_filter in_sh.nii.gz out_sh.nii.gz
```
For tuning the hyper-parameter values (filter sigmas) to the data, an automatic calibration script is provided:
```
aodf_calibrate_filter in_sh.nii.gz calibrate_output.csv
```
The script outputs a csv file containing the self-supervised loss associated with different hyper-parameter values.

For a description of available options, use `--help` option.
