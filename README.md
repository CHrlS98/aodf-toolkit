# AODF

GPU-accelerated python implementation of ODF filtering algorithm for asymmetric ODF estimation in diffusion MRI. The method is presented in [Poirier et al. (2024)](https://doi.org/10.1016/j.neuroimage.2024.120516).

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

## Citing
The method is described in a Neuroimage article. The corresponding BibTeX entry is given below.
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
