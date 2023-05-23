# AODF

GPU-accelerated python implementation of ODF filtering algorithm for asymmetric ODF estimation in diffusion MRI.

## Installation
The application can be installed using `pip`. The application is tested with `python 3.10`. We recommend installing the application inside a virtual environment (see python `venv` instructions [here](https://docs.python.org/3/library/venv.html)).

```
pip install -e .
```

## Usage
Once installed, the application can be called from the command line.
```
aodf_filter in_sh.nii.gz out_sh.nii.gz
```

For a description of possible options, `aodf_filter --help`.

## Citing
The method is described in an article (preprint). The corresponding BibTeX entry is given below.
```
@article{poirier_filtering_2022,
    author = {Poirier, Charles and Descoteaux, Maxime},
    title = {Filtering Methods for Asymmetric ODFs: Where and How Asymmetry Occurs in the White Matter},
    year = {2022},
    doi = {10.1101/2022.12.18.520881},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2022/12/19/2022.12.18.520881},
    journal = {bioRxiv}
}
```
