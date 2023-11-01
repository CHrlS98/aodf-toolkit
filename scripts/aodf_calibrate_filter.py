#!/usr/bin/env python3
import argparse
from functools import partial
import nibabel as nib
import numpy as np
from aodf.filtering.aodf_filter import AsymmetricFilter
from scilpy.io.utils import get_sh_order_and_fullness
from aodf.filtering.utils import get_sf_range, add_sh_basis_arg, parse_sh_basis
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf, sph_harm_ind_list
import os
import pandas as pd
import itertools


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_sh',
                   help='Input symmetric fODF image.')
    p.add_argument('out_table',
                   help='Output .csv table for filtering scores.')

    p.add_argument('--sphere_filter', default='repulsion200',
                   help='Sphere used for filtering. [%(default)s]')
    p.add_argument('--sphere_highres', default='repulsion724',
                   help='Sphere used for estimating range of SF amplitudes. '
                        '[%(default)s]')

    p.add_argument('--crop_size', type=int, default=60,
                   help='Width of the patch considered for calibration '
                        '(voxels). [%(default)s]')

    add_sh_basis_arg(p)
    return p


def _filter_sh_functional(in_sh, sh_order, sh_basis, full_basis,
                          sphere_str, spatial, align, angle, range,
                          sf_range, j_invariance=False, legacy=True):
    # if a parameter is None, disable the filter
    disable_spatial = disable_align = disable_angle = disable_range = False
    if spatial is None:
        disable_spatial = True
        spatial = 1.0
    if align is None:
        disable_align = True
        align = 1.0
    if angle is None:
        disable_angle = True
        angle = 1.0
    if range is None:
        disable_range = True
        range = 1.0

    # scale range parameter by SF range
    range *= sf_range

    # initialize j-invariant filter
    afilter = AsymmetricFilter(sh_order=sh_order, sh_basis=sh_basis,
                               legacy=legacy, full_basis=full_basis,
                               sphere_str=sphere_str,
                               sigma_spatial=spatial,
                               sigma_align=align,
                               sigma_range=range,
                               sigma_angle=angle,
                               disable_spatial=disable_spatial,
                               disable_align=disable_align,
                               disable_range=disable_range,
                               disable_angle=disable_angle,
                               j_invariance=j_invariance)

    # filter sh
    filtered_sh = afilter(in_sh)
    return filtered_sh


def _compute_symmetric_sf(in_sh, sphere, sh_basis, legacy=True):
    order, full = get_sh_order_and_fullness(in_sh.shape[-1])
    if full:
        _, l_list = sph_harm_ind_list(order, full_basis=full)
        in_sh = in_sh[..., l_list % 2 == 0]
    # reconstruct symmetric part of signal
    return sh_to_sf(in_sh, sphere, order, sh_basis,
                    full_basis=False, legacy=legacy)


def mse(image1, image2):
    return np.mean((image1 - image2)**2)


def _product_from_dict(dictionary):
    """Utility function to convert parameter ranges to parameter combinations.
    from skimage.restoration.j_invariant
    """
    keys = dictionary.keys()
    for element in itertools.product(*dictionary.values()):
        yield dict(zip(keys, element))


def crop_center(data, size):
    """
    Extract a patch of `size` voxels width at center of `data`.
    """
    return data[data.shape[0]//2-size//2:data.shape[0]//2+size//2,
                data.shape[1]//2-size//2:data.shape[1]//2+size//2,
                data.shape[2]//2-size//2:data.shape[2]//2+size//2]


def grid_search(in_sh, denoise_func, aggregate_func, tested_parameters, crop_size):
    # this will be the input symmetric SF
    noisy_input = aggregate_func(in_sh)
    losses = []
    parameters = []
    min_loss = np.inf

    for args in _product_from_dict(tested_parameters):
        out_sh = denoise_func(in_sh, **args)
        denoised = aggregate_func(out_sh)
        loss = mse(crop_center(noisy_input, crop_size),
                   crop_center(denoised, crop_size))

        if loss > min_loss:
            status = ''
        else:
            min_loss = loss
            status = '<--'
        print(args, loss, status)
        losses.append(loss)
        parameters.append(args)
    
    return parameters, losses


def assert_crop_volume(parser, crop_size, volume_shape):
    volume_shape = np.asarray(volume_shape)
    if (volume_shape < crop_size).any():
        parser.error('Crop size with padding exceeds volume dimensions.')


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    sh_im = nib.load(args.in_sh)
    in_sh = sh_im.get_fdata()

    _, ext = os.path.splitext(args.out_table)
    if ext != '.csv':
        parser.error('Invalid extension for output. Expected .csv.')

    sh_basis, legacy = parse_sh_basis(args.sh_basis)

    # TODO: These values should be set by argparser
    tested_sigmas = {
        'spatial':[1.0, 1.2, 1.4, 1.6, 1.8],
        'align': [0.4, 0.6, 0.8, 1.0, 1.2],
        'range': [0.2, 1.0, None],
        'angle': [None, 0.1, 0.2]
    }

    # crop the data and add padding by maximum filter width
    crop_size = args.crop_size + 2*int(3.0*np.max(tested_sigmas['spatial']) + 0.5)
    assert_crop_volume(parser, crop_size, in_sh.shape[:3])

    in_sh = in_sh[in_sh.shape[0]//2-crop_size//2:in_sh.shape[0]//2+crop_size//2,
                  in_sh.shape[1]//2-crop_size//2:in_sh.shape[1]//2+crop_size//2,
                  in_sh.shape[2]//2-crop_size//2:in_sh.shape[2]//2+crop_size//2]

    order, full = get_sh_order_and_fullness(in_sh.shape[-1])
    sh_to_sf_sphere = get_sphere(args.sphere_filter)

    # compute SF range for scaling sigma_range parameter
    sf_range = get_sf_range(in_sh, order, full, args.sphere_highres)
    print('SF range is:', sf_range)

    # filtering function
    denoise_func = partial(_filter_sh_functional, sh_order=order,
                           sh_basis=sh_basis, full_basis=full,
                           sphere_str=args.sphere_filter, sf_range=sf_range,
                           j_invariance=True, legacy=legacy)

    aggregate_func = partial(_compute_symmetric_sf,
                             sphere=sh_to_sf_sphere,
                             sh_basis=sh_basis,
                             legacy=legacy)

    # launch grid search
    params, losses = grid_search(in_sh, denoise_func=denoise_func,
                                 aggregate_func=aggregate_func,
                                 tested_parameters=tested_sigmas,
                                 crop_size=args.crop_size)

    # table output
    out_dict = {
        'spatial': [],
        'align': [],
        'angle': [],
        'range': [],
        'loss': []
    }
    for sigmas, loss in zip(params, losses):
        out_dict['spatial'].append(sigmas['spatial'])
        out_dict['align'].append(sigmas['align'])
        out_dict['angle'].append(sigmas['angle'])
        out_dict['range'].append(sigmas['range'])
        out_dict['loss'].append(loss)

    # save results to file
    df = pd.DataFrame(out_dict)
    df.to_csv(args.out_table, index=False)


if __name__ == '__main__':
    main()
