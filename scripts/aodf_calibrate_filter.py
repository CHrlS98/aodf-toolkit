#!/usr/bin/env python3
import argparse
from functools import partial
import nibabel as nib
import numpy as np
from aodf.filtering.aodf_filter import AsymmetricFilter
from aodf.filtering.utils import get_sf_range
from scilpy.reconst.multi_processes import peaks_from_sh
from scilpy.io.utils import add_sh_basis_args, get_sh_order_and_fullness
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf, sph_harm_ind_list

from skimage.restoration import calibrate_denoiser, denoise_invariant
from skimage.metrics.simple_metrics import mean_squared_error
import pandas as pd
import itertools

NBR_PROCESSES = 8


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_sh')
    p.add_argument('out_table')

    p.add_argument('--sphere_filter', default='repulsion200')
    p.add_argument('--sphere_nufid', default='repulsion724')
    p.add_argument('--at', default=0.1, type=float)
    p.add_argument('--rt', default=0.1, type=float)

    add_sh_basis_args(p)
    return p


def _sh_to_nufid(in_sh, sphere, sh_basis, a_threshold=0.1,
                 r_threshold=0.1, n_peaks=10, mask=None, nbr_processes=1):
    _, full_basis = get_sh_order_and_fullness(in_sh.shape[-1])
    _, values, _ =\
            peaks_from_sh(in_sh, sphere, relative_peak_threshold=r_threshold,
                          absolute_threshold=a_threshold, min_separation_angle=25,
                          normalize_peaks=False, npeaks=n_peaks,
                          sh_basis_type=sh_basis, nbr_processes=nbr_processes,
                          full_basis=full_basis, is_symmetric=False)
    nufid = np.count_nonzero(values, axis=-1).astype(float)
    return nufid


def _filter_to_nufid_functional(in_sh, sh_order, sh_basis, legacy, full_basis,
                                spatial, align, angle, range, sf_range,
                                channel_axis, nufid_sph):
    assert channel_axis == -1
    print('spatial: {}\nalign: {}\nrange: {}\nangle: {}'
          .format(spatial, align, range, angle))

    # cases for disabling filters
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
    print('(true range: {})\n'.format(range))

    afilter = AsymmetricFilter(sh_order, sh_basis, legacy, full_basis,
                               'repulsion200', spatial, align, angle, range,
                               disable_spatial=disable_spatial,
                               disable_align=disable_align,
                               disable_angle=disable_angle,
                               disable_range=disable_range)
    filtered_sh = afilter(in_sh)
    return _sh_to_nufid(filtered_sh, nufid_sph, sh_basis,
                        n_peaks=10, nbr_processes=NBR_PROCESSES)


def _filter_sh_functional(in_sh, sh_order, sh_basis, full_basis,
                          sphere_str, spatial, align, angle, range,
                          sf_range, channel_axis=-1):
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
                               legacy=True, full_basis=full_basis,
                               sphere_str=sphere_str,
                               sigma_spatial=spatial,
                               sigma_align=align,
                               sigma_range=range,
                               sigma_angle=angle,
                               disable_spatial=disable_spatial,
                               disable_align=disable_align,
                               disable_range=disable_range,
                               disable_angle=disable_angle,
                               j_invariance=False)

    # filter sh
    filtered_sh = afilter(in_sh)
    return filtered_sh


def _compute_symmetric_sf(in_sh, sphere, sh_basis, full_basis):
    order, full = get_sh_order_and_fullness(in_sh.shape[-1])
    if full:
        _, l_list = sph_harm_ind_list(order, full_basis=full)
        in_sh = in_sh[..., l_list % 2 == 0]
    # reconstruct symmetric part of signal
    return sh_to_sf(in_sh, sphere, order, sh_basis,
                    full_basis=False, legacy=True)


def _product_from_dict(dictionary):
    """Utility function to convert parameter ranges to parameter combinations.
    from skimage.restoration.j_invariant
    """
    keys = dictionary.keys()
    for element in itertools.product(*dictionary.values()):
        yield dict(zip(keys, element))


def _generate_grid_slice(shape, *, offset, stride=3):
    """Generate slices of uniformly-spaced points in an array.
    """
    phases = np.unravel_index(offset, (stride,) * len(shape))
    mask = tuple(slice(p, None, stride) for p in phases)

    return mask


def _calibrate_denoiser_search(in_sh, denoise_func, aggregate_func,
                               tested_parameters, stride=4):
    # grid mask for estimating loss
    spatialdims = in_sh.ndim - 1
    n_masks = stride ** spatialdims
    mask = _generate_grid_slice(in_sh.shape[:spatialdims],
                                offset=n_masks // 2, stride=stride)

    _, in_full = get_sh_order_and_fullness(in_sh.shape[-1])
    noisy_nufid = aggregate_func(in_sh, full_basis=in_full)
    losses = []
    parameters = []

    for args in _product_from_dict(tested_parameters):
        out_sh = denoise_invariant(
            in_sh, denoise_function=denoise_func,
            stride=stride, masks=[mask],
            out_shape=np.append(in_sh.shape[:-1], [81]),
            denoiser_kwargs=args)

        # proxy metrics on which to evaluate image quality
        denoised_nufid = aggregate_func(out_sh, full_basis=True)
        loss = mean_squared_error(noisy_nufid[mask], denoised_nufid[mask])

        print(args, loss)
        losses.append(loss)
        parameters.append(args)

    return parameters, losses


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    sh_im = nib.load(args.in_sh)
    in_sh = sh_im.get_fdata()

    # crop the data (60x60x60) to study WM
    # crop_size = 40
    # in_sh = in_sh[in_sh.shape[0]//2-crop_size//2:in_sh.shape[0]//2+crop_size//2,
    #               in_sh.shape[1]//2-crop_size//2:in_sh.shape[1]//2+crop_size//2,
    #               in_sh.shape[2]//2-crop_size//2:in_sh.shape[2]//2+crop_size//2]
    # print(in_sh.shape)
    # nib.save(nib.Nifti1Image(in_sh, sh_im.affine), 'debug.nii.gz')

    order, full = get_sh_order_and_fullness(in_sh.shape[-1])
    nufid_sph = get_sphere(args.sphere_nufid)
    sh_to_sf_sphere = get_sphere(args.sphere_filter)

    # filtering function
    sf_range = get_sf_range(in_sh, order, full, args.sphere_nufid)
    print('sf range:', sf_range)
    denoise_func = partial(_filter_sh_functional, sh_order=order,
                           sh_basis=args.sh_basis, full_basis=full,
                           sphere_str=args.sphere_filter, sf_range=sf_range)

    # function summarizing the content of the filtered output
    # aggregate_func = partial(_sh_to_nufid, sphere=nufid_sph,
    #                          sh_basis=args.sh_basis, a_threshold=args.at,
    #                          r_threshold=args.rt)
    # aggregate_func = partial(sh_to_sf, sphere=sh_to_sf_sphere, sh_order=order,
    #                          basis_type=args.sh_basis, legacy=True)

    aggregate_func = partial(_compute_symmetric_sf, sphere=sh_to_sf_sphere,
                             sh_basis=args.sh_basis)

    tested_sigmas = {
        'spatial':[0.6, 1.0, 2.0],
        'align': [0.8, 1.4, 2.0],
        'range': [0.2, 1.0, None],
        'angle': [None, 0.1, 0.2],
        'channel_axis': [-1]
    }
    max_spatial = np.max(tested_sigmas['spatial'])
    stride = int(max_spatial * 3.0 + 0.5) + 1
    print('stride: ', stride)

    params, losses = _calibrate_denoiser_search(in_sh, denoise_func=denoise_func,
                                                aggregate_func=aggregate_func,
                                                tested_parameters=tested_sigmas,
                                                stride=stride)

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

    return

    base_nufid = _sh_to_nufid(in_sh, nufid_sph, args.sh_basis, False, NBR_PROCESSES)

    sf_range = get_sf_range(in_sh, order, full, 'repulsion200')

    _denoise_func = partial(_filter_to_nufid_functional, sh_order=order,
                            sh_basis=args.sh_basis, legacy=True, full_basis=full,
                            nufid_sph=nufid_sph, sf_range=sf_range)

    tested_sigmas = {
        'spatial':[0.6, 1.0, 1.4],
        'align': [0.8, 1.4],
        'range': [0.2, 1.0, None],
        'angle': [None, 0.262],
        'channel_axis': [-1]
    }

    # set stride such that the filter does not use another
    # interpolated voxel when predicting an output
    max_spatial = np.max(tested_sigmas['spatial'])
    stride = int(max_spatial * 3.0 + 0.5) + 1
    print('stride: ', stride)

    # calibrate using "tweaked" skimage `calibrate_denoiser`
    _, extra_output = calibrate_denoiser(
        in_sh, _denoise_func, extra_output=True,
        denoise_parameters=tested_sigmas, ref=base_nufid,
        stride=stride, approximate_loss=False
    )

    # table output
    out_dict = {
        'spatial': [],
        'align': [],
        'angle': [],
        'range': [],
        'loss': []
    }
    for sigmas, loss in zip(*extra_output):
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
