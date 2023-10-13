# -*- coding:utf-8 -*-
"""
Copy of the module scilpy.io.utils.py.
"""
import numpy as np
import os


def get_sh_order_and_fullness(ncoeffs):
    """
    Get the order of the SH basis from the number of SH coefficients
    as well as a boolean indicating if the basis is full.
    """
    # the two curves (sym and full) intersect at ncoeffs = 1, in what
    # case both bases correspond to order 1.
    sym_order = (-3.0 + np.sqrt(1.0 + 8.0 * ncoeffs)) / 2.0
    if sym_order.is_integer():
        return sym_order, False
    full_order = np.sqrt(ncoeffs) - 1.0
    if full_order.is_integer():
        return full_order, True
    raise ValueError('Invalid number of coefficients for SH basis.')


def assert_inputs_exist(parser, required, optional=None):
    """Assert that all inputs exist. If not, print parser's usage and exit.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    required: string or list of paths
        Required paths to be checked.
    optional: string or list of paths
        Optional paths to be checked.
    """
    def check(path):
        if not os.path.isfile(path):
            parser.error('Input file {} does not exist'.format(path))

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        if optional_file is not None:
            check(optional_file)


def assert_outputs_exist(parser, args, required, optional=None,
                         check_dir_exists=True):
    """
    Assert that all outputs don't exist or that if they exist, -f was used.
    If not, print parser's usage and exit.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    args: argparse namespace
        Argument list.
    required: string or list of paths to files
        Required paths to be checked.
    optional: string or list of paths to files
        Optional paths to be checked.
    check_dir_exists: bool
        Test if output directory exists.
    """
    def check(path):
        if os.path.isfile(path) and not args.overwrite:
            parser.error('Output file {} exists. Use -f to force '
                         'overwriting'.format(path))

        if check_dir_exists:
            path_dir = os.path.dirname(path)
            if path_dir and not os.path.isdir(path_dir):
                parser.error('Directory {}/ \n for a given output file '
                             'does not exists.'.format(path_dir))

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        if optional_file:
            check(optional_file)


def add_overwrite_arg(parser):
    parser.add_argument(
        '-f', dest='overwrite', action='store_true',
        help='Force overwriting of the output files.')


def add_processes_arg(parser):
    parser.add_argument('--processes', dest='nbr_processes',
                        metavar='NBR', type=int, default=1,
                        help='Number of sub-processes to start. \n'
                        'Default: [%(default)s]')


def add_sh_basis_args(parser, mandatory=False):
    """Add spherical harmonics (SH) bases argument.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    mandatory: bool
        Whether this argument is mandatory.
    """
    choices = ['descoteaux07', 'tournier07']
    def_val = 'descoteaux07'
    help_msg = 'Spherical harmonics basis used for the SH coefficients.\nMust ' +\
               'be either \'descoteaux07\' or \'tournier07\' [%(default)s]:\n' +\
               '    \'descoteaux07\': SH basis from the Descoteaux et al.\n' +\
               '                      MRM 2007 paper\n' +\
               '    \'tournier07\'  : SH basis from the Tournier et al.\n' +\
               '                      NeuroImage 2007 paper.'

    if mandatory:
        arg_name = 'sh_basis'
    else:
        arg_name = '--sh_basis'

    parser.add_argument(arg_name,
                        choices=choices, default=def_val,
                        help=help_msg)
