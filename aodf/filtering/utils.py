from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
import numpy as np


def get_sf_range(data, sh_order, full_basis, sphere_name):
    sphere = get_sphere(sphere_name)
    sf = np.array([sh_to_sf(sh, sphere, sh_order, full_basis=full_basis)
                   for sh in data], dtype=np.float32)
    vrange = np.max(sf) - np.min(sf)
    return vrange


def parse_sh_basis(sh_basis_name):
    basis = 'descoteaux07' if 'descoteaux07' in sh_basis_name else 'tournier07'
    legacy = 'legacy' in sh_basis_name
    return basis, legacy


def add_sh_basis_arg(parser):
    parser.add_argument('--sh_basis', default='descoteaux07_legacy',
                        choices=['tournier07', 'descoteaux07',
                                 'tournier07_legacy', 'descoteaux07_legacy'],
                        help='SH basis used for signal representation.'
                             ' [%(default)s]')
    return parser
