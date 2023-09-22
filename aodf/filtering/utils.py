from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
import numpy as np


def get_sf_range(data, sh_order, full_basis, sphere_name):
    sphere = get_sphere(sphere_name)
    sf = np.array([sh_to_sf(sh, sphere, sh_order, full_basis=full_basis)
                   for sh in data], dtype=np.float32)
    vrange = np.max(sf) - np.min(sf)
    return vrange
