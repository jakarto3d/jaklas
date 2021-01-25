from typing import Dict

import pylas
import numpy as np

from .header import Header


def read(
    path,
    *,
    offset=None,
    combine_xyz=True,
    xyz_dtype=np.float64,
    other_dims=None,
    ignore_missing_dims=False,
) -> Dict:
    if offset is None:
        offset = np.array([0, 0, 0])

    data = {}
    
    las = pylas.read(str(path))

    x = (las.x - offset[0]).astype(xyz_dtype)
    y = (las.y - offset[1]).astype(xyz_dtype)
    z = (las.z - offset[2]).astype(xyz_dtype)

    if combine_xyz:
        data["xyz"] = np.hstack([x[np.newaxis].T, y[np.newaxis].T,z[np.newaxis].T])
    else:
        data["x"] = x
        data["y"] = y
        data["z"] = z

    del x, y, z

    if other_dims is None:
        other_dims = set(las.point_format.dimension_names) - set("XYZ")

    for dim in other_dims:
        if not ignore_missing_dims and not hasattr(las, dim):
            raise KeyError(f"Dimension not found in file {dim}")

        data[dim] = getattr(las, dim)

    return data


def read_header(path) -> Header:
    """Read some header information from las file.

    The main use case of this function if when you have a large list
    of las files, and you want to quickly scan bounding boxes.

    Should be roughly 50x faster than laspy,
    and close to 10x than the master branch of pylas.
    """
    with open(path, "rb") as f:
        return Header(f)


def read_pandas(path, *, offset=None, xyz_dtype="d", other_dims=None):
    import pandas as pd

    return pd.DataFrame(
        read(
            path,
            offset=offset,
            combine_xyz=False,
            xyz_dtype=xyz_dtype,
            other_dims=other_dims,
        )
    )
