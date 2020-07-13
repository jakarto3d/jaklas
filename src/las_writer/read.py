from typing import Dict

import laspy
import numpy as np


def read(
    path,
    *,
    offset=None,
    combine_xyz=True,
    xyz_dtype=np.float64,
    other_dims=["intensity", "gps_time"],
) -> Dict:
    if offset is None:
        offset = np.array([0, 0, 0])

    data = {}
    with laspy.file.File(path) as f:
        x = (f.x - offset[0]).astype(xyz_dtype)
        y = (f.y - offset[1]).astype(xyz_dtype)
        z = (f.z - offset[2]).astype(xyz_dtype)
        if combine_xyz:
            xyz = np.empty((f.header.count, 3), xyz_dtype)
            xyz[:, 0] = x
            xyz[:, 1] = y
            xyz[:, 2] = z
            data["xyz"] = xyz
            del x, y, z
        else:
            data["x"] = x
            data["y"] = y
            data["z"] = z

        for dim in other_dims:
            if not hasattr(f, dim):
                raise KeyError(f"Dimension not found in file {dim}")

            # np.array is required, because f contains a memory view and will close
            data[dim] = np.array(getattr(f, dim))

    return data


def read_pandas(
    path, *, offset=None, xyz_dtype="d", other_dims=["intensity", "gps_time"]
):
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
