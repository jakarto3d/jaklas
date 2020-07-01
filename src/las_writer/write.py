from pathlib import Path
from typing import Union

import numpy as np
import laspy


def write(
    point_data, output_path: Union[Path, str], precision=(0.0001, 0.0001, 0.0001)
):
    """Write point cloud data to an output path.

    Args:
        point_data (dict-like): Any object that implements the __getitem__ method.
            So a dictionnary, a pandas DataFrame, or a numpy structured array will
            all work.
        output_path (Union[Path, str]): The output path to write the las file.
            If the output directory doesn't exist, it's created.
        precision (tuple, optional): The coordinate precision.
            Coordinates in the las file are stored in int32. This means that there is
            always a slight loss in precision. Most real world use cases are not
            affected if the scale is set correctly. So this should be set to the
            smallest error you can afford to have in the final file.
            Setting this to a number too small could lead to errors when using
            large coordinates.
            Defaults to (0.0001, 0.0001, 0.0001).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with laspy.file.File(str(output_path), mode="w", header=laspy.header.Header()) as f:
        f.header.offset = [
            np.min(point_data["x"]),
            np.min(point_data["y"]),
            np.min(point_data["z"]),
        ]
        f.header.scale = precision

        f.x = point_data["x"]
        f.y = point_data["y"]
        f.z = point_data["z"]
