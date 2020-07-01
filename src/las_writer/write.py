from pathlib import Path
from typing import Union, Tuple

import numpy as np
import laspy

from . import point_formats


def write(
    point_data,
    output_path: Union[Path, str],
    point_format: int = None,
    precision: Tuple[float] = (0.0001, 0.0001, 0.0001),
):
    """Write point cloud data to an output path.

    Args:
        point_data (dict-like): Any object that implements the __getitem__ method.
            So a dictionnary, a pandas DataFrame, or a numpy structured array will
            all work.
        output_path (Union[Path, str]): The output path to write the las file.
            The output directory is created if it doesn't exist.
        point_format (int, optional): The las point format type (only formats 0 to 3
            are accepted). If None is given, the best point format will be guessed
            based on the provided fields.
        precision (Tuple[float], optional): The coordinate precision.
            Coordinates in the las file are stored in int32. This means that there is
            always a slight loss in precision. Most real world use cases are not
            affected if the scale is set correctly. So this should be set to the
            smallest error you can afford to have in the final file.
            Setting this to a number too small could lead to errors when using
            large coordinates.
            Defaults to (0.0001, 0.0001, 0.0001).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not point_format:
        point_format = point_formats.guess_best_format(point_data)

    with laspy.file.File(
        str(output_path),
        mode="w",
        header=laspy.header.Header(point_format=point_format),
    ) as f:
        f.header.offset = [
            np.min(point_data["x"]),
            np.min(point_data["y"]),
            np.min(point_data["z"]),
        ]
        f.header.scale = precision

        f.x = point_data["x"]
        f.y = point_data["y"]
        f.z = point_data["z"]
