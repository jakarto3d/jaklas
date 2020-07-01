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

    The data is not scaled in any way.

    For example: the red channel is stored as uint16 inside a las file.
    If the red data in the source point_data is uint8, it will be casted as-is to
    uint16. So a value of 25 in the red channel in uint8 will be converted to 25 in
    uint16, which is twice as dark.

    Look in point_formats.py to see the type of each destination field.

    Args:
        point_data (dict-like): Any object that implements the __getitem__ method.
            So a dictionnary, a pandas DataFrame, or a numpy structured array will
            all work.
        output_path (Union[Path, str]): The output path to write the las file.
            The output directory is created if it doesn't exist.
        point_format (int, optional): The las point format type identifier 
            Only formats 0, 1, 2, 3, 6 and 7 are accepted.
            If None is given, the best point format will be guessed
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
        point_format = point_formats.best_point_format(point_data)

    if point_format not in point_formats.supported_point_formats:
        raise ValueError(
            f"Unsupported point format {point_format} "
            f"(not in {point_formats.supported_point_formats})"
        )

    point_format_type = point_formats.point_formats[point_format]

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

        if "gps_time" in point_format_type and "gps_time" in point_data:
            f.gps_time = point_data["gps_time"]

        if "intensity" in point_format_type and "intensity" in point_data:
            f.intensity = point_data["intensity"]

        if "classification" in point_format_type and "classification" in point_data:
            f.classification = point_data["classification"]

        colors = ["red", "green", "blue"]
        if all(c in point_format_type and c in point_data for c in colors):
            f.red = point_data["red"]
            f.green = point_data["green"]
            f.blue = point_data["blue"]
