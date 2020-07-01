from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import laspy
import numpy as np

from . import point_formats


def write(
    point_data,
    output_path: Union[Path, str],
    *,
    point_format: int = None,
    precision: Tuple[float] = (0.0001, 0.0001, 0.0001),
    data_min_max: Optional[Dict[str, Tuple]] = None,
):
    """Write point cloud data to an output path.

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
        data_min_max (dict): Scale some dimensions according to these minimum and maximum
            values. Only these fields can be scaled: intensity, red, green, blue
            For example: the red channel is stored as uint16 inside a las file.
            If the red data in the source point_data is uint8, you can set
            data_min_max = {'red': (0, 255)}
            and the data will be scaled to the uint16 range 0-65536.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if data_min_max is None:
        data_min_max = {}

    if not point_format:
        point_format = point_formats.best_point_format(point_data)

    if point_format not in point_formats.supported_point_formats:
        raise ValueError(
            f"Unsupported point format {point_format} "
            f"(not in {point_formats.supported_point_formats})"
        )

    point_format_type = point_formats.point_formats[point_format]

    for coords in ["xyz", "XYZ"]:
        if all(c in point_data for c in coords):
            xyz = [point_data[c] for c in coords]
            break
    else:
        raise ValueError("Could not find xyz coordinates from input data.")

    with laspy.file.File(
        str(output_path),
        mode="w",
        header=laspy.header.Header(point_format=point_format),
    ) as f:
        f.header.offset = list(map(np.min, xyz))
        f.header.scale = precision

        f.x = xyz[0]
        f.y = xyz[1]
        f.z = xyz[2]

        if "gps_time" in point_format_type and "gps_time" in point_data:
            f.gps_time = point_data["gps_time"]

        if "intensity" in point_format_type and "intensity" in point_data:
            f.intensity = scale_data(
                "intensity", point_data["intensity"], data_min_max.get("intensity")
            )

        if "classification" in point_format_type and "classification" in point_data:
            f.classification = point_data["classification"]

        colors = ["red", "green", "blue"]
        if all(c in point_format_type and c in point_data for c in colors):
            for c in colors:
                setattr(f, c, scale_data(c, point_data[c], data_min_max.get(c)))


def scale_data(field_name, data, min_max):
    if min_max is None:
        return data

    dtype = point_formats.point_formats[3][field_name]

    max_value = 2 ** (np.dtype(dtype).itemsize * 8)

    offset = min_max[0] or 0
    scale = max_value / (min_max[1] - min_max[0] + 1)

    return (data - offset) * scale

