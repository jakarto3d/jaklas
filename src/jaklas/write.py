from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import laspy
import pyproj
from laspy.vlrs.known import WktCoordinateSystemVlr

from . import point_formats


def write(
    point_data,
    output_path: Union[Path, str],
    *,
    crs: Optional[int] = None,
    xyz_offset: Tuple[float] = None,
    point_format: Optional[int] = None,
    scale: Tuple[float] = None,
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
        xyz_offset (Tuple[float], optional): Apply this xyz offset before
            writing the coordinates. This can be useful for large coordinates
            loaded as float32 with an offset.
            The offset is applied by adding it to the coordinate.
            Defaults to (0, 0, 0).
        crs (int, optional): The EPSG code to write in the las header.
        point_format (int, optional): The las point format type identifier
            Only formats 0, 1, 2, 3, 6 and 7 are accepted.
            If None is given, the best point format will be guessed
            based on the provided fields.
        scale (Tuple[float], optional): The coordinate precision.
            Coordinates in the las file are stored in int32. This means that there is
            always a slight loss in precision. Most real world use cases are not
            affected if the scale is set correctly. So this should be set to the
            smallest error you can afford to have in the final file.
            Setting this to a number too small could lead to errors when using
            large coordinates.
            The default computes the scale based on the maximum range of a 32 bits
            signed integer.
        data_min_max (dict): Scale some dimensions according to these minimum and
            maximum values. Only these fields can be scaled: intensity, red, green,
            blue For example: the red channel is stored as uint16 inside a las file.
            If the red data in the source point_data is uint8, you can set
            data_min_max = {'red': (0, 255)}
            and the data will be scaled to the uint16 range 0-65536.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if data_min_max is None:
        data_min_max = {}

    standard_dimensions = point_formats.standard_dimensions | {"xyz", "XYZ"}
    extra_dimensions = sorted(set(point_data) - standard_dimensions)

    if not point_format:
        point_format = point_formats.best_point_format(point_data, extra_dimensions)

    if point_format not in point_formats.supported_point_formats:
        raise ValueError(
            f"Unsupported point format {point_format} "
            f"(not in {point_formats.supported_point_formats})"
        )

    point_format_type = point_formats.point_formats[point_format]

    xyz = None
    for coords in ["xyz", "XYZ"]:
        if coords in point_data:
            # expects an array of the shape (n_points, 3)
            xyz = [
                point_data[coords][:, 0],
                point_data[coords][:, 1],
                point_data[coords][:, 2],
            ]
            break
        if all(c in point_data for c in coords):
            xyz = [point_data[c] for c in coords]
            break

    if not xyz:
        raise ValueError("Could not find xyz coordinates from input data.")

    las = laspy.create(file_version="1.4", point_format=point_format)

    if crs is not None:
        wkt = pyproj.CRS.from_epsg(crs).to_wkt()
        las.vlrs.append(WktCoordinateSystemVlr(wkt))
        las.header.global_encoding.wkt = 1

    extra_bytes_params = [
        laspy.point.format.ExtraBytesParams(name=dim, type=point_data[dim].dtype)
        for dim in extra_dimensions
    ]
    las.add_extra_dims(extra_bytes_params)

    min_, max_, offset = _min_max_offset(xyz)

    offset = offset if xyz_offset is None else xyz_offset
    if xyz_offset is None:
        xyz_offset = (0, 0, 0)

    min_ += xyz_offset
    max_ += xyz_offset
    las.header.mins, las.header.maxs = min_, max_
    scales = scale if scale else _get_scale(min_, max_, offset)
    las.change_scaling(scales=scales, offsets=offset)

    las.x = xyz[0].astype("d") + xyz_offset[0]
    las.y = xyz[1].astype("d") + xyz_offset[1]
    las.z = xyz[2].astype("d") + xyz_offset[2]

    if "gps_time" in point_format_type and "gps_time" in point_data:
        las.gps_time = point_data["gps_time"]

    if "intensity" in point_format_type and "intensity" in point_data:
        las.intensity = scale_data(
            "intensity", point_data["intensity"], data_min_max.get("intensity")
        )

    if "classification" in point_format_type and "classification" in point_data:
        # convert pd.Series to numpy array, if applicable
        las.classification = np.array(point_data["classification"])

    colors = ["red", "green", "blue"]
    if all(c in point_format_type and c in point_data for c in colors):
        for c in colors:
            setattr(las, c, scale_data(c, point_data[c], data_min_max.get(c)))

    for dim in extra_dimensions:
        setattr(las, dim, point_data[dim])

    las.write(str(output_path))


def scale_data(field_name, data, min_max):
    """Scale data using the min_max bounds and the destination data type"""

    if min_max is None:
        return data

    dtype = point_formats.point_formats[3][field_name]

    min_value = np.iinfo(np.dtype(dtype)).min
    max_value = np.iinfo(np.dtype(dtype)).max

    offset = (min_max[0] - min_value) or 0

    scale = (max_value - min_value) / (min_max[1] - min_max[0])

    return (data - offset) * scale


def _min_max_offset(xyz: List[np.ndarray]) -> Tuple[Tuple[float]]:
    minimums = np.array(list(map(np.min, xyz)), "d")
    maximums = np.array(list(map(np.max, xyz)), "d")
    offset = np.mean([minimums, maximums], axis=0)
    return minimums, maximums, offset


def _get_scale(minimums, maximums, offset) -> Tuple[float]:
    max_long = np.iinfo(np.int32).max - 1

    offsetted_max_ranges = np.max(
        [np.abs(minimums - offset), np.abs(maximums - offset)], axis=0
    )

    scale = offsetted_max_ranges / max_long

    return tuple(scale)
