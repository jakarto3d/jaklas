from typing import List

import numpy as np
from laspy.point.dims import COMPOSED_FIELDS_6, POINT_FORMAT_DIMENSIONS

# laspy standard attributes depending on the point format
_base_format_0 = {
    "X": "i4",
    "Y": "i4",
    "Z": "i4",
    "intensity": "u2",
    "scan_angle_rank": "u1",
    "user_data": "u1",
    "point_source_id": "u2",
    # composed field (bit_fields):
    "return_number": "u1",
    "number_of_returns": "u1",
    "scan_direction_flag": "B",
    "edge_of_flight_line": "B",
    # composed field (raw_classification):
    "classification": "u1",
    "synthetic": "B",
    "key_point": "B",
    "withheld": "B",
}
_gps_time = {"gps_time": "f8"}
_colors = {
    "red": "u2",
    "green": "u2",
    "blue": "u2",
}
_base_format_6 = {
    **_base_format_0,
    **_gps_time,
    "classification_flags": "u1",
    "scanner_channel": "u1",
    "scan_angle": "u2",
    "overlap": "B",
}
point_formats = {
    0: {**_base_format_0},
    1: {**_base_format_0, **_gps_time},
    2: {**_base_format_0, **_colors},
    3: {**_base_format_0, **_gps_time, **_colors},
    # for now, formats >= 6 are used mostly because of more
    # classification classes (256 vs 32) and classifications 64-255 are user definable
    6: {**_base_format_6},
    7: {**_base_format_6, **_colors},
}

supported_point_formats = list(point_formats)

# everything that is not an extra dimension
standard_dimensions = {
    name for format in POINT_FORMAT_DIMENSIONS.values() for name in format
} | {"x", "y", "z"}
# substitute composed fields
for composed_field, values in COMPOSED_FIELDS_6.items():
    for subfield in values:
        standard_dimensions.add(subfield.name)
    standard_dimensions.remove(composed_field)


def best_point_format(
    data, extra_dimensions: List[str] = None, default_format=6
) -> int:
    """Returns the best point format depending on keys in the provided object.

    Args:
        data (dict-like): Any object that implements the __getitem__ method.
            So a dictionnary, a pandas DataFrame, or a numpy structured array will
            all work.

    Returns:
        int: The best point format. If none is matched, return 0 as a default.
    """
    if extra_dimensions is None:
        extra_dimensions = []
    min_point_format = 0
    if "classification" in data and np.any(data["classification"] >= 2 ** 5):
        # if there are more than 32 classes, we must use point formats >= 6
        min_point_format = 6

    def data_conforms_to_format(data, format_):
        data_fields = [
            f for f in data if f not in extra_dimensions + ["xyz", "XYZ", "x", "y", "z"]
        ]
        return all(k in format_ for k in data_fields)

    possible_formats = [
        n
        for n, format_ in point_formats.items()
        if data_conforms_to_format(data, format_) and n >= min_point_format
    ]

    if not possible_formats:
        return default_format

    return min(possible_formats, key=lambda n: len(point_formats[n]))
