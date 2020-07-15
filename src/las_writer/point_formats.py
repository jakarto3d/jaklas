import numpy as np

from typing import List

# Note some less often used attributes are omitted
_base = {
    "x": "i4",
    "y": "i4",
    "z": "i4",
    "intensity": "u2",
    "classification": "u1",
    "return_num": "u1",
    "num_returns": "u1",
    "edge_flight_line": "B",
    "scan_dir_flag": "B",
    "scan_angle_rank": "u1",
    "user_data": "u1",
    "pt_src_id": "u2",
}
_gps_time = {"gps_time": "f8"}
_colors = {
    "red": "u2",
    "green": "u2",
    "blue": "u2",
}
_base_6 = {
    **_base,
    "scan_angle": "u2",
}
point_formats = {
    0: {**_base},
    1: {**_base, **_gps_time},
    2: {**_base, **_colors},
    3: {**_base, **_gps_time, **_colors},
    # formats >= 6 are used mostly because of more classification classes (256 vs 32)
    # and classifications 64-255 are user definable
    6: {**_base_6, **_gps_time},
    7: {**_base_6, **_gps_time, **_colors},
}

supported_point_formats = list(point_formats)

# everything that is not an extra dimension
standard_dimensions = {
    "x",
    "y",
    "z",
    "X",
    "Y",
    "Z",
    "intensity",
    "gps_time",
    "classification",
    "red",
    "green",
    "blue",
    "flag_byte",
    "return_num",
    "num_returns",
    "scan_dir_flag",
    "edge_flight_line",
    "scanner_channel",
    "synthetic",
    "key_point",
    "withheld",
    "overlap",
    "classification_flags",
    "classification_byte",
    "raw_classification",
    "scan_angle_rank",
    "scan_angle",
    "user_data",
    "pt_src_id",
    "nir",
    "wave_packet_desc_index",
    "byte_offset_to_waveform_data",
    "waveform_packet_size",
    "return_point_waveform_loc",
    "x_t",
    "y_t",
    "z_t",
}

allowed_standard_dimension_names = ["xyz", "XYZ", "X", "Y", "Z"]


def best_point_format(data, extra_dimensions: List[str] = None) -> int:
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
            f
            for f in data
            if f not in extra_dimensions + allowed_standard_dimension_names
        ]
        return all(k in format_ for k in data_fields)

    possible_formats = [
        n
        for n, format_ in point_formats.items()
        if data_conforms_to_format(data, format_) and n >= min_point_format
    ]

    if not possible_formats:
        return min_point_format

    return min(possible_formats, key=lambda n: len(point_formats[n]))
