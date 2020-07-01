# Note some less often used attributes are omitted

_base = {
    "x": "i4",
    "y": "i4",
    "z": "i4",
    "intensity": "u2",
    "classification": "u1",
}
_gps_time = {"gps_time": "f8"}
_colors = {
    "red": "u2",
    "green": "u2",
    "blue": "u2",
}
point_formats = {
    0: {**_base},
    1: {**_base, **_gps_time},
    2: {**_base, **_colors},
    3: {**_base, **_gps_time, **_colors},
}


def best_point_format(data) -> int:
    """Returns the best point format depending on keys in the provided object.

    Args:
        data (dict-like): Any object that implements the __getitem__ method.
            So a dictionnary, a pandas DataFrame, or a numpy structured array will
            all work.

    Returns:
        int: The best point format. If none is matched, return 0 as a default.
    """
    possible_formats = [
        n for n, f in point_formats.items() if all(k in f for k in data)
    ]
    if not possible_formats:
        return 0
    return min(possible_formats, key=lambda n: len(point_formats[n]))
