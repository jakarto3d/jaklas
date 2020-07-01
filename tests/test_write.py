from pathlib import Path


import laspy
import numpy as np
import pandas as pd
import pytest

import las_writer

DATA_DIR = Path(__file__).parent / "data"


xyz = np.random.random((100, 3)) * 100
gps_time = np.random.random(100) * 100
rgb = np.random.random((100, 3)) * 100

point_data = {
    "x": xyz[:, 0],
    "y": xyz[:, 1],
    "z": xyz[:, 2],
}

point_data_color = {
    **point_data,
    "red": rgb[:, 0],
    "green": rgb[:, 1],
    "blue": rgb[:, 2],
}

point_data_gps_time = {
    **point_data,
    "gps_time": gps_time,
}

point_data_gps_time_color = {
    **point_data,
    **point_data_color,
    **point_data_gps_time,
}

point_data_pandas = pd.DataFrame(point_data)


TEMP_OUTPUT = DATA_DIR / "temp.las"


@pytest.mark.parametrize("point_data", [point_data, point_data_pandas])
def test_write(point_data):
    las_writer.write(point_data, TEMP_OUTPUT)
    with laspy.file.File(TEMP_OUTPUT) as f:
        assert np.allclose(f.x, point_data["x"], atol=0.0001)
        assert np.allclose(f.y, point_data["y"], atol=0.0001)
        assert np.allclose(f.z, point_data["z"], atol=0.0001)


@pytest.mark.parametrize(
    "data,point_format",
    [
        (point_data, 0),
        (point_data_gps_time, 1),
        (point_data_color, 2),
        (point_data_gps_time_color, 3),
    ],
)
def test_write_point_format(data, point_format):
    assert las_writer.best_point_format(data) == point_format
