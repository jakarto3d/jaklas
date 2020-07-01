from pathlib import Path


import laspy
import numpy as np
import pandas as pd
import pytest

import las_writer

DATA_DIR = Path(__file__).parent / "data"


xyz = np.random.random((100, 3)) * 100
intensity = np.random.random(100) * 100
classification = (np.random.random(100) * 31).astype("u1")
classification_large = (np.random.random(100) * 255).astype("u1")
gps_time = np.random.random(100) * 100
rgb = np.random.random((100, 3)) * 100

point_data = {
    "x": xyz[:, 0],
    "y": xyz[:, 1],
    "z": xyz[:, 2],
    "intensity": intensity,
    "classification": classification,
}

point_data_large_classification = {
    "x": xyz[:, 0],
    "y": xyz[:, 1],
    "z": xyz[:, 2],
    "intensity": intensity,
    "classification": classification_large,
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
point_data_color_pandas = pd.DataFrame(point_data_color)
point_data_gps_time_pandas = pd.DataFrame(point_data_gps_time)
point_data_gps_time_color_pandas = pd.DataFrame(point_data_gps_time_color)


TEMP_OUTPUT = DATA_DIR / "temp.las"


@pytest.mark.parametrize("data", [point_data, point_data_pandas])
def test_write_simple(data):
    las_writer.write(data, TEMP_OUTPUT)
    with laspy.file.File(TEMP_OUTPUT) as f:
        assert np.allclose(f.x, data["x"], atol=0.0001)
        assert np.allclose(f.y, data["y"], atol=0.0001)
        assert np.allclose(f.z, data["z"], atol=0.0001)
        assert np.allclose(f.intensity, data["intensity"].astype("u2"))
        assert np.allclose(f.classification, data["classification"])


@pytest.mark.parametrize("data", [point_data_gps_time, point_data_gps_time_pandas])
def test_write_gps_time(data):
    las_writer.write(data, TEMP_OUTPUT)
    with laspy.file.File(TEMP_OUTPUT) as f:
        assert np.allclose(f.gps_time, data["gps_time"], atol=0.0001)


@pytest.mark.parametrize("data", [point_data_color, point_data_color_pandas])
def test_write_color(data):
    las_writer.write(data, TEMP_OUTPUT)
    with laspy.file.File(TEMP_OUTPUT) as f:
        assert np.allclose(f.red, data["red"].astype("u2"))
        assert np.allclose(f.green, data["green"].astype("u2"))
        assert np.allclose(f.blue, data["blue"].astype("u2"))


@pytest.mark.parametrize(
    "data,point_format",
    [
        (point_data, 0),
        (point_data_gps_time, 1),
        (point_data_color, 2),
        (point_data_gps_time_color, 3),
        (point_data_large_classification, 6),
        (point_data_pandas, 0),
        (point_data_gps_time_pandas, 1),
        (point_data_color_pandas, 2),
        (point_data_gps_time_color_pandas, 3),
    ],
)
def test_write_point_format(data, point_format):
    assert las_writer.best_point_format(data) == point_format


def test_write_large_classifications():
    data = point_data_large_classification
    las_writer.write(data, TEMP_OUTPUT)
    with laspy.file.File(TEMP_OUTPUT) as f:
        assert f.point_format.fmt == "6"
        assert np.allclose(f.x, data["x"], atol=0.0001)
        assert np.allclose(f.y, data["y"], atol=0.0001)
        assert np.allclose(f.z, data["z"], atol=0.0001)
        assert np.allclose(f.intensity, data["intensity"].astype("u2"))
        assert np.allclose(f.classification, data["classification"])

