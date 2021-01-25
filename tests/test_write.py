from copy import deepcopy
from pathlib import Path

import jaklas
import numpy as np
import pandas as pd
import pylas
import pytest

TEMP_DIR = Path(__file__).parent / "temp"


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


TEMP_OUTPUT = TEMP_DIR / "temp.las"
TEMP_OUTPUT_LAZ = TEMP_DIR / "temp.laz"


@pytest.mark.parametrize("data", [point_data, point_data_pandas])
def test_write_simple(data):
    jaklas.write(data, TEMP_OUTPUT)
    f = pylas.read(str(TEMP_OUTPUT))
    assert np.allclose(f.x, data["x"], atol=0.0001)
    assert np.allclose(f.y, data["y"], atol=0.0001)
    assert np.allclose(f.z, data["z"], atol=0.0001)
    assert np.allclose(f.intensity, data["intensity"].astype("u2"))
    assert np.allclose(f.classification, data["classification"])


@pytest.mark.parametrize("data", [point_data, point_data_pandas])
def test_write_simple_laz(data):
    jaklas.write(data, TEMP_OUTPUT_LAZ)
    f = pylas.read(str(TEMP_OUTPUT_LAZ))
    assert f.header.are_points_compressed
    assert np.allclose(f.x, data["x"], atol=0.0001)
    assert np.allclose(f.y, data["y"], atol=0.0001)
    assert np.allclose(f.z, data["z"], atol=0.0001)
    assert np.allclose(f.intensity, data["intensity"].astype("u2"))
    assert np.allclose(f.classification, data["classification"])


@pytest.mark.parametrize("data", [point_data_gps_time, point_data_gps_time_pandas])
def test_write_X_Y_Z(data):
    data = deepcopy(data)
    data["X"] = data["x"]
    data["Y"] = data["y"]
    data["Z"] = data["z"]
    del data["x"]
    del data["y"]
    del data["z"]
    jaklas.write(data, TEMP_OUTPUT)
    f = pylas.read(str(TEMP_OUTPUT))
    assert np.allclose(f.x, data["X"], atol=0.0001)
    assert np.allclose(f.y, data["Y"], atol=0.0001)
    assert np.allclose(f.z, data["Z"], atol=0.0001)
    assert np.allclose(f.intensity, data["intensity"].astype("u2"))
    assert np.allclose(f.classification, data["classification"])
    assert np.allclose(f.gps_time, data["gps_time"])


@pytest.mark.parametrize("data", [point_data, point_data_pandas])
def test_write_offset(data):
    xyz_offset = (1, 2, 3)
    jaklas.write(data, TEMP_OUTPUT, xyz_offset=xyz_offset)
    f = pylas.read(str(TEMP_OUTPUT))
    assert np.allclose(f.x, data["x"] + xyz_offset[0], atol=0.0001)
    assert np.allclose(f.y, data["y"] + xyz_offset[1], atol=0.0001)
    assert np.allclose(f.z, data["z"] + xyz_offset[2], atol=0.0001)
    assert np.allclose(f.intensity, data["intensity"].astype("u2"))
    assert np.allclose(f.classification, data["classification"])


@pytest.mark.parametrize("data", [point_data, point_data_pandas])
def test_write_offset_large_coordinates(data):
    data = deepcopy(data)
    xyz_offset = (3e5, 5e6, 100)
    data["x"] = data["x"].astype("f")
    data["y"] = data["y"].astype("f")
    data["z"] = data["z"].astype("f")
    jaklas.write(data, TEMP_OUTPUT, xyz_offset=xyz_offset)
    f = pylas.read(str(TEMP_OUTPUT))
    assert np.allclose(f.x, data["x"].astype("d") + xyz_offset[0], atol=0.0001)
    assert np.allclose(f.y, data["y"].astype("d") + xyz_offset[1], atol=0.0001)
    assert np.allclose(f.z, data["z"].astype("d") + xyz_offset[2], atol=0.0001)
    assert np.allclose(f.intensity, data["intensity"].astype("u2"))
    assert np.allclose(f.classification, data["classification"])


@pytest.mark.parametrize("data", [point_data])
def test_write_xyz(data):
    data = deepcopy(data)
    data["xyz"] = np.vstack([data["x"], data["y"], data["z"]]).T
    del data["x"]
    del data["y"]
    del data["z"]
    jaklas.write(data, TEMP_OUTPUT)
    f = pylas.read(str(TEMP_OUTPUT))
    assert np.allclose(f.x, data["xyz"][:, 0], atol=0.0001)
    assert np.allclose(f.y, data["xyz"][:, 1], atol=0.0001)
    assert np.allclose(f.z, data["xyz"][:, 2], atol=0.0001)
    assert np.allclose(f.intensity, data["intensity"].astype("u2"))
    assert np.allclose(f.classification, data["classification"])


@pytest.mark.parametrize("data", [point_data_gps_time])
def test_write_xyz_with_gps_time(data):
    data = deepcopy(data)
    data["xyz"] = np.vstack([data["x"], data["y"], data["z"]]).T
    del data["x"]
    del data["y"]
    del data["z"]
    jaklas.write(data, TEMP_OUTPUT)
    f = pylas.read(str(TEMP_OUTPUT))
    assert np.allclose(f.x, data["xyz"][:, 0], atol=0.0001)
    assert np.allclose(f.y, data["xyz"][:, 1], atol=0.0001)
    assert np.allclose(f.z, data["xyz"][:, 2], atol=0.0001)
    assert np.allclose(f.intensity, data["intensity"].astype("u2"))
    assert np.allclose(f.classification, data["classification"])
    assert np.allclose(f.gps_time, data["gps_time"])


@pytest.mark.parametrize("data", [point_data_gps_time, point_data_gps_time_pandas])
def test_write_gps_time(data):
    jaklas.write(data, TEMP_OUTPUT)
    f = pylas.read(str(TEMP_OUTPUT))
    assert np.allclose(f.gps_time, data["gps_time"], atol=0.0001)


@pytest.mark.parametrize("data", [point_data_color, point_data_color_pandas])
def test_write_color(data):
    jaklas.write(data, TEMP_OUTPUT)
    f = pylas.read(str(TEMP_OUTPUT))
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
    assert jaklas.best_point_format(data) == point_format


def test_write_large_classifications():
    data = point_data_large_classification
    jaklas.write(data, TEMP_OUTPUT)
    f = pylas.read(str(TEMP_OUTPUT))
    assert f.point_format.id == 6
    assert np.allclose(f.x, data["x"], atol=0.0001)
    assert np.allclose(f.y, data["y"], atol=0.0001)
    assert np.allclose(f.z, data["z"], atol=0.0001)
    assert np.allclose(f.intensity, data["intensity"].astype("u2"))
    assert np.allclose(f.classification, data["classification"])


def test_write_scaled():
    data = point_data_color
    data_min_max = {
        "intensity": (0, 255),
        "red": (0, 255),
        "green": (0, 255),
        "blue": (0, 255),
    }

    def u1_to_u2(data):
        return data * (2 ** 8 + 1)

    jaklas.write(data, TEMP_OUTPUT, data_min_max=data_min_max)
    f = pylas.read(str(TEMP_OUTPUT))
    assert np.allclose(f.intensity, u1_to_u2(data["intensity"]).astype("u2"))
    assert np.allclose(f.red, u1_to_u2(data["red"]).astype("u2"))
    assert np.allclose(f.green, u1_to_u2(data["green"]).astype("u2"))
    assert np.allclose(f.blue, u1_to_u2(data["blue"]).astype("u2"))


@pytest.mark.parametrize("data", [point_data, point_data_pandas])
def test_write_extra_dimensions(data):
    data["new_stuff"] = (np.random.random(100) * 100).astype("u1")
    jaklas.write(data, TEMP_OUTPUT)
    f = pylas.read(str(TEMP_OUTPUT))
    assert np.allclose(f.new_stuff, data["new_stuff"])
    assert f.new_stuff.dtype == np.dtype("u1")


@pytest.mark.parametrize("data", [point_data_gps_time, point_data_gps_time_pandas])
def test_write_extra_dimensions_gps_time(data):
    data["new_stuff"] = (np.random.random(100) * 100).astype("u1")
    jaklas.write(data, TEMP_OUTPUT)
    f = pylas.read(str(TEMP_OUTPUT))
    assert np.allclose(f.new_stuff, data["new_stuff"])
    assert np.allclose(f.gps_time, data["gps_time"])
    assert f.new_stuff.dtype == np.dtype("u1")
