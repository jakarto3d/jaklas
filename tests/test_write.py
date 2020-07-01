from pathlib import Path


import laspy
import numpy as np
import pandas as pd
import pytest

import las_writer

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def xyz():
    return np.random.random((100, 3)) * 100


@pytest.fixture
def point_data_dict(xyz):
    return {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
    }


@pytest.fixture
def point_data_pandas(point_data_dict):
    return pd.DataFrame(point_data_dict)


@pytest.mark.parametrize("point_data", [point_data_dict, point_data_pandas])
def test_write_dict(point_data):
    output = DATA_DIR / "temp.las"
    las_writer.write(point_data, output)
    with laspy.file.File(output) as f:
        assert np.all_close(f.x == point_data["x"], e=0.0001)
        assert np.all_close(f.y == point_data["y"], e=0.0001)
        assert np.all_close(f.z == point_data["z"], e=0.0001)
