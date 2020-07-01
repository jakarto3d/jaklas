from pathlib import Path


import laspy
import numpy as np
import pandas as pd
import pytest

import las_writer

DATA_DIR = Path(__file__).parent / "data"


xyz = np.random.random((100, 3)) * 100

point_data_dict = {
    "x": xyz[:, 0],
    "y": xyz[:, 1],
    "z": xyz[:, 2],
}

point_data_pandas = pd.DataFrame(point_data_dict)


@pytest.mark.parametrize("point_data", [point_data_dict, point_data_pandas])
def test_write_dict(point_data):
    output = DATA_DIR / "temp.las"
    las_writer.write(point_data, output)
    with laspy.file.File(output) as f:
        assert np.allclose(f.x, point_data["x"], atol=0.0001)
        assert np.allclose(f.y, point_data["y"], atol=0.0001)
        assert np.allclose(f.z, point_data["z"], atol=0.0001)
