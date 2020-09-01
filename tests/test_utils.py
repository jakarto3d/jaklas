from copy import deepcopy

import pytest
import numpy as np
import pandas as pd

from jaklas import utils

xyz = np.random.random((100, 3)) * 100
intensity = np.random.random(100) * 100
classification = (np.random.random(100) * 31).astype("u1")
gps_time = np.random.random(100) * 100


point_data = {
    "x": xyz[:, 0],
    "y": xyz[:, 1],
    "z": xyz[:, 2],
    "classification": classification,
    "gps_time": gps_time,
}

point_data_pandas = pd.DataFrame(point_data)


@pytest.mark.parametrize("data", [point_data, point_data_pandas])
def test_sort(data):
    data = deepcopy(data)
    sorted_data = utils.sort(data, "gps_time")
    assert not np.allclose(data["x"], sorted_data["x"])
    assert np.allclose(np.sort(data["gps_time"]), sorted_data["gps_time"])
