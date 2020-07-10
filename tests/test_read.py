from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from las_writer import read, read_pandas

TEST_DATA = Path(__file__).parent / "data"
very_small_las = TEST_DATA / "very_small.las"


def test_read_small():
    data = read(very_small_las, None)
    assert len(data["xyz"]) == 71


def test_read_offset():
    data1 = read(very_small_las, None)
    data2 = read(very_small_las, (1, 1, 1))
    assert np.allclose(data1["xyz"], data2["xyz"] + 1)


def test_read_not_combined():
    data = read(very_small_las, None, combine_xyz=False)
    assert "x" in data and "y" in data and "z" in data
    assert "xyz" not in data


def test_read_other_dims():
    data = read(very_small_las, None, other_dims=["classification"])
    assert "gps_time" not in data and "intensity" not in data
    assert "classification" in data


def test_read_wrong_dim():
    with pytest.raises(KeyError):
        data = read(very_small_las, None, other_dims=["wrong"])


def test_read_pandas():
    df = read_pandas(very_small_las, None)
    assert len(df["x"]) == 71
    assert "x" in df and "y" in df and "z" in df
    assert "gps_time" in df and "intensity" in df
