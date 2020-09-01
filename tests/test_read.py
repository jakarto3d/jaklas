from copy import deepcopy
from pathlib import Path
import laspy

import numpy as np
import pytest

from jaklas import read, read_header, read_pandas, write

TEST_DATA = Path(__file__).parent / "data"
TEMP_DIR = Path(__file__).parent / "temp"
very_small_las = TEST_DATA / "very_small.las"


def test_read_small():
    data = read(very_small_las)
    assert len(data["xyz"]) == 71


def test_read_with_offset():
    data1 = read(very_small_las)
    data2 = read(very_small_las, offset=(1, 1, 1))
    assert np.allclose(data1["xyz"], data2["xyz"] + 1)


def test_read_not_combined():
    data = read(very_small_las, combine_xyz=False)
    assert "x" in data and "y" in data and "z" in data
    assert "xyz" not in data


def test_read_other_dims():
    data = read(very_small_las, other_dims=["classification"])
    assert "gps_time" not in data and "intensity" not in data
    assert "classification" in data


def test_read_wrong_dim():
    with pytest.raises(KeyError):
        data = read(very_small_las, other_dims=["wrong"])


def test_read_pandas():
    df = read_pandas(very_small_las)
    assert len(df["x"]) == 71
    assert "x" in df and "y" in df and "z" in df
    assert "gps_time" in df and "intensity" in df


def test_read_dtype():
    data = read(very_small_las, xyz_dtype="f")
    assert data["xyz"].dtype == np.float32
    data = read(very_small_las)
    assert data["xyz"].dtype == np.float64


def test_read_all_fields():
    data = read(very_small_las)
    out = TEMP_DIR / "out.las"
    data["something_else"] = np.full(len(data["xyz"]), fill_value=1)
    write(data, out)
    data_out = read(out)
    assert sorted(list(data_out)) == [
        "classification",
        "flag_byte",
        "gps_time",
        "intensity",
        "pt_src_id",
        "scan_angle_rank",
        "something_else",
        "user_data",
        "xyz",
    ]


def test_read_offset():
    header = read_header(very_small_las)

    with laspy.file.File(very_small_las) as f:
        assert f.header.scale == header.scale
        assert f.header.offset == header.offset
        assert f.header.min == header.min
        assert f.header.max == header.max
