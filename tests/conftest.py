import shutil
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def cleanup_data_dir():
    DATA_DIR.mkdir(exist_ok=True)
    yield
    shutil.rmtree(DATA_DIR, ignore_errors=True)
