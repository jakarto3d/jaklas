import shutil
from pathlib import Path

import pytest

TEMP_DIR = Path(__file__).parent / "temp"


@pytest.fixture(autouse=True)
def cleanup_temp_dir():
    TEMP_DIR.mkdir(exist_ok=True)
    yield
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
