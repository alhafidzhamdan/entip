import pytest

from entip_train.encoders import HandcraftedEncoder
from entip_train.service import TrainingService
from entip_train.slides import synthetic_slide
from entip_train.store import Store


@pytest.fixture(scope="session")
def slide():
    return synthetic_slide("syn-a", seed=0)


@pytest.fixture(scope="session")
def slide_b():
    return synthetic_slide("syn-b", seed=1)


@pytest.fixture
def svc(tmp_path, slide, slide_b):
    s = TrainingService(Store(tmp_path / "t.sqlite"), HandcraftedEncoder(), tmp_path / "art")
    for sl in (slide, slide_b):
        s.register_scan(sl.scan, sl.reader, sl.cells, sl.regions)
    return s
