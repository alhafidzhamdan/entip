import pytest

from entip_train.slides import synthetic_slide


@pytest.fixture(scope="session")
def slide():
    return synthetic_slide("syn-a", seed=0)
