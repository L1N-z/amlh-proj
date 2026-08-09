import pytest

from amlh.data import SplitResult, load_test, load_train, make_validation_split


@pytest.fixture(scope="session")
def train():
    return load_train()


@pytest.fixture(scope="session")
def test():
    return load_test()


@pytest.fixture(scope="session")
def split(train) -> SplitResult:
    return make_validation_split(train)
