import pytest


@pytest.mark.parametrize("test_input", ["thesis", "work"])
def test_placeholder(test_input):
    assert isinstance(test_input, str)
