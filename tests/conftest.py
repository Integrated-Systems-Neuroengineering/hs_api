import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--shuffle", 
        action="store_true", 
        default=False, 
        help="Enable random shuffling for network mapping"
    )

@pytest.fixture
def shuffle_mode(request):
    return request.config.getoption("--shuffle")