import os

from ants_platform import AntsPlatform
from ants_platform.logger import ants_platform_logger

"""
Level	Numeric value
logging.DEBUG	10
logging.INFO	20
logging.WARNING	30
logging.ERROR	40
"""


def test_default_ants_platform():
    AntsPlatform()

    assert ants_platform_logger.level == 30


def test_via_env():
    os.environ["ANTS_PLATFORM_DEBUG"] = "True"

    AntsPlatform()

    assert ants_platform_logger.level == 10

    os.environ.pop("ANTS_PLATFORM_DEBUG")


def test_debug_ants_platform():
    AntsPlatform(debug=True)
    assert ants_platform_logger.level == 10

    # Reset
    ants_platform_logger.setLevel("WARNING")
