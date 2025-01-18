from importlib.metadata import version

from . import pp, tl
from .rectangle import load_tutorial_data, rectangle

__all__ = ["pp", "tl", "load_tutorial_data", "rectangle"]

__version__ = version("rectanglepy")
