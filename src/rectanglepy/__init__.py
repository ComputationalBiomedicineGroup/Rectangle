from importlib.metadata import version

from . import pp, tl
from .parameters import RectangleAdvancedParameters
from .rectangle import load_tutorial_data, rectangle

__all__ = ["pp", "tl", "RectangleAdvancedParameters", "load_tutorial_data", "rectangle"]

__version__ = version("rectanglepy")
