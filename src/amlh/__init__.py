"""AMLH patient question classification coursework package."""

from importlib import import_module

__all__ = ["arm3_llm"]


def __getattr__(name: str):
	if name in __all__:
		return import_module(f"{__name__}.{name}")
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
	return sorted(list(globals().keys()) + __all__)
