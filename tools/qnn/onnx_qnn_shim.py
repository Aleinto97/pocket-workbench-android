"""Run a QAIRT Python entry point on a modern interpreter.

QAIRT 2.50 targets Python 3.12 and carries two assumptions that break on a
current environment:

1. It imports ``distutils``, removed in Python 3.12. ``setuptools`` ships a
   replacement, but only activates it through a ``.pth`` file, and ``.pth``
   files are not processed for plain ``PYTHONPATH`` entries (we install with
   ``pip --target``). Importing setuptools here installs the hook explicitly.
2. It reads ``onnx.version.version``, a module attribute removed in recent
   onnx releases. We re-expose it with the installed version so the SDK's
   ``version.parse(onnx.version.version) < version.parse("1.6.0")`` check works.

Usage: python onnx_qnn_shim.py <path-to-sdk-python-script> [args...]
"""
import runpy
import sys
import types

import setuptools  # noqa: F401  installs the distutils shim

import onnx

if not hasattr(onnx, "version"):
    onnx.version = types.SimpleNamespace(version=onnx.__version__)

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
