import sys, types, runpy
import onnx

if not hasattr(onnx, "version"):
    onnx.version = types.SimpleNamespace(version=onnx.__version__)

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
