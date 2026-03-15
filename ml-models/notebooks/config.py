# Shim — loads ml-models/config.py by absolute path and re-exports ALL symbols.
# Works regardless of CWD or how Jupyter was launched.
import importlib.util, sys, os
from pathlib import Path

# Absolute path to ml-models/ — resolved relative to THIS file, not CWD
_here    = Path(os.path.abspath(__file__))          # .../ml-models/notebooks/config.py
_ml_root = str(_here.parent.parent)                 # .../ml-models

# Ensure ml-models/ is at the FRONT of sys.path (do this before loading real config)
_paths = [os.path.normcase(os.path.normpath(p)) for p in sys.path]
_norm  = os.path.normcase(os.path.normpath(_ml_root))
if _norm not in _paths:
    sys.path.insert(0, _ml_root)
    print(f"[config shim] Added to sys.path: {_ml_root}")

# Load the real ml-models/config.py directly by file path
_real = os.path.join(_ml_root, "config.py")
_spec = importlib.util.spec_from_file_location("_ml_config", _real)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export every public name into this shim's namespace
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
