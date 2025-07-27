# numpy_core_shim.py
import sys, types
import numpy as np

# Build a dummy numpy._core module that forwards to numpy.core
core = types.ModuleType("numpy._core")
core.multiarray = np.core.multiarray
core.numeric    = np.core.numeric
core.multiarray._reconstruct = np.core.multiarray._reconstruct
sys.modules["numpy._core"] = core
sys.modules["numpy._core.numeric"] = core.numeric
sys.modules["numpy._core.multiarray"] = core.multiarray
