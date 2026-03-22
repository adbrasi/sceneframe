"""Dataset Maker - CLI to create labeled video datasets from anime/series episodes."""

import ctypes
import glob
import importlib
import os


def _setup_nvidia_libs():
    """Preload pip-installed NVIDIA CUDA libraries so TensorFlow finds the GPU."""
    if os.environ.get("_DATASET_MAKER_CUDA_SETUP"):
        return

    nvidia_packages = [
        "nvidia.cuda_runtime.lib",
        "nvidia.cudnn.lib",
        "nvidia.cublas.lib",
        "nvidia.cufft.lib",
        "nvidia.curand.lib",
        "nvidia.cusolver.lib",
        "nvidia.cusparse.lib",
        "nvidia.nccl.lib",
        "nvidia.nvjitlink.lib",
    ]

    for pkg in nvidia_packages:
        try:
            mod = importlib.import_module(pkg)
            for so in glob.glob(os.path.join(mod.__path__[0], "*.so*")):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        except ImportError:
            pass

    os.environ["_DATASET_MAKER_CUDA_SETUP"] = "1"


_setup_nvidia_libs()
