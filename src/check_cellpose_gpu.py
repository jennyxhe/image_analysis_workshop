"""Check whether Cellpose can use a GPU on this computer.

Cellpose runs on PyTorch ("torch").  PyTorch can use:
  - CUDA : NVIDIA graphics cards (Windows / Linux)
  - MPS  : the GPU built into Apple Silicon Macs (M1, M2, M3, ...)
If neither is available, everything runs on the CPU (works, but slower).

Run it from the top folder of the workshop repository with:

    uv run src/check_cellpose_gpu.py
"""

import platform
import time

import numpy as np
import torch
from cellpose import core, models

print(f"Computer: {platform.system()} {platform.machine()}, "
      f"Python {platform.python_version()}, torch {torch.__version__}")

# ---------------------------------------------------------------------------
# 1. What does PyTorch see?
# ---------------------------------------------------------------------------
print(f"NVIDIA GPU (CUDA) available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"Apple GPU (MPS) available:   {torch.backends.mps.is_available()}")

# ---------------------------------------------------------------------------
# 2. Does Cellpose agree?  use_gpu() actually tries to run a small
#    calculation on the GPU, so this is the real test.
# ---------------------------------------------------------------------------
gpu_ok = core.use_gpu()
print(f"Cellpose can use the GPU:    {gpu_ok}")

# ---------------------------------------------------------------------------
# 3. Speed test: segment the same random 512x512 image on CPU and on GPU.
#    The first run includes loading the model, so we run each one twice
#    and time only the second run.  (The first time ever, Cellpose
#    downloads its model from the internet.)
# ---------------------------------------------------------------------------
image = np.random.default_rng(0).random((512, 512)).astype(np.float32)
for use_gpu in [False, True] if gpu_ok else [False]:
    model = models.CellposeModel(gpu=use_gpu)
    model.eval(image)                       # warm-up
    start = time.perf_counter()
    model.eval(image)
    seconds = time.perf_counter() - start
    print(f"{'GPU' if use_gpu else 'CPU'}: {seconds:.1f} s for one 512x512 image "
          f"(device: {model.device})")

print("\nConclusion:", "use gpu=True in your Cellpose scripts." if gpu_ok
      else "no usable GPU - Cellpose will run on the CPU.")
