# Zeus NPU

This document describes how to install SGLang on Zeus. The Zeus stack follows
the same broad model as Ascend NPU: install the official PyTorch CPU wheel,
install the out-of-tree Zeus backend packages, then install SGLang from source
with a platform-specific pyproject.

## Installation

### Python Environment

Python 3.12 is recommended for the current dependency set.

```bash
conda create -n sglang_zeus python=3.12 -y
conda activate sglang_zeus
python -m pip install --upgrade pip setuptools wheel
```

### PyTorch and Zeus Backend

Install the PyTorch CPU stack first. `torch_zeus` is a PyTorch extension and
must be built or installed against the active torch ABI.

```bash
python -m pip install \
  torch==2.10.0+cpu \
  torchvision==0.25.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

Install the Zeus backend packages from source:

```bash
cd /path/to/torch_zeus
python -m pip install -e .

cd /path/to/torch_zeus/sgl-kernel-zeus/python
python -m pip install -e .
```

Or install prebuilt wheels without allowing pip to replace torch:

```bash
python -m pip install --no-deps /path/to/torch_zeus-*.whl
python -m pip install --no-deps /path/to/sgl_kernel_zeus-*.whl
```

### Install SGLang

Use the Zeus-specific pyproject, matching the source install style used by
other non-default hardware backends.

```bash
# git clone https://github.com/sgl-project/sglang.git
cd sglang
cp python/pyproject_zeus.toml python/pyproject.toml
python -m pip install -e python --no-build-isolation
```

After this step, `python/pyproject.toml` contains the Zeus-specific pyproject.
Restore the default pyproject before doing CUDA development.

## Verification

```bash
python - <<'PY'
import torch
import torchvision
import torch_zeus
import torch_zeus._C
import sgl_kernel_zeus
import sglang
import sglang.srt

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch.zeus.is_available:", torch.zeus.is_available())
print("zeus device count:", torch.zeus.device_count())
print("sglang:", sglang.__version__)
print("OK")
PY
```

Expected torch stack:

```text
torch: 2.10.0+cpu
torchvision: 0.25.0+cpu
torch.zeus.is_available: True
```
