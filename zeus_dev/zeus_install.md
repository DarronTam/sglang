# SGLang on Zeus — Install & Launch Guide

> **Current install path**: Zeus now follows the standard third-party stack:
> official PyTorch CPU wheel + `torch_zeus` backend adaptor +
> `sgl_kernel_zeus` kernels + `python/pyproject_zeus.toml`.
>
> The old `.zeus_editable` flow is legacy. Use:
>
> ```bash
> python -m pip install torch==2.9.1+cpu torchvision==0.24.1+cpu \
>   --index-url https://download.pytorch.org/whl/cpu
> python -m pip install --no-deps /path/to/torch_zeus-*.whl
> python -m pip install --no-deps /path/to/sgl_kernel_zeus-*.whl
> python zeus_dev/check_zeus_env.py
> cp python/pyproject_zeus.toml python/pyproject.toml
> python -m pip install -e python --no-build-isolation
> ```
>
> For local development, you can run the helper wrapper after installing the
> PyTorch and Zeus backend packages:
>
> ```bash
> bash zeus_dev/setup_zeus_dev.sh
> ```

This document is the **pure-Zeus (no CUDA)** install path for SGLang. It is the
Zeus counterpart of the standard third-party flow: official PyTorch CPU wheel first,
hardware backend adaptor second, SGLang source install last.

The flow below targets SGLang `0.5.6.post2` on Python 3.12. Use Python 3.12
for the main install path because all pinned runtime dependencies have matching
wheels for it.

> **Python 3.13 note**: upstream SGLang declares Python `>=3.10`, so Python
> 3.13 is allowed by SGLang's own package metadata. The current install failure
> on Python 3.13 comes from a transitive pinned dependency:
> `runtime_common` pins `outlines==0.1.11`, which pins
> `outlines_core==0.1.26`. PyPI publishes `outlines_core==0.1.26` wheels for
> Python 3.10-3.12, but not for Python 3.13. On Python 3.13, pip falls back to
> the sdist and can fail with inconsistent `outlines_core` metadata.

> **Pinned versions known to work together (2026-04-28)**
> | Package | Version |
> | --- | --- |
> | python | `3.12.x` |
> | torch | `2.9.1+cpu` (PyTorch CPU index) |
> | torchvision | `0.24.1+cpu` (PyTorch CPU index) |
> | triton | `3.6.0` (PyPI, OK for documented launch flags) |
> | sglang | `0.5.6.post2` (editable, `pyproject_zeus.toml`) |
> | torch_zeus + sgl_kernel_zeus | local source |

## Quick start

If you already have the PyTorch CPU stack, `torch_zeus`, and
`sgl_kernel_zeus` installed in an active Python 3.12 environment, the SGLang
side is one script:

```bash
bash zeus_dev/setup_zeus_dev.sh
```

That script checks the environment, copies `python/pyproject_zeus.toml` to
`python/pyproject.toml`, and runs `pip install -e python --no-build-isolation`.
It does not install `torch_zeus` or `sgl_kernel_zeus`; install those first.
After it succeeds, continue with §3 (HF cache) and the smoke tests below.

## 0. Prerequisites and box layout

Pick a project root that holds all the source trees. The rest of this doc
assumes you have exported `ZEUS_ROOT` to point at it, plus an `HF_CACHE_DIR`
on a writable data partition:

```bash
export ZEUS_ROOT=/path/to/your/zeus_workspace
export HF_CACHE_DIR=/path/to/your/hf_cache
```

Expected layout under `$ZEUS_ROOT`:

```
$ZEUS_ROOT/
├── torch_zeus/                  # torch_zeus source (provides torch.zeus)
│   └── sgl-kernel-zeus/python/  # sgl_kernel_zeus source (fused kernels)
├── sglang/                      # this repo (pyproject_zeus.toml lives here)
└── triton_qmnpu/                # optional: custom triton 3.4.x source
```

You also need:

- **Python 3.12 environment**. Python 3.13 is allowed by SGLang metadata, but
  if you use it, handle `outlines_core==0.1.26` explicitly because the pinned
  release has no cp313 wheel and the sdist metadata path can fail during pip
  resolution.

- **Network reachability** for `huggingface.co`, `pypi.org`,
  and `download.pytorch.org`. Test with
  `curl -m 8 -sI <url>` before starting.

## 1. Install PyTorch CPU stack and Zeus backend packages

Install the PyTorch CPU stack first. `torch_zeus` and `sgl_kernel_zeus` are C++
extensions compiled against the active torch ABI.

```bash
python -m pip install \
  torch==2.9.1+cpu \
  torchvision==0.24.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

Then install the unreleased internal Zeus packages. They are NOT on PyPI and
are intentionally not referenced by `python/pyproject_zeus.toml`.

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

Smoke-test in isolation:

```bash
python -c "
import torch_zeus, torch_zeus._C
import sgl_kernel_zeus
import torch
print('torch:', torch.__version__)
print('torch.zeus.is_available:', torch.zeus.is_available())
print('zeus device count:', torch.zeus.device_count())
"
```

Expected:

```
torch: 2.9.1+cpu
torch.zeus.is_available: True
zeus device count: 1
```

> **Critical**: note the exact `torch.__version__` printed here. `torch_zeus._C`
> and `sgl_kernel_zeus` are C++ extensions compiled against this specific
> torch ABI. If pip replaces torch later, rebuild or reinstall the Zeus backend
> packages against the new torch.

## 2. Install SGLang with `pyproject_zeus.toml`

SGLang ships several pyproject files:

- `python/pyproject.toml` — default **CUDA** build. Pulls in `cuda-python`,
  `flashinfer_python`, `sgl-kernel`, `nvidia-*`, `torch_memory_saver`. **Do not
  use this on zeus.**
- `python/pyproject_other.toml` — non-CUDA hardware. Defines `srt_hip`,
  `srt_npu`, and `srt_hpu`.
- `python/pyproject_zeus.toml` — Zeus source install. It pins the SGLang
  runtime dependency set and expects `torch_zeus` / `sgl_kernel_zeus` to be
  installed before SGLang.

Validate the active Python environment first:

```bash
cd "path/to//sglang"
python zeus_dev/check_zeus_env.py
```

Then install:

```bash
cp python/pyproject_zeus.toml python/pyproject.toml
python -m pip install -e python --no-build-isolation
```

The wrapper does the same check, pyproject copy, and editable install:

```bash
bash zeus_dev/setup_zeus_dev.sh
```

This flow avoids the old `python/.zeus_editable` project and avoids the
post-install torch/torchvision repair step. If pip tries to replace torch,
stop and reinstall the PyTorch CPU stack before rebuilding `torch_zeus`.

### 2a. Verify after install

```bash
python -c "
import torch, torchvision, torch_zeus, torch_zeus._C, sgl_kernel_zeus
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('torch.zeus.is_available:', torch.zeus.is_available())
print('zeus device count:', torch.zeus.device_count())
import sglang.srt
import sglang
print('sglang:', sglang.__version__)
print('OK')
"
```

Expected:
```
torch: 2.9.1+cpu
torchvision: 0.24.1+cpu
torch.zeus.is_available: True
zeus device count: 1
sglang: 0.5.6.post2
OK
```

<!-- ### 2c. Things left over but harmless

- **`triton 3.6.0` (PyPI)** replaces any custom `triton 3.4.x` that may have
  been built from `$ZEUS_ROOT/triton_qmnpu`. This is **OK** for the documented
  launch flags (`--attention-backend torch_native --sampling-backend pytorch
  --disable-cuda-graph`) and for `demo_zeus_layer_compare.py` (all 13 stages
  PASS with stock triton). Rebuild the custom triton only if you later need
  a triton-backed code path:
  ```bash
  cd "$ZEUS_ROOT/triton_qmnpu/python"
  pip install -e .   # 20–60 min LLVM/C++ build
  ```
- **`nvidia-cublas-cu13`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`,
  `cuda-toolkit`, `cuda-bindings`, etc.** were pulled in by torchao /
  torchvision dep chains. They sit on disk (~2 GB) but the zeus path never
  loads them. Safe to ignore, or run
  `pip uninstall -y nvidia-cublas-cu13 nvidia-cudnn-cu13 nvidia-nccl-cu13
  nvidia-cusparselt-cu13 nvidia-curand-cu13 nvidia-cusolver-cu13
  nvidia-cusparse-cu13 nvidia-cufft-cu13 nvidia-cufile-cu13
  nvidia-cuda-runtime-cu13 nvidia-cuda-nvrtc-cu13 nvidia-cuda-cupti-cu13
  nvidia-nvjitlink-cu13 nvidia-nvtx-cu13 nvidia-nvshmem-cu13 cuda-toolkit
  cuda-bindings cuda-pathfinder` if you want the disk back. -->

## 3. HuggingFace cache location

Point HF cache at a writable data partition (do **not** use
`~/.cache/huggingface` if home is on a small volume):

```bash
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1   # optional, faster downloads
```

Add `HF_HOME` (and optionally `HF_HUB_ENABLE_HF_TRANSFER`) to your shell rc
(`~/.bashrc` / `~/.zshrc`) so future sessions inherit them.

## 4. Smoke-test: `demo_zeus_layer_compare.py`

This is the canonical layer-by-layer correctness test for Qwen2.5-0.5B. It
loads the HF model, runs each component on CPU (golden reference) and on
zeus, and prints per-stage `PASS`/`DIFF`.

```bash
cd "$ZEUS_ROOT/sglang/zeus_dev"
python demo_zeus_layer_compare.py
```

A clean run produces 13 PASSes ending with:

```
============================================================
Summary
============================================================
  embedding            : PASS
  rmsnorm              : PASS
  silu_and_mul         : PASS
  rope                 : PASS
  qkv_proj             : PASS
  o_proj               : PASS
  mlp                  : PASS
  store_kv_cache       : PASS
  extend_attention     : PASS
  decode_attention     : PASS
  transformer_block    : PASS
  lm_head              : PASS
  full_model           : PASS
============================================================
```

The full-model stage feeds `"Hello, this is a test for Zeus device
comparison."` through 24 transformer layers and the LM head; greedy token
should match between CPU and zeus (token 358 = `" I"`, top-5 overlap 5/5).

Expected harmless warnings:
- `[ZEUS Fallback] Operator 'aten::arange.start_out'` — known fallback,
  see §6.
- `Only CUDA support GGUF/AWQ quantization currently` — module-load
  warnings; the demo doesn't use these paths.
- `Zeus does not fully support inductor yet, using eager` — torch.compile
  falls back to eager on zeus.

## 5. Launch the SGLang server

```bash
SGLANG_DEVICE=zeus python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --device zeus \
  --dtype bfloat16 \
  --disable-cuda-graph \
  --attention-backend torch_native \
  --sampling-backend pytorch \
  --mem-fraction-static 0.5 \
  --max-running-requests 1 \
  --tp-size 1 \
  --host 127.0.0.1 --port 38900
```

Why each non-default flag:

- `SGLANG_DEVICE=zeus` — forces SGLang's internal `is_cuda()` / `is_zeus()`
  dispatch onto the zeus branch even on a box where CUDA might also be
  visible.
- `--device zeus` — explicit device selection.
- `--disable-cuda-graph` — there is no cuda graph capture on zeus.
- `--attention-backend torch_native` — the only attention backend currently
  wired up with zeus fallbacks (see `torch_native_backend.py::_sdpa`). Do
  not use `flashinfer`, `triton`, or `fa3` — they all require CUDA.
- `--sampling-backend pytorch` — bypass `sgl_kernel` sampling ops.

Send a request:

```bash
curl -s -m 300 -X POST http://127.0.0.1:38900/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is",
       "sampling_params": {"max_new_tokens": 8, "temperature": 0}}'
```

Should return text continuing the prompt coherently, e.g.
`" Paris. It is the largest city in"`.

## 6. Things that are slow / missing

The first end-to-end zeus run will still print `[ZEUS Fallback]` messages for residual ops.
These are correctness-correct but slower than native zeus implementations.

SGLang-side zeus compatibility is now complete enough to:
- Pass all 13 stages of `demo_zeus_layer_compare.py` against Qwen2.5-0.5B.
- Boot the launch_server, warm up, and serve a generation request end to
  end with documented launch flags.

Remaining speedups are expected to come from `torch_zeus` /
`sgl_kernel_zeus` closing their fallback list — not from further SGLang
patches.

## 7. Quick reference — copy-paste install script

For a fresh box, the full sequence is (set the two env vars at the top to
match your environment):

```bash
# 0a. Workspace paths — EDIT THESE
export HF_CACHE_DIR=/path/to/your/hf_cache

# 0b. Python environment
# Use Python 3.12 for the main Zeus install path.
# Python 3.13 is allowed by SGLang metadata, but currently needs an outlines_core cp313 workaround.
conda create -n zeus_sglang_py312 python=3.12 -y
conda activate zeus_sglang_py312
python -m pip install --upgrade pip setuptools wheel

# 1. PyTorch CPU stack
python -m pip install \
  torch==2.9.1+cpu \
  torchvision==0.24.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu

# 2. torch_zeus + sgl_kernel_zeus from source
cd torch_zeus && python -m pip install -e .
cd torch_zeus/sgl-kernel-zeus/python && python -m pip install -e .

# 3. SGLang with the Zeus pyproject
cd sglang
python zeus_dev/check_zeus_env.py
bash zeus_dev/setup_zeus_dev.sh

# 4. HF cache
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1

# 5. Verify
python -c "
import torch, torchvision, torch_zeus, torch_zeus._C, sgl_kernel_zeus, sglang
assert torch.__version__ == '2.9.1+cpu', torch.__version__
assert torch.zeus.is_available()
print('install OK | torch', torch.__version__, '| sglang', sglang.__version__)
"

# 6. Layer-by-layer smoke test
cd "sglang/zeus_dev"
python demo_zeus_layer_compare.py
```

If the verify line at step 5 prints `install OK | torch 2.9.1+cpu | sglang
0.5.6.post2` and step 6 ends with 13 `PASS` lines, your zeus install is
ready.
