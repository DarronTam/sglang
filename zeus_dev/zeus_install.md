# SGLang on Zeus — Install & Launch Guide

This document describes how to install and run SGLang on a pure-Zeus (no CUDA)
box. It is the zeus counterpart of the AMD (`srt_hip`) / NPU (`srt_npu`) flows
documented in `python/pyproject_other.toml`.

## 1. Prerequisites

Zeus support depends on two internal packages that are currently **in
development** and **not published to PyPI or any public index**:

| Package            | Provides                                               | Where it lives              |
| ------------------ | ------------------------------------------------------ | --------------------------- |
| `torch_zeus`       | the `torch.zeus` device, CPU-fallback shim, pack_weights | internal source tree        |
| `sgl_kernel_zeus`  | fused rmsnorm / silu_and_mul / store_kv_cache / etc.   | internal source tree        |

Until these are released, they must be installed **from their local source
trees** before you install the SGLang extra. A canonical working layout on the
dev box is:

```
/root/project/
├── torch_zeus/                 # torch_zeus source
│   └── sgl-kernel-zeus/python/ # sgl_kernel_zeus source
└── sglang/                     # this repo
```

### Installing `torch_zeus` and `sgl_kernel_zeus`

```bash
# 1. torch_zeus (provides torch.zeus device)
cd /root/project/torch_zeus
pip install -e .

# 2. sgl_kernel_zeus (fused kernels)
cd /root/project/torch_zeus/sgl-kernel-zeus/python
pip install -e .
```

Smoke-test them in isolation before moving on:

```bash
python -c "
import torch_zeus, torch_zeus._C
import sgl_kernel_zeus
import torch
print('torch.zeus.is_available:', torch.zeus.is_available())
print('zeus device count:', torch.zeus.device_count())
"
```

Both imports must succeed without a circular-import warning. If you see
`Failed to import C++ extension: cannot import name '_C' from partially
initialized module 'torch_zeus'`, re-pull `torch_zeus` — that circular import
was fixed upstream on 2026-04-15.

## 2. Install SGLang with the `srt_zeus` extra

SGLang ships two pyproject files:

- `python/pyproject.toml` — the default **CUDA** build. Pulls in
  `cuda-python`, `flashinfer_python`, `sgl-kernel`, `nvidia-*`,
  `torch_memory_saver`. **Do not use this on zeus.**
- `python/pyproject_other.toml` — the "other hardware" build. Contains
  `srt_hip` / `srt_npu` / `srt_hpu` and, as of 2026-04-15, `srt_zeus`.

Install with:

```bash
cd /root/project/sglang
ln -sf pyproject_other.toml python/pyproject.toml   # switch pyproject
pip install -e "python[srt_zeus]"
```

Or, if you prefer to keep `pyproject.toml` pointing at the CUDA build, use the
`--config-settings` trick:

```bash
cd /root/project/sglang/python
pip install -e . \
  --config-settings editable_mode=compat \
  --config-settings pyproject=pyproject_other.toml
# then manually trigger the extra:
pip install "sglang[srt_zeus] @ ."
```

The `srt_zeus` extra deliberately only lists `runtime_common` + `torch`. It
does **not** try to pin `torch_zeus` or `sgl_kernel_zeus`, because those are
unreleased — you must install them from source first (step 1).

## 3. Launch the server

Set `SGLANG_DEVICE=zeus` so SGLang's internal `is_cuda()` / `is_zeus()`
dispatch picks the zeus branch even on a box where CUDA is also available:

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

Key flags, and why:

- `--device zeus` — explicit device selection.
- `--disable-cuda-graph` — there is no cuda graph capture on zeus.
- `--attention-backend torch_native` — the only attention backend currently
  wired up with zeus fallbacks (see `torch_native_backend.py::_sdpa`). Do not
  use `flashinfer`, `triton`, or `fa3` — they all require CUDA.
- `--sampling-backend pytorch` — bypass `sgl_kernel` sampling ops.

## 4. Smoke-test a generation request

```bash
curl -s -m 300 -X POST http://127.0.0.1:38900/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is",
       "sampling_params": {"max_new_tokens": 8, "temperature": 0}}'
```

On a CPU-fallback-heavy path (current state, 2026-04-15) expect ~6 s/token for
Qwen2.5-0.5B. The request should return text that continues the prompt
coherently, e.g. `" Paris. It is the largest city in"`.

## 5. Things that are slow / missing

The first end-to-end zeus run goes through a lot of `[ZEUS Fallback]` messages
for ops like `arange.start_out`, `scatter.src_out`, `add.Tensor`,
`index.Tensor_out`, and `zenl_add_kernel`. These are correctness-correct but
cost ~100x vs a native zeus implementation. The queue of zeus ops to upstream
is tracked in `zeus_dev/update_log_20260415.md` under the "Known caveats"
section.

SGLang-side zeus compatibility is now complete enough to boot the server,
warm up, and serve a generation request end to end. The remaining speedups
are expected to come from `torch_zeus` / `sgl_kernel_zeus` closing their
fallback list — not from further SGLang patches.
