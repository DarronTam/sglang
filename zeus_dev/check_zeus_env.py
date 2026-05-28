#!/usr/bin/env python3
"""Validate the Python environment before installing SGLang for Zeus."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import importlib.util
import sys
from pathlib import Path


EXPECTED_TORCH = "2.10.0"
EXPECTED_TORCHVISION = "0.25.0"
EXPECTED_TORCHAO = "0.9.0"
KNOWN_PYTORCH_BUILDS = {
    "cpu",
    "cu118",
    "cu121",
    "cu124",
    "cu126",
    "cu128",
    "cu129",
    "rocm6.3",
    "xpu",
}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def base_version(version: str) -> str:
    return version.split("+", 1)[0]


def local_build(version: str) -> str:
    return version.split("+", 1)[1] if "+" in version else ""


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def read_torchvision_version() -> str:
    spec = importlib.util.find_spec("torchvision")
    if spec is None or spec.origin is None:
        fail("torchvision is not installed. Install the CPU-paired torchvision wheel first.")

    version_file = Path(spec.origin).parent / "version.py"
    for line in version_file.read_text().splitlines():
        if line.startswith("__version__ = "):
            return line.split("=", 1)[1].strip().strip("'\"")
    fail(f"could not read torchvision version from {version_file}")


def install_hint(package: str, version: str, build: str) -> str:
    if build in KNOWN_PYTORCH_BUILDS:
        return (
            f"python -m pip install --no-deps --force-reinstall "
            f"{package}=={version}+{build} --index-url https://download.pytorch.org/whl/{build}"
        )
    return f"install {package}=={version} from the same wheel family as torch"


def require_import(module: str, detail: str) -> object:
    try:
        return importlib.import_module(module)
    except Exception as exc:
        fail(f"could not import {module}: {exc}\n  {detail}")


def main() -> int:
    if sys.version_info < (3, 10):
        fail(f"Python >=3.10 is required, got {sys.version.split()[0]}")

    torch = require_import(
        "torch",
        "Install torch before torch_zeus, for example from the PyTorch CPU wheel index.",
    )
    torch_version = torch.__version__
    torch_build = local_build(torch_version)

    if base_version(torch_version) != EXPECTED_TORCH:
        fail(
            f"expected torch {EXPECTED_TORCH}, got {torch_version}\n"
            f"  {install_hint('torch', EXPECTED_TORCH, torch_build or 'cpu')}"
        )

    torchvision_version = read_torchvision_version()
    torchvision_build = local_build(torchvision_version)
    if base_version(torchvision_version) != EXPECTED_TORCHVISION:
        fail(
            f"expected torchvision {EXPECTED_TORCHVISION}, got {torchvision_version}\n"
            f"  {install_hint('torchvision', EXPECTED_TORCHVISION, torch_build or 'cpu')}"
        )
    if torch_build != torchvision_build:
        fail(
            "torch and torchvision builds do not match.\n"
            f"  torch: {torch_version}\n"
            f"  torchvision: {torchvision_version}\n"
            f"  {install_hint('torchvision', EXPECTED_TORCHVISION, torch_build or 'cpu')}"
        )

    require_import(
        "torchvision",
        "The package is installed, but its C++ extension did not load. Reinstall torchvision from the matching PyTorch index.",
    )

    torchao_version = package_version("torchao")
    if torchao_version is not None and base_version(torchao_version) != EXPECTED_TORCHAO:
        fail(f"expected torchao {EXPECTED_TORCHAO}, got {torchao_version}")

    require_import(
        "torch_zeus",
        "Install torch_zeus from its source tree or wheel before installing SGLang.",
    )
    require_import(
        "torch_zeus._C",
        "torch_zeus is present, but its C++ extension is not loadable. Rebuild it against the active torch.",
    )
    require_import(
        "sgl_kernel_zeus",
        "Install sgl_kernel_zeus from the torch_zeus/sgl-kernel-zeus package first.",
    )

    if not hasattr(torch, "zeus"):
        fail("torch.zeus is not registered. Importing torch_zeus did not attach the Zeus backend.")
    if not torch.zeus.is_available():
        fail("torch.zeus.is_available() returned False")
    device_count = torch.zeus.device_count()
    if device_count <= 0:
        fail(f"torch.zeus.device_count() returned {device_count}")

    print("[zeus-check] OK")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  torch: {torch_version}")
    print(f"  torchvision: {torchvision_version}")
    print(f"  torchao: {torchao_version or '<not installed yet>'}")
    print(f"  zeus device count: {device_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
