"""Utility helpers for exporting ONNX policies across MJLab tasks."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import torch


def export_policy_module_to_onnx(
  policy_module: torch.nn.Module,
  inputs: torch.Tensor | Sequence[torch.Tensor],
  export_path: str,
  *,
  input_names: Sequence[str],
  output_names: Sequence[str],
  dynamic_shapes: Sequence[dict[int, "torch.export.Dim"]] | None = None,
  dynamic_axes: dict[str, dict[int, str]] | None = None,
  opset_version: int = 18,
  verbose: bool = False,
) -> None:
  """Export a policy module to ONNX with compatibility across PyTorch versions."""
  policy_module.to("cpu")
  policy_module.eval()

  example_inputs: Iterable[torch.Tensor]
  if isinstance(inputs, torch.Tensor):
    example_inputs = (inputs,)
  else:
    example_inputs = tuple(inputs)

  export_kwargs: dict[str, object] = {
    "export_params": True,
    "opset_version": opset_version,
    "verbose": verbose,
  }
  if _supports_dynamic_shapes(dynamic_shapes):
    export_kwargs["dynamic_shapes"] = dynamic_shapes
  else:
    export_kwargs["dynamic_axes"] = dynamic_axes or {}
    export_kwargs["dynamo"] = False

  with torch.no_grad():
    torch.onnx.export(
      policy_module,
      example_inputs,
      export_path,
      input_names=input_names,
      output_names=output_names,
      **export_kwargs,
    )


def _supports_dynamic_shapes(
  dynamic_shapes: Sequence[dict[int, "torch.export.Dim"]] | None,
) -> bool:
  if dynamic_shapes is None:
    return False
  export_mod = getattr(torch, "export", None)
  if export_mod is None or not hasattr(export_mod, "Dim"):
    return False
  return _torch_version_ge(2, 9)


def _torch_version_ge(major: int, minor: int) -> bool:
  version_str = getattr(torch, "__version__", "")
  match = re.match(r"(\\d+)\\.(\\d+)", version_str)
  if not match:
    return False
  current_major = int(match.group(1))
  current_minor = int(match.group(2))
  return (current_major, current_minor) >= (major, minor)
