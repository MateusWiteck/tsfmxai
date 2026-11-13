"""Adapt tsfmxai and LMC-Synth batches to the Chronos-2 batch contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype)


def adapt_grouped_batch(
    context: Any,
    future: Any,
    *,
    num_examples: int,
    channels_per_example: int,
    equation_targets: Any | None = None,
    equation_mask: Any | None = None,
    noise_std: Any | None = None,
) -> dict[str, Any]:
    """Build Chronos-2 tensors while retaining one group ID per multivariate item.

    Inputs use [examples, time, channels]. Chronos-2 uses one row per channel
    and group_ids to indicate which channels interact in a multivariate example.
    Its batch_size therefore counts rows/channels, not multivariate examples.
    """
    if num_examples < 1 or channels_per_example < 1:
        raise ValueError("num_examples and channels_per_example must be positive")
    context = _tensor(context, dtype=torch.float32)
    future = _tensor(future, dtype=torch.float32)
    if (context.ndim != 3 or future.ndim != 3
            or context.shape[0] != num_examples or future.shape[0] != num_examples
            or context.shape[2] != channels_per_example
            or future.shape[2] != channels_per_example):
        raise ValueError("context/future must have [examples, time, channels] layout")
    if not torch.isfinite(context).all() or not torch.isfinite(future).all():
        raise ValueError("Chronos-2 targets and contexts must be finite")

    rows = num_examples * channels_per_example
    context_rows = context.permute(0, 2, 1).reshape(rows, context.shape[1]).contiguous()
    future_rows = future.permute(0, 2, 1).reshape(rows, future.shape[1]).contiguous()
    groups = torch.arange(num_examples, dtype=torch.long).repeat_interleave(channels_per_example)
    adapted: dict[str, Any] = {
        "context": context_rows,
        "future_target": future_rows,
        # Chronos-2 treats finite future_covariates as known inputs and masks
        # them out of the forecast loss. These rows are targets, so mark them
        # missing rather than supplying known zero-valued covariates.
        "future_covariates": torch.full(
            (rows, future.shape[1]), float("nan"), dtype=torch.float32
        ),
        "group_ids": groups,
        "target_idx_ranges": [
            (i * channels_per_example, (i + 1) * channels_per_example)
            for i in range(num_examples)
        ],
        "num_examples": num_examples,
        "channels_per_example": channels_per_example,
    }
    if equation_targets is not None:
        equations = _tensor(equation_targets, dtype=torch.long)
        if equations.shape[:2] != (num_examples, channels_per_example):
            raise ValueError("equation target leading dimensions must match examples/channels")
        adapted["equation_targets"] = equations.reshape(rows, *equations.shape[2:])
    if equation_mask is not None:
        mask = _tensor(equation_mask, dtype=torch.bool)
        if mask.shape[:2] != (num_examples, channels_per_example):
            raise ValueError("equation mask leading dimensions must match examples/channels")
        adapted["equation_mask"] = mask.reshape(rows, *mask.shape[2:])
    if noise_std is not None:
        std = _tensor(noise_std, dtype=torch.float32)
        if tuple(std.shape) != (num_examples, channels_per_example):
            raise ValueError("noise_std must have [examples, channels] shape")
        adapted["noise_std"] = std.reshape(rows)
    return adapted


def adapt_tsfmxai_batch(batch: Any, *, layout: str = "batch_time_channel",
                        output_patch_size: int | None = None) -> dict[str, Any]:
    """Adapt a padded tsfmxai batch; layout must match TrainingFormat.layout."""
    context = _tensor(batch.context, dtype=torch.float32)
    future = _tensor(batch.future, dtype=torch.float32)
    if context.ndim != 3:
        raise ValueError("tsfmxai context must have rank 3")
    if layout == "batch_channel_time":
        context = context.transpose(1, 2)
        future = future.transpose(1, 2)
    elif layout != "batch_time_channel":
        raise ValueError("unknown tsfmxai layout")
    examples = []
    for i, row in enumerate(batch.channel_mask):
        keep = torch.as_tensor(row, dtype=torch.bool)
        examples.append((context[i, :, keep], future[i, :, keep], keep))
    counts = {int(keep.sum()) for _, _, keep in examples}
    if len(counts) != 1:
        raise ValueError("Chronos-2 batch adapter requires equal channel counts per example")
    n_channels = counts.pop()
    ctx = torch.stack([item[0] for item in examples])
    fut = torch.stack([item[1] for item in examples])
    eq, eqmask, noise = [], [], []
    for i, (_, _, keep) in enumerate(examples):
        eq.append(torch.as_tensor(batch.equation_targets[i, keep]))
        eqmask.append(torch.as_tensor(batch.equation_mask[i, keep]))
        noise.append(torch.as_tensor(batch.noise_targets["std"][i, keep]))
    adapted = adapt_grouped_batch(
        ctx, fut, num_examples=len(examples), channels_per_example=n_channels,
        equation_targets=torch.stack(eq), equation_mask=torch.stack(eqmask),
        noise_std=torch.stack(noise),
    )
    if output_patch_size is None:
        raise ValueError("output_patch_size is required for Chronos-2 training")
    if type(output_patch_size) is not int or output_patch_size < 1:
        raise ValueError("output_patch_size must be a positive integer")
    adapted["num_output_patches"] = int(np.ceil(fut.shape[1] / output_patch_size))
    return adapted
