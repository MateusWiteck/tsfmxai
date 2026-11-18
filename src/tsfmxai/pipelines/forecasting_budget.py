"""Matched forecasting-only training for indexed synthetic batch streams."""
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import time
import warnings

import torch
from tsfmxai.pipelines.chronos2_lmc import _inputs, _sync


def train_online_plan(plan, budget, output, *, workers=12, device='cuda'):
    """Run the own-generator forecasting control with the baseline architecture."""
    from tsfmxai.datasets.online import make_dataloader
    from tsfmxai.pipelines.chronos2_lmc import scratch_model
    if plan.total_batches < budget.max_updates:
        raise ValueError('Plan must cover the declared update budget')
    if (plan.training.batch_size, plan.training.context_length,
            plan.training.prediction_length) != (16, 96, 96):
        raise ValueError('Use batch=16, context=96, horizon=96')
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(2021)
    model = scratch_model(seed=2021)
    loader = make_dataloader(plan, num_workers=workers)
    charged = replace(budget, preparation_seconds=budget.preparation_seconds + time.perf_counter()-started)
    return train_forecasting(model, loader, charged, output, device=device)


@dataclass(frozen=True)
class ForecastingBudget:
    max_hours: float | None = None
    max_updates: int = 445433
    stage_boundary: int = 346996
    preparation_seconds: float = 0.
    checkpoint_interval: int = 2500

    def __post_init__(self):
        if self.max_hours is not None and (not math.isfinite(self.max_hours) or self.max_hours <= 0):
            raise ValueError('max_hours must be finite and positive')
        if not 0 < self.stage_boundary < self.max_updates or self.checkpoint_interval < 1:
            raise ValueError('Invalid update budget')
        if not math.isfinite(self.preparation_seconds) or self.preparation_seconds < 0:
            raise ValueError('Invalid preparation time')


def train_forecasting(model, loader, budget, output, *, device='cuda'):
    """Train scratch Chronos on [16,96,6] context/future batches.

    Clock includes loader startup, input waits, updates and periodic saves.
    Charge earlier preparation through preparation_seconds. Final save is
    separately reported; an in-flight batch may overrun the deadline. This
    entry point starts a new run; it never silently resumes an existing run.
    """
    if budget.max_hours is not None:
        warnings.warn("The legacy update-indexed OneCycle can end before its final LR phase "
                      "under a time limit. Use scripts/run_prior_baseline.py for the "
                      "prospective matched-time study.", FutureWarning, stacklevel=2)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / 'latest.pt'
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    began = time.perf_counter()
    elapsed = lambda: budget.preparation_seconds + time.perf_counter() - began
    expired = lambda: budget.max_hours is not None and elapsed() >= budget.max_hours * 3600
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(.9, .999), eps=1e-8)
    iterator = iter(loader)
    completed = 0
    scheduler = None
    reason = 'max_updates'

    def save():
        temporary = directory / 'latest.tmp'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=None if scheduler is None else scheduler.state_dict(),
                        updates=completed, elapsed_seconds=elapsed(), budget=budget,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []), temporary)
        os.replace(temporary, checkpoint)

    for step in range(budget.max_updates):
        if expired():
            reason = 'max_hours'
            break
        if step in (0, budget.stage_boundary):
            stage_steps = budget.stage_boundary if step == 0 else budget.max_updates - step
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=5e-4 if step == 0 else 1e-4, total_steps=stage_steps,
                pct_start=.3, anneal_strategy='cos', cycle_momentum=True,
                base_momentum=.85, max_momentum=.95, div_factor=25, final_div_factor=1e4)
        _sync(device)
        before = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            reason = 'data_exhausted'
            break
        ready = time.perf_counter()
        if expired():
            reason = 'max_hours'
            break
        for key in ('context', 'future'):
            if tuple(batch[key].shape) != (16, 96, 6):
                raise ValueError(f'{key} must have shape [16,96,6]')
        optimizer.zero_grad(set_to_none=True)
        result = model(**_inputs(batch, model, device))
        if not torch.isfinite(result.loss):
            raise FloatingPointError('Nonfinite forecasting loss')
        result.loss.backward()
        optimizer.step()
        scheduler.step()
        _sync(device)
        completed = step + 1
        row = dict(update=completed, loss=result.loss.item(), next_batch_seconds=ready-before,
                   train_seconds=time.perf_counter()-ready, elapsed_seconds=elapsed())
        with (directory / 'telemetry.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        if completed % budget.checkpoint_interval == 0:
            save()
    training_seconds = elapsed()
    save()
    summary = dict(updates=completed, stop_reason=reason, training_seconds=training_seconds,
                   final_save_seconds=elapsed()-training_seconds, total_seconds=elapsed(),
                   max_hours=budget.max_hours)
    (directory / 'result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
