"""Scratch Chronos-2 training on pinned TimePFN data, without an equation head."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch

from tsfmxai.datasets.chronos2_adapter import adapt_grouped_batch
from tsfmxai.datasets.lmc_synth import TimePFNBatches, batch_loader, verify_timepfn

CHRONOS_REVISION = "10afa9ebe016e514f9d7dc1aa873f66af57e116b"
CONFIG_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


@dataclass(frozen=True)
class TrainingSpec:
    independent_lr: float = 5e-4
    # LOCAL choice: newest author guidance says lower LR, without a number.
    lmc_lr: float = 1e-4
    workers: int = 2
    microbatch: int = 4
    checkpoint_interval: int = 2500
    validation_batches: int = 4
    steps_per_stage: int | None = None  # None = all published-loader windows.
    device: str = "cuda"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0

    def __post_init__(self):
        if self.early_stopping_patience < 1 or self.early_stopping_min_delta < 0:
            raise ValueError("Invalid early stopping settings")
        if not 0 < self.lmc_lr < self.independent_lr:
            raise ValueError("The latest author guidance requires a smaller LMC LR")
        if min(self.microbatch, self.checkpoint_interval, self.validation_batches) < 1 or self.workers < 0:
            raise ValueError("Invalid training controls")
        if self.steps_per_stage is not None and self.steps_per_stage < 1:
            raise ValueError("steps_per_stage must be positive")


def scratch_model(seed=2021, *, tiny=False):
    """Download architecture configuration only. Never load pretrained weights."""
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model
    if tiny:
        config = Chronos2CoreConfig(
            d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2, dropout_rate=0.,
            chronos_config=dict(context_length=96, input_patch_size=16,
                               output_patch_size=16, input_patch_stride=16,
                               quantiles=[0.1, 0.5, 0.9], max_output_patches=6,
                               use_reg_token=True))
    else:
        config = Chronos2CoreConfig.from_pretrained("amazon/chronos-2", revision=CONFIG_REVISION)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Chronos2Model(config)


def _inputs(batch, model, device, start=0, stop=None):
    context, future = batch["context"][start:stop], batch["future"][start:stop]
    adapted = adapt_grouped_batch(context, future, num_examples=len(context),
                                  channels_per_example=context.shape[2])
    result = {key: adapted[key].to(device, non_blocking=True)
              for key in ("context", "future_target", "future_covariates", "group_ids")}
    result["num_output_patches"] = math.ceil(future.shape[1] / model.chronos_config.output_patch_size)
    return result


@torch.no_grad()
def evaluate(model, loader, device, microbatch):
    was_training = model.training
    model.eval()
    loss_sum = squared = absolute = elements = examples = 0
    median = list(model.chronos_config.quantiles).index(0.5)
    for batch in loader:
        for start in range(0, len(batch["context"]), microbatch):
            inputs = _inputs(batch, model, device, start, start + microbatch)
            output = model(**inputs)
            target = inputs["future_target"]
            error = output.quantile_preds[:, median, :target.shape[1]] - target
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            elements += error.numel()
            n = min(microbatch, len(batch["context"]) - start)
            loss_sum += output.loss.item() * n
            examples += n
    model.train(was_training)
    return dict(loss=loss_sum / examples, mse=squared / elements, mae=absolute / elements,
                examples=examples)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _observe_early_stopping(state, model, score, position, settings):
    """Freeze selection at the first trigger; later improvements cannot leak in."""
    if not math.isfinite(score):
        raise FloatingPointError("Nonfinite validation loss")
    if state["trigger"] is not None:
        return
    if state["best"] is None or score < state["best"]["loss"] - settings.early_stopping_min_delta:
        state["best"] = dict(position, loss=score)
        state["weights"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        state["bad_checks"] = 0
    else:
        state["bad_checks"] += 1
        if state["bad_checks"] >= settings.early_stopping_patience:
            state["trigger"] = dict(position, loss=score)


def train(model, data, settings, upstream, cache, output, *, resume=False):
    """Two-stage fixed-budget training with atomic resumable checkpoints.

    Timing synchronizes at update boundaries: next_batch_seconds measures exposed
    input waiting, NOT GPU idle time. These diagnostic boundaries affect overlap.
    Checkpoints are trusted local pickle files; never load an untrusted download.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"Use a new output directory or resume=True: {checkpoint}")
    if resume and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for pilot/full; CPU is for explicit smoke tests")
    random.seed(data.seed)
    np.random.seed(data.seed)
    torch.manual_seed(data.seed)
    model.to(settings.device)
    if torch.device(settings.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    signature = dict(data=asdict(data), training=asdict(settings),
                     generator=verify_timepfn(upstream), architecture=model.config.to_dict())
    # JSON canonicalization removes incidental tuples and dtype objects.
    signature = json.loads(json.dumps(signature, default=str))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.independent_lr,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    saved = None
    history = []
    early = dict(best=None, weights=None, bad_checks=0, trigger=None)
    elapsed_before = 0.
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Resume configuration/provenance does not match")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        early = saved["early_stopping"]
        elapsed_before = saved["elapsed_seconds"]
    began = time.perf_counter()
    completed_before = sum(data.updates(name) if settings.steps_per_stage is None
                           else min(data.updates(name), settings.steps_per_stage)
                           for name in ("independent",))
    stage_rates = {}
    for stage, family in enumerate(("independent", "lmc")):
        total = data.updates(family)
        if settings.steps_per_stage is not None:
            total = min(total, settings.steps_per_stage)
        if total < 1:
            raise ValueError("Dataset has no complete batch")
        if saved and stage < saved["stage"]:
            continue
        start = saved["next_step"] if saved and stage == saved["stage"] else 0
        lr = settings.independent_lr if stage == 0 else settings.lmc_lr
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total, pct_start=0.3,
            anneal_strategy="cos", cycle_momentum=True, base_momentum=0.85,
            max_momentum=0.95, div_factor=25, final_div_factor=1e4)
        if saved and stage == saved["stage"]:
            scheduler.load_state_dict(saved["scheduler"])
            optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["torch_rng"])
            if settings.device.startswith("cuda"):
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if start == total:
            continue
        dataset = TimePFNBatches(data, upstream, cache, family, limit=total, start=start)
        loader = batch_loader(dataset, settings.workers, pin_memory=settings.device.startswith("cuda"))
        validation = batch_loader(TimePFNBatches(data, upstream, cache, family, split=1,
                                                  limit=settings.validation_batches))
        iterator = iter(loader)
        stage_started = time.perf_counter()
        model.train()
        for step in range(start, total):
            _sync(settings.device)
            before = time.perf_counter()
            batch = next(iterator)
            ready = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = 0.
            used_lr = optimizer.param_groups[0]["lr"]
            for offset in range(0, data.batch_size, settings.microbatch):
                inputs = _inputs(batch, model, settings.device, offset, offset + settings.microbatch)
                result = model(**inputs)
                weight = min(settings.microbatch, data.batch_size - offset) / data.batch_size
                if not torch.isfinite(result.loss):
                    raise FloatingPointError(f"Nonfinite loss at {family}/{step}")
                (result.loss * weight).backward()
                loss += result.loss.detach().item() * weight
            optimizer.step()
            scheduler.step()
            _sync(settings.device)
            end = time.perf_counter()
            row = dict(stage=family, update=step + 1, loss=loss, learning_rate=used_lr,
                       next_batch_seconds=ready - before, train_seconds=end - ready,
                       examples=data.batch_size)
            history.append(row)
            if (step + 1) % (100 if total > 100 else 1) == 0 or step + 1 == total:
                observed = step + 1 - start
                rate = (time.perf_counter() - stage_started) / observed
                stage_rates[family] = rate
                remaining_current = (total - step - 1) * rate
                if family == "independent":
                    lmc_total = data.updates("lmc")
                    if settings.steps_per_stage is not None:
                        lmc_total = min(lmc_total, settings.steps_per_stage)
                    # Before LMC starts, its cost is unknown; show an explicit
                    # provisional scenario using the measured independent rate.
                    remaining_total = remaining_current + lmc_total * rate
                    basis = "LMC extrapolated from independent stage"
                else:
                    remaining_total = remaining_current
                    basis = "LMC observed"
                print(json.dumps({"progress_stage": family, "update": step + 1,
                                  "total_updates_stage": total,
                                  "stage_eta_hours": remaining_current / 3600,
                                  "full_run_eta_hours": remaining_total / 3600,
                                  "eta_basis": basis,
                                  "estimated_finish_utc": datetime.fromtimestamp(
                                      time.time() + remaining_total, timezone.utc).isoformat()}), flush=True)
            # Append telemetry continuously; a resumed checkpoint is authoritative.
            with (output / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            if (step + 1) % settings.checkpoint_interval == 0 or step + 1 == total:
                row["validation"] = evaluate(model, validation, settings.device, settings.microbatch)
                position = dict(stage=family, stage_update=step + 1, global_update=len(history),
                                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                elapsed_seconds=elapsed_before + time.perf_counter() - began)
                # Compare a fixed LMC validation distribution only, after the
                # independent warm-up. Continue full training after the trigger.
                if family == "lmc":
                    _observe_early_stopping(early, model, row["validation"]["loss"], position, settings)
                payload = dict(signature=signature, stage=stage, next_step=step + 1,
                               model=model.state_dict(), optimizer=optimizer.state_dict(),
                               scheduler=scheduler.state_dict(), history=history,
                               early_stopping=early, elapsed_seconds=position["elapsed_seconds"],
                               last_position=position, torch_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state_all() if settings.device.startswith("cuda") else [])
                temporary = output / "latest.tmp"
                torch.save(payload, temporary)
                os.replace(temporary, checkpoint)
                print(json.dumps(row), flush=True)
        del iterator, loader
    test = {family: evaluate(model, batch_loader(TimePFNBatches(
        data, upstream, cache, family, split=2, limit=settings.validation_batches)),
        settings.device, settings.microbatch) for family in ("independent", "lmc")}
    model.save_pretrained(output / "model_without_early_stopping")
    final_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    end_position = final_checkpoint["last_position"]
    model.load_state_dict(early["weights"])
    model.save_pretrained(output / "model_early_stopping")
    early_test = {family: evaluate(model, batch_loader(TimePFNBatches(
        data, upstream, cache, family, split=2, limit=settings.validation_batches)),
        settings.device, settings.microbatch) for family in ("independent", "lmc")}
    model.load_state_dict(final_checkpoint["model"])
    selection = dict(
        policy="LMC validation loss only; independent stage always completes; restore best before trigger",
        patience=settings.early_stopping_patience, min_delta=settings.early_stopping_min_delta,
        triggered=early["trigger"] is not None,
        stop_reason="patience_exhausted" if early["trigger"] else "budget_exhausted_without_trigger",
        stopped_at=early["trigger"] or end_position, selected_checkpoint=early["best"],
        bad_checks=early["bad_checks"], test=early_test,
        model_directory="model_early_stopping")
    summary = dict(updates=len(history), test=test, invocation_seconds=time.perf_counter() - began,
                   early_stopping=selection,
                   without_early_stopping=dict(stop_reason="budget_exhausted", stopped_at=end_position,
                                              model_directory="model_without_early_stopping", test=test),
                   parameters=sum(p.numel() for p in model.parameters()),
                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if settings.device.startswith("cuda") else 0)
    (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, history
