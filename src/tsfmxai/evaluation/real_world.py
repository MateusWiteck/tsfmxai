"""Deterministic real-world multivariate forecasting evaluation inputs and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


AIR_CHANNELS = ("O3 [ppbv]", "NO [ppbv]", "NO2 [ppbv]", "CO [ppmv]", "PM2.5 [µg/m**3]")
ENERGY_CHANNELS = ("load_mw", "wind_capacity_factor", "solar_capacity_factor", "hydro_inflow_mwh")


def _hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Index must be timezone-aware")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate timestamps")
    if len(frame) < 2 or not (frame.index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("Expected a continuous hourly grid")
    return frame.astype(float)


def load_ibirapuera(path: str | Path) -> pd.DataFrame:
    """Read PANGAEA 940276, preserving missing observations as NaN."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        skiprows = next(i for i, line in enumerate(handle) if line.startswith("Date/Time\t"))
    frame = pd.read_csv(path, sep="\t", skiprows=skiprows)
    if not set(AIR_CHANNELS).issubset(frame.columns):
        raise ValueError("Unexpected PANGAEA columns")
    times = pd.to_datetime(frame.pop("Date/Time"), utc=True)
    frame = frame.loc[:, AIR_CHANNELS]
    frame.index = pd.DatetimeIndex(times)
    return _hourly_frame(frame)


def _read_zip_csv(zf: ZipFile, suffix: str, *, sep: str) -> pd.DataFrame:
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one archive member ending in {suffix!r}; found {len(matches)}")
    return pd.read_csv(BytesIO(zf.read(matches[0])), sep=sep)


def load_brazil_energy(zip_path: str | Path, *, state: str = "BA", year: int = 2019) -> pd.DataFrame:
    """Join four EnDAT series in UTC for one state and reference year.

    Wind/solar timestamps are UTC. ONS-derived load/inflow timestamps are
    Brasília time (UTC-3), as specified by the source paper. No future values
    are used for filling or scaling.
    """
    if len(state) != 2 or not 2012 <= year <= 2020:
        raise ValueError("Choose a Brazilian state code and year 2012–2020")
    with ZipFile(zip_path) as zf:
        load = _read_zip_csv(zf, f"/distribute_by_consumer/Hourly_electricity_demand_per_state_{year}.csv", sep=",")
        load = load.loc[load["state"] == state, ["time", "value"]]
        if load.empty:
            raise ValueError(f"No load rows for {state}")
        load.index = pd.to_datetime(load.pop("time")).dt.tz_localize("Etc/GMT+3").dt.tz_convert("UTC")
        load = load.rename(columns={"value": ENERGY_CHANNELS[0]})

        members = (
            (f"/onshore/EnDAT_on_wind_per_unit_generation_weather_year_{year}.csv", ";", ENERGY_CHANNELS[1], True),
            (f"/PV/EnDAT_solar_pv_per_unit_generation_weather_year_{year}.csv", ";", ENERGY_CHANNELS[2], True),
            (f"/by_hydropower_plants_operation/ONS_state_inflow_distribute_by_hydropower_plants_operation_reference_year_{year}.csv", ",", ENERGY_CHANNELS[3], False),
        )
        pieces = [load]
        for suffix, sep, label, is_utc in members:
            table = _read_zip_csv(zf, suffix, sep=sep)
            if state not in table:
                raise ValueError(f"State {state} missing from {suffix}")
            times = pd.to_datetime(table.iloc[:, 0])
            times = times.dt.tz_localize("UTC" if is_utc else "Etc/GMT+3").dt.tz_convert("UTC")
            piece = pd.DataFrame({label: pd.to_numeric(table[state], errors="coerce").to_numpy()},
                                 index=pd.DatetimeIndex(times))
            pieces.append(piece)
    return _hourly_frame(pd.concat(pieces, axis=1, join="inner").loc[:, ENERGY_CHANNELS])


@dataclass(frozen=True)
class WindowSet:
    context: np.ndarray  # [window, time, channel], NaN means unavailable
    target: np.ndarray
    origins: tuple[str, ...]
    channels: tuple[str, ...]
    eligible_count: int


def concatenate_window_sets(parts: list[WindowSet]) -> WindowSet:
    """Combine years with the same channel order and window dimensions."""
    if not parts or any(part.channels != parts[0].channels or
                        part.context.shape[1:] != parts[0].context.shape[1:] or
                        part.target.shape[1:] != parts[0].target.shape[1:] for part in parts):
        raise ValueError("Window sets need matching channels and shapes")
    return WindowSet(np.concatenate([part.context for part in parts]),
                     np.concatenate([part.target for part in parts]),
                     tuple(origin for part in parts for origin in part.origins),
                     parts[0].channels, sum(part.eligible_count for part in parts))


def select_windows(frame: pd.DataFrame, *, year: int, context_length: int = 96,
                   horizon: int = 96, stride: int = 24, coverage: float = 0.8,
                   limit: int | None = None) -> WindowSet:
    """Select evenly spaced eligible origins without looking at target values.

    Eligibility uses observation masks, including the target mask, but no target
    values. A fixed grid and evenly spaced indices prevent cherry-picking dates.
    """
    if (min(context_length, horizon, stride) < 1 or (limit is not None and limit < 1)
            or not 0 < coverage <= 1):
        raise ValueError("Invalid window controls")
    values = frame.to_numpy(dtype=np.float32)
    present = np.isfinite(values)
    indices = []
    for origin in range(context_length, len(frame) - horizon + 1, stride):
        if frame.index[origin].year != year or frame.index[origin+horizon-1].year != year:
            continue
        if (present[origin-context_length:origin].mean(axis=0).min() >= coverage
                and present[origin:origin+horizon].mean(axis=0).min() >= coverage):
            indices.append(origin)
    if not indices:
        raise ValueError(f"No eligible windows for {year}")
    if limit is None:
        selected = np.asarray(indices)
    else:
        selected = np.asarray(indices)[np.linspace(0, len(indices)-1, min(limit, len(indices)), dtype=int)]
    return WindowSet(
        context=np.stack([values[i-context_length:i] for i in selected]),
        target=np.stack([values[i:i+horizon] for i in selected]),
        origins=tuple(frame.index[i].isoformat() for i in selected),
        channels=tuple(frame.columns), eligible_count=len(indices),
    )


def last_observed_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    """Past-only persistence baseline; missing observations never become targets."""
    context = np.asarray(context, dtype=float)
    last = np.full((len(context), context.shape[2]), np.nan)
    for t in range(context.shape[1]):
        last = np.where(np.isfinite(context[:, t, :]), context[:, t, :], last)
    if not np.isfinite(last).all():
        raise ValueError("Each channel needs a past observation")
    return np.repeat(last[:, None, :], horizon, axis=1)


def observed_metrics(prediction: np.ndarray, target: np.ndarray,
                     channels: tuple[str, ...]) -> dict:
    """MAE, RMSE and per-channel WAPE on observed real targets only."""
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != len(channels):
        raise ValueError("Expected aligned [window, horizon, channel] arrays")
    mask = np.isfinite(target)
    if not np.isfinite(prediction[mask]).all() or not mask.any():
        raise ValueError("Nonfinite prediction or no observed targets")
    error = prediction - target
    per_channel = {}
    for channel, name in enumerate(channels):
        observed = mask[:, :, channel]
        residual = error[:, :, channel][observed]
        target_magnitude = np.abs(target[:, :, channel][observed]).sum()
        per_channel[name] = {"mae": float(np.abs(residual).mean()),
                             "rmse": float(np.sqrt(np.square(residual).mean())),
                             "wape_percent": (float(100 * np.abs(residual).sum() / target_magnitude)
                                              if target_magnitude > np.finfo(float).eps else float("nan")),
                             "observed": int(observed.sum())}
    return {"mae": float(np.abs(error[mask]).mean()),
            "rmse": float(np.sqrt(np.square(error[mask]).mean())),
            "observed": int(mask.sum()), "per_channel": per_channel}


def forecast_chronos2(model, windows: WindowSet, *, device: str = "cpu",
                      batch_size: int = 8, channel_order=None) -> np.ndarray:
    """Predict each window jointly with all its channels, using only past values.

    ``channel_order`` feeds the channels to the model in another order and
    returns predictions in the original order. With a channel-index embedding
    the forecast can depend on this order, which the evaluation reports.
    """
    import math
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n_total = windows.context.shape[2]
    order = np.arange(n_total) if channel_order is None else np.asarray(channel_order, dtype=int)
    if sorted(order.tolist()) != list(range(n_total)):
        raise ValueError("channel_order must be a permutation of the channels")
    inverse = np.argsort(order)
    horizon = windows.target.shape[1]
    patches = math.ceil(horizon / model.chronos_config.output_patch_size)
    quantiles = list(model.chronos_config.quantiles)
    median = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))
    if abs(quantiles[median] - 0.5) > 1e-8:
        raise ValueError("Chronos model has no median quantile")
    model.eval()
    predictions = []
    n_channels = windows.context.shape[2]
    with torch.inference_mode():
        for start in range(0, len(windows.context), batch_size):
            block = windows.context[start:start+batch_size][:, :, order]
            n_windows = len(block)
            rows = torch.as_tensor(block.transpose(0, 2, 1).reshape(-1, block.shape[1]).copy(), device=device)
            group = torch.arange(n_windows, dtype=torch.long, device=device).repeat_interleave(n_channels)
            output = model(context=rows, group_ids=group, num_output_patches=patches)
            prediction = output.quantile_preds[:, median, :horizon].reshape(n_windows, n_channels, horizon)
            prediction = prediction.permute(0, 2, 1).cpu().numpy()
            predictions.append(prediction)
    return np.concatenate(predictions)[:, :, inverse]
