from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from tsfmxai.evaluation.real_world import (
    ETT_CHANNELS, WEATHER_CHANNELS, benchmark_test_period, load_ett, load_jena_weather,
    select_windows,
)


def write_ett(path: Path, hours: int) -> None:
    times = pd.date_range("2016-07-01", periods=hours, freq="h")
    frame = pd.DataFrame({"date": times.strftime("%Y-%m-%d %H:%M:%S")})
    for k, name in enumerate(ETT_CHANNELS):
        frame[name] = np.arange(hours) + k
    frame.to_csv(path, index=False)


def test_ett_loads_hourly_grid_and_standard_test_split(tmp_path: Path):
    path = tmp_path / "ETTh1.csv"
    write_ett(path, 20 * 30 * 24 + 5)
    frame = load_ett(path)
    assert tuple(frame.columns) == ETT_CHANNELS and frame.index.tz is not None
    first, last = benchmark_test_period("etth1", frame)
    assert first == frame.index[11520] and last == frame.index[14399]
    windows = select_windows(frame, period=(first, last), context_length=96, horizon=96, stride=24)
    assert all(first <= pd.Timestamp(o) for o in windows.origins)
    # The last target hour of the last window stays inside the split.
    last_origin = frame.index.get_loc(pd.Timestamp(windows.origins[-1]))
    assert frame.index[last_origin + 95] <= last
    # Context may come from before the split.
    assert windows.context[0, 0, 0] < 11520


def test_window_selection_needs_exactly_one_of_year_or_period(tmp_path: Path):
    path = tmp_path / "ETTh2.csv"
    write_ett(path, 400)
    frame = load_ett(path)
    with pytest.raises(ValueError):
        select_windows(frame, context_length=4, horizon=4)
    with pytest.raises(ValueError):
        select_windows(frame, year=2016, period=(frame.index[0], frame.index[-1]),
                       context_length=4, horizon=4)


def test_jena_weather_masks_sentinel_and_averages_hour_ending_readings(tmp_path: Path):
    times = pd.date_range("2020-01-01 00:10", periods=12, freq="10min")  # 00:10 .. 02:00
    header = '"Date Time",' + ",".join(f'"{c.replace("2", "²")}"' for c in WEATHER_CHANNELS)
    lines = [header]
    for i, time in enumerate(times):
        values = [float(i)] * len(WEATHER_CHANNELS)
        if i == 2:
            values[11] = -9999.0  # wind speed sentinel
        lines.append(time.strftime("%d.%m.%Y %H:%M:%S") + ", " + ", ".join(f"{v:.2f}" for v in values))
    half = len(lines) // 2 + 1
    paths = []
    for k, chunk in enumerate((lines[:half], [header] + lines[half:])):
        path = tmp_path / f"part{k}.zip"
        with ZipFile(path, "w") as zf:
            zf.writestr(f"part{k}.csv", ("\n".join(chunk) + "\n").encode("latin-1"))
        paths.append(path)
    frame = load_jena_weather(paths)
    assert tuple(frame.columns) == WEATHER_CHANNELS
    assert list(frame.index.strftime("%H:%M")) == ["01:00", "02:00"]
    assert frame.iloc[0, 0] == pytest.approx(np.mean(range(6)))
    assert frame.iloc[0, 11] == pytest.approx(np.mean([0, 1, 3, 4, 5]))
    assert frame.iloc[1, 0] == pytest.approx(np.mean(range(6, 12)))


def test_weather_test_split_is_last_fifth():
    index = pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")
    frame = pd.DataFrame({"x": np.arange(100.0)}, index=index)
    first, last = benchmark_test_period("weather", frame)
    assert first == index[80] and last == index[99]
    with pytest.raises(ValueError):
        benchmark_test_period("traffic", frame)
