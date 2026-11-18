from pathlib import Path
from zipfile import ZipFile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tsfmxai.evaluation.real_world import (
    load_brazil_energy, load_ibirapuera, last_observed_forecast,
    observed_metrics, select_windows, forecast_chronos2, WindowSet,
)


def test_missing_air_targets_are_masked_and_never_imputed(tmp_path: Path):
    path = tmp_path / "air.tsv"
    rows = ["/* metadata */", "Date/Time\tHeight [m]\tO3 [ppbv]\tNO [ppbv]\tNO2 [ppbv]\tCO [ppmv]\tPM2.5 [µg/m**3]"]
    times = pd.date_range("2019-01-01", periods=12, freq="h")
    for i, time in enumerate(times):
        no2 = "" if i == 7 else str(i + 3)
        rows.append(f"{time:%Y-%m-%dT%H:%M}\t2\t{i+1}\t{i+2}\t{no2}\t{i+4}\t{i+5}")
    path.write_text("\n".join(rows), encoding="utf-8")
    frame = load_ibirapuera(path)
    windows = select_windows(frame, year=2019, context_length=4, horizon=4, stride=1,
                             coverage=0.75, limit=4)
    assert frame.index.tz is not None
    assert len(windows.origins) == 4
    assert any(np.isnan(windows.target).ravel())
    naive = last_observed_forecast(windows.context, 4)
    result = observed_metrics(naive, windows.target, windows.channels)
    assert result["observed"] < 4 * 4 * 5
    assert np.isfinite(result["mae"])


def test_wape_uses_observed_target_magnitudes_and_handles_zero_denominator():
    target = np.array([[[0.0, 0.0], [2.0, 0.0], [np.nan, 0.0]]])
    prediction = np.array([[[1.0, 1.0], [1.0, 1.0], [999.0, 1.0]]])
    result = observed_metrics(prediction, target, ("positive", "zero"))
    assert result["per_channel"]["positive"]["wape_percent"] == 100.0
    assert np.isnan(result["per_channel"]["zero"]["wape_percent"])
    assert result["per_channel"]["positive"]["observed"] == 2


def test_energy_join_converts_brasilia_time_to_utc(tmp_path: Path):
    path = tmp_path / "energy.zip"
    local = pd.date_range("2019-01-01", periods=6, freq="h")
    utc = local + pd.Timedelta(hours=3)
    def csv(frame, sep):
        return frame.to_csv(index=True, sep=sep).encode()
    load = pd.DataFrame({"time": local, "state": ["BA"] * 6, "value": np.arange(6) + 10})
    load.insert(0, "Unnamed: 0", np.arange(6))
    wind = pd.DataFrame({"BA": np.arange(6) / 10}, index=utc)
    solar = pd.DataFrame({"BA": np.arange(6) / 20}, index=utc)
    hydro = pd.DataFrame({"BA": np.arange(6) + 100}, index=local)
    with ZipFile(path, "w") as zf:
        zf.writestr("x/distribute_by_consumer/Hourly_electricity_demand_per_state_2019.csv",
                    load.to_csv(index=False))
        zf.writestr("x/onshore/EnDAT_on_wind_per_unit_generation_weather_year_2019.csv", csv(wind, ";"))
        zf.writestr("x/PV/EnDAT_solar_pv_per_unit_generation_weather_year_2019.csv", csv(solar, ";"))
        zf.writestr("x/by_hydropower_plants_operation/ONS_state_inflow_distribute_by_hydropower_plants_operation_reference_year_2019.csv", csv(hydro, ","))
    frame = load_brazil_energy(path, state="BA", year=2019)
    assert len(frame) == 6
    assert frame.index[0] == pd.Timestamp("2019-01-01T03:00:00Z")
    assert frame.iloc[0].to_list() == [10, 0, 0, 100]


def test_batched_chronos_groups_do_not_mix_windows():
    import torch

    class GroupMeanModel:
        chronos_config = SimpleNamespace(output_patch_size=2, quantiles=[0.5])

        def eval(self):
            return self

        def __call__(self, *, context, group_ids, num_output_patches):
            result = torch.empty((len(context), 1, num_output_patches * 2))
            for group in torch.unique(group_ids):
                rows = group_ids == group
                result[rows] = context[rows].mean()
            return SimpleNamespace(quantile_preds=result)

    contexts = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    windows = WindowSet(contexts, np.zeros((4, 4, 2), dtype=np.float32),
                        tuple(str(i) for i in range(4)), ("a", "b"), 4)
    model = GroupMeanModel()
    np.testing.assert_allclose(forecast_chronos2(model, windows, batch_size=3),
                               forecast_chronos2(model, windows, batch_size=1))


def test_origin_future_must_stay_inside_requested_year():
    frame = pd.DataFrame({"value": np.arange(240, dtype=float)},
                         index=pd.date_range("2018-12-28", periods=240, freq="h", tz="UTC"))
    windows = select_windows(frame, year=2018, context_length=24, horizon=48,
                             stride=24, limit=None)
    assert all(pd.Timestamp(origin).year == 2018 for origin in windows.origins)
    assert all(pd.Timestamp(origin).year == 2018 and
               (pd.Timestamp(origin) + pd.Timedelta(hours=47)).year == 2018
               for origin in windows.origins)


def test_dlinear_fit_uses_earlier_train_windows_only():
    from tsfmxai.baselines.dlinear import fit_dlinear, predict_dlinear

    base = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    train = WindowSet(base, base + 1, ("a", "b"), ("x", "y"), 2)
    validation = WindowSet(base + 2, base + 3, ("c", "d"), ("x", "y"), 2)
    fit = fit_dlinear(train, validation, seed=7, epochs=2, batch_size=2, patience=2)
    np.testing.assert_allclose(fit.mean, np.concatenate((base, base + 1)).mean((0, 1)))
    prediction = predict_dlinear(fit, validation)
    assert prediction.shape == validation.target.shape
    assert np.isfinite(prediction).all()
