import torch
from tsfmxai.pipelines import forecasting_budget as module


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def forward(self):
        return type('Output', (), {'loss': self.weight.square()})()


def test_time_budget_charges_preparation_and_saves(tmp_path):
    budget = module.ForecastingBudget(max_hours=1, preparation_seconds=3601)
    result = module.train_forecasting(Model(), [], budget, tmp_path, device='cpu')
    assert result['stop_reason'] == 'max_hours'
    assert result['updates'] == 0
    assert (tmp_path / 'latest.pt').exists()


def test_update_budget_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_inputs', lambda *args: {})
    batch = {key: torch.zeros(16, 96, 6) for key in ('context', 'future')}
    result = module.train_forecasting(Model(), [batch] * 5,
        module.ForecastingBudget(max_updates=4, stage_boundary=2), tmp_path, device='cpu')
    assert result['updates'] == 4
    assert result['stop_reason'] == 'max_updates'
    assert len((tmp_path / 'telemetry.jsonl').read_text().splitlines()) == 4
