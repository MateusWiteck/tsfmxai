"""Scratch construction must not load weights and must permit real updates."""
import pytest
import torch

chronos_config = pytest.importorskip('chronos.chronos2.config')
chronos_model = pytest.importorskip('chronos.chronos2.model')

from tsfmxai.pipelines.chronos2_equation_training import (
    Chronos2TrainingSpec, build_forecaster,
)


def test_scratch_initialization_and_optimizer_update(monkeypatch):
    def configuration(*args, **kwargs):
        return chronos_config.Chronos2CoreConfig(
            d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2,
            dropout_rate=0., chronos_config=dict(
                context_length=16, input_patch_size=4, output_patch_size=4,
                input_patch_stride=4, quantiles=[0.1, 0.5, 0.9],
                max_output_patches=1, use_reg_token=True,
            ),
        )

    def forbidden(*args, **kwargs):
        raise AssertionError('Scratch mode must never load pretrained weights')

    monkeypatch.setattr(chronos_config.Chronos2CoreConfig, 'from_pretrained', configuration)
    monkeypatch.setattr(chronos_model.Chronos2Model, 'from_pretrained', forbidden)
    spec = Chronos2TrainingSpec('configuration-only', 'pinned-revision', 8, 4)
    before_rng = torch.get_rng_state().clone()
    first = build_forecaster(spec)
    assert torch.equal(before_rng, torch.get_rng_state())
    second = build_forecaster(spec)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    optimizer = torch.optim.AdamW(first.parameters(), lr=1e-3)
    initial = [p.detach().clone() for p in first.parameters()]
    output = first(context=torch.randn(2, 16), group_ids=torch.tensor([0, 0]),
                   future_covariates=torch.full((2, 4), float('nan')),
                   num_output_patches=1, future_target=torch.randn(2, 4))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(initial, first.parameters()))


def test_freezing_random_forecaster_is_rejected():
    with pytest.raises(ValueError, match='freeze'):
        Chronos2TrainingSpec('config', 'revision', 8, 4, freeze_forecaster=True)
