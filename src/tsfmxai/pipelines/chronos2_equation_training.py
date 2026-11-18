"""Build and load an equation-aware Chronos-2 pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from tsfmxai.models.chronos2_equation import Chronos2EquationModel, equation_training_loss

@dataclass(frozen=True)
class Chronos2TrainingSpec:
    model_id: str
    revision: str
    skeleton_dimension: int
    output_patch_size: int
    embedding_weight: float = 1.0
    noise_weight: float = 1.0
    freeze_forecaster: bool = False
    initialization: str = 'random'
    initialization_seed: int = 20261112

    def __post_init__(self):
        if not self.model_id or not self.revision:
            raise ValueError('Pin model_id and immutable revision')
        if self.skeleton_dimension < 1 or self.output_patch_size < 1:
            raise ValueError('dimensions must be positive')
        if self.initialization not in ('random', 'pretrained'):
            raise ValueError('initialization must be random or pretrained')
        if self.initialization == 'random' and self.freeze_forecaster:
            raise ValueError('Cannot freeze a randomly initialized forecaster')


def build_forecaster(spec: Chronos2TrainingSpec):
    """Construct on CPU; random mode downloads configuration only, never weights."""
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(spec.initialization_seed)
        if spec.initialization == 'random':
            config = Chronos2CoreConfig.from_pretrained(spec.model_id, revision=spec.revision)
            model = Chronos2Model(config)
        else:
            model = Chronos2Model.from_pretrained(spec.model_id, revision=spec.revision)
    if model.chronos_config.output_patch_size != spec.output_patch_size:
        raise ValueError('Configured output_patch_size differs from Chronos architecture')
    return model


def load_equation_model(spec: Chronos2TrainingSpec, *, device: str | torch.device):
    """Construct Chronos-2 and auxiliary heads with reproducible initialization."""
    forecaster = build_forecaster(spec)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(spec.initialization_seed + 1)
        model = Chronos2EquationModel(forecaster, skeleton_dimension=spec.skeleton_dimension,
                                      freeze_forecaster=spec.freeze_forecaster)
    return model.to(device)


def training_step(model: Chronos2EquationModel, batch: dict[str, torch.Tensor], spec: Chronos2TrainingSpec):
    output = model(batch)
    loss, metrics = equation_training_loss(
        output, batch, embedding_weight=spec.embedding_weight, noise_weight=spec.noise_weight)
    return loss, metrics, output


def move_tensor_batch(batch, device):
    """Move tensor fields to device; keep Python metadata on CPU."""
    return {key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()}
