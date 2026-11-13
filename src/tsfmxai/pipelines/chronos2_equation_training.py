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

    def __post_init__(self):
        if not self.model_id or not self.revision:
            raise ValueError('Pin model_id and immutable revision')
        if self.skeleton_dimension < 1 or self.output_patch_size < 1:
            raise ValueError('dimensions must be positive')


def load_equation_model(spec: Chronos2TrainingSpec, *, device: str | torch.device):
    """Load the pinned Chronos-2 checkpoint and attach supervised auxiliary heads."""
    try:
        from chronos.chronos2.pipeline import Chronos2Pipeline
    except ImportError as error:
        raise ImportError("Install the project Chronos-2 checkout before loading this adapter") from error
    pipeline = Chronos2Pipeline.from_pretrained(spec.model_id, revision=spec.revision)
    model = Chronos2EquationModel(pipeline.model, skeleton_dimension=spec.skeleton_dimension,
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
