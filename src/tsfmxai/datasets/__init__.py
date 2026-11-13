"""Dataset, windowing, and batching interfaces."""

from tsfmxai.datasets.multiseries import (
    TrajectoryConfig, TrainingFormat, TrainingBatch, TrainingBatchStream,
    GenerationStats, generate_training_batches,
)

from tsfmxai.datasets.parallel_batch import (
    MechanismJobResult,
    ParallelGenerationConfig,
    SyntheticForecastBatch,
    assemble_diverse_batch,
    derive_job_seed,
    generate_mechanism_job,
    generate_mechanism_jobs,
)

__all__ = [
    "TrajectoryConfig",
    "TrainingFormat",
    "TrainingBatch",
    "TrainingBatchStream",
    "GenerationStats",
    "generate_training_batches",
    "MechanismJobResult",
    "ParallelGenerationConfig",
    "SyntheticForecastBatch",
    "assemble_diverse_batch",
    "derive_job_seed",
    "generate_mechanism_job",
    "generate_mechanism_jobs",
]
from tsfmxai.datasets.online import (
    CurriculumStage, OnlineBatchDataset, OnlineBatchPlan, make_dataloader,
)
