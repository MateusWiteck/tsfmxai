"""Dataset, windowing, and batching interfaces."""

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
    "MechanismJobResult",
    "ParallelGenerationConfig",
    "SyntheticForecastBatch",
    "assemble_diverse_batch",
    "derive_job_seed",
    "generate_mechanism_job",
    "generate_mechanism_jobs",
]
