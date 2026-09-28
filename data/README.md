# Data

- `external/`: immutable third-party inputs;
- `raw/`: generated or collected raw observations;
- `interim/`: intermediate transformations;
- `processed/`: model-ready datasets;
- `samples/`: small versioned examples for tests and documentation.

The first four stages are ignored by Git. Their configurations, seeds, and
manifests provide the reproducible record.
