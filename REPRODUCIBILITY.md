# Reproducibility

The current local environment uses Python 3.11.9. The committed project target
is Python 3.11.

Every generated benchmark artifact should eventually record:

- library and grammar versions;
- the complete configuration;
- global and derived random seeds;
- mechanism, trajectory, and window identifiers;
- numerical backend and hardware;
- model and dataset versions;
- the command used to reproduce the result.

Full synthetic datasets are generated artifacts and do not belong in Git.
Commit their configurations, seeds, manifests, small test samples, and final
scientific summaries instead. Local exploratory work belongs in
`experiments_local/`; curated evidence belongs in `experiments/`.

Executable setup, test, data-generation, benchmark, documentation, and report
commands will be added once the first library pipeline is implemented. Until
then, environment-specific dependencies remain documented with their local
experiments rather than being presented as project-wide dependencies.
