# Project instructions

This repository develops a benchmark for synthetic time-series mechanisms and
known explanation ground truths.

## Start here

Read `README.md`, `docs/research/problem_definition.md`, `ARCHITECTURE.md`, and
`REPRODUCIBILITY.md` before changing the design.

## Source code

Reusable logic belongs under `src/tsfmxai/`. Do not implement reusable library
behavior only inside notebooks or experiment directories. Add tests under
`tests/` for promoted code.

## Experiments

Exploratory work belongs under ignored `experiments_local/`. A result becomes
part of `experiments/` only after its configuration, provenance, explicit
seeds, metrics, scientific summary, and reproduction instructions are
recorded.

For an executed notebook experiment, keep that complete experiment record in
one clearly labelled cell inside the notebook. Save only the executed notebook
and do not create README, configuration, manifest, metrics, or summary sidecar
files that duplicate its contents. Sidecars are appropriate for non-notebook
experiments or when external artifacts require their own machine-readable
manifest.

## Scientific references

Use `references/library.bib` as the canonical bibliography. Verify claims
against the corresponding paper or reviewed summary, and record page-level
evidence when possible. Preserve matching basenames for paper PDFs and notes.

## Thesis

The thesis source lives under `report/`. Do not invent citations or manually
edit generated build artifacts.

## Before finishing

Run the relevant tests, inspect `git status`, and confirm that generated data,
local experiments, paper PDFs, and LaTeX build files remain ignored.
