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
part of `experiments/` only after it has a configuration, manifest, metrics,
summary, and reproduction instructions.

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
