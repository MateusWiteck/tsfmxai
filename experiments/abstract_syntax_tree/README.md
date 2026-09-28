# Formula-to-AST prototype

## Research question

Can a recurrence supplied as text be converted into a validated canonical AST,
reconstructed as a readable formula, and executed to generate a time series
without losing the mechanism's structure?

## Hypothesis

A restricted grammar containing constants, time, lag references, addition, and
multiplication is sufficient to represent and execute the initial reference
mechanism while retaining an inspectable tree.

## Experiment

The executed notebook
`1_formula_to_ast_and_series.ipynb` implements the complete path:

```text
writing guide -> user formula -> FormulaMechanism -> validated AST
                                      |                  |
                                      v                  v
                                 plot method      recursive series
```

The reference recurrence is:

```text
x_0(t) = 0.7 * x_0(t-1) + 0.2 * t + 1.0
```

Users can call `FormulaMechanism.instructions()` before writing a formula. A
constructed object exposes `generate(...)`, `plot(...)`, `tree`,
`canonical_formula`, and `converted_formula`. Formula-validation errors include
the same writing guide.

## Reproduction

From the repository root, execute the notebook with the Python 3.11 project
environment. The notebook records the formula, seed, initial condition, number
of generated steps, outputs, and validation residual.

The executable `nbclient` command is recorded in `manifest.json`. It reads the
notebook, runs every cell with the project kernel, and saves the outputs back to
the same notebook.
