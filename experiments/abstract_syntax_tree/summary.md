# Scientific summary

The restricted parser converted the supplied recurrence into a nine-node AST
and reconstructed the same canonical formula from that tree. Starting from
`x_0(0) = 0.5`, the AST generated 40 subsequent values. Re-evaluating the
retained mechanism at every generated time produced a maximum absolute
recurrence residual of `0.0`, and the canonical parse-format-parse round trip
passed.

The public `FormulaMechanism` object supplies the formula-writing guide,
retains the AST and generated series, reconstructs the mathematical formula,
and plots its own observations. An invalid formula returns the writing guide
alongside the validation error.

This validates the first deterministic, univariate path from a user formula to
an inspectable mechanism and an executable trajectory. The next experiment can
add JSON serialization and multiple hand-authored formulas before tree
sampling is introduced.
