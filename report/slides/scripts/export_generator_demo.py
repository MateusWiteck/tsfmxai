"""Export one real mechanism for the "zoom" slides (generator, equation head).

Read-only use of the project library: nothing outside report/slides/ is written.
Run from the repository root:

    .venv/Scripts/python -I report/slides/scripts/export_generator_demo.py

Every number shown on the generator/equation-head slides comes from this file:
the sampled AST, its binding, the tanh protection, the certified scale, the
canonical polynomial, the codec tokens and one simulated trajectory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from tsfmxai.generators import CombinatorialTreeConfig, CombinatorialTreeSampler  # noqa: E402
from tsfmxai.generators.random_expression import SampledMechanism  # noqa: E402
from tsfmxai.generators.shao_wu import (  # noqa: E402
    ShaoWuConfig,
    build_shao_wu_mechanism,
    saturate_product_lags,
    sample_shao_wu_trajectories,
)
from tsfmxai.representations import canonicalize_polynomial  # noqa: E402
from tsfmxai.representations.expression import (  # noqa: E402
    Add,
    Lag,
    MechanismBinding,
    Multiply,
    Parameter,
    Tanh,
    lag_slots,
    operator_count,
    parameter_slots,
    to_bound_formula,
)
from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec  # noqa: E402

CONFIG = json.loads((ROOT / "configs" / "canonicalization_5m_raw_tanh.json").read_text())["generator"]
DEMO_INDEX, DEMO_SEED = 1383, 2383  # found by the search in find_demo()
TRAJ_STEPS = 120


def find_demo(sampler: CombinatorialTreeSampler) -> SampledMechanism:
    """A readable AST: 5 operators, a product over a sum, lags <= 8, 3-5 terms."""
    for i in range(4000):
        m = sampler.sample(seed=1000 + i, sample_index=i)
        if operator_count(m.expression) != 5:
            continue
        nodes = list(walk(m.expression))
        if not any(isinstance(x, Multiply) and (isinstance(x.left, Add) or isinstance(x.right, Add)) for x in nodes):
            continue
        lags = [m.binding.lag_values[x.slot] for x in nodes if isinstance(x, Lag)]
        if len(lags) < 3 or max(lags) > 8:
            continue
        if not 3 <= len(canonicalize_polynomial(m.expression, m.binding, max_degree=4).terms) <= 5:
            continue
        return m
    raise RuntimeError("no demo found")


def walk(e):
    yield e
    for c in (getattr(e, "left", None), getattr(e, "right", None), getattr(e, "operand", None)):
        if c is not None:
            yield from walk(c)


def tree_json(e, b: MechanismBinding):
    """Nested dict: t = add | mul | par | lag | tanh."""
    if isinstance(e, Parameter):
        return {"t": "par", "slot": e.slot, "v": float(b.parameter_values[e.slot])}
    if isinstance(e, Lag):
        return {"t": "lag", "slot": e.slot, "lag": int(b.lag_values[e.slot]), "series": e.variable}
    if isinstance(e, Tanh):
        return {"t": "tanh", "k": [tree_json(e.operand, b)]}
    kind = "add" if isinstance(e, Add) else "mul"
    return {"t": kind, "k": [tree_json(e.left, b), tree_json(e.right, b)]}


def poly_json(p):
    return [
        {"atoms": [list(a) if len(a) == 3 else [a[0], a[1], "raw"] for a in t.monomial],
         "coef": float(t.coefficient)}
        for t in p.terms
    ]


def fmt_atom(a):
    return f"tanh(x[t-{a[1]}])" if len(a) == 3 else f"x[t-{a[1]}]"


def main() -> None:
    cfg = CombinatorialTreeConfig(**CONFIG)
    sampler = CombinatorialTreeSampler(cfg)
    sampled = sampler.sample(seed=DEMO_SEED, sample_index=DEMO_INDEX)
    assert operator_count(sampled.expression) == 5, "demo index drifted; rerun find_demo"
    b = sampled.binding

    raw_poly = canonicalize_polynomial(sampled.expression, b, max_degree=4)
    protected_expr = saturate_product_lags(sampled.expression)
    prot_poly = canonicalize_polynomial(protected_expr, b, max_degree=4)

    # Three different trees with the same function (distributivity / commutativity).
    c_slot = next(s for s in b.parameter_values)
    lag_nodes = [x for x in walk(sampled.expression) if isinstance(x, Lag)]
    # Rebuild from the demo's own parts: x2*(x8 + c) + (x3 + c') + x6
    by_lag = {b.lag_values[x.slot]: x for x in lag_nodes}
    pars = [x for x in walk(sampled.expression) if isinstance(x, Parameter)]
    p_big = next(p for p in pars if abs(b.parameter_values[p.slot] - 0.592136) < 1e-6)
    x2, x8 = by_lag[2], by_lag[8]
    same_a = Multiply(x2, Add(x8, p_big))
    same_b = Add(Multiply(x2, x8), Multiply(p_big, x2))
    same_c = Add(Multiply(x8, x2), Multiply(x2, p_big))
    def sub(t):
        return MechanismBinding({s: b.parameter_values[s] for s in parameter_slots(t)},
                                {s: b.lag_values[s] for s in lag_slots(t)})

    keys = {canonicalize_polynomial(t, sub(t), max_degree=4).canonical_key for t in (same_a, same_b, same_c)}
    assert len(keys) == 1, "equivalent trees must share one canonical key"
    same = [tree_json(t, b) for t in (same_a, same_b, same_c)]
    same_poly = poly_json(canonicalize_polynomial(same_a, sub(same_a), max_degree=4))

    # Certified scale (selective tanh = product_lags, as in the canonicalization corpus).
    sw_cfg = ShaoWuConfig()
    mech = build_shao_wu_mechanism((protected_expr,), (b,), sw_cfg, seed=DEMO_SEED, sample_index=DEMO_INDEX)
    final_expr, final_b = mech.expressions[0], mech.bindings[0]
    final_poly = canonicalize_polynomial(final_expr, final_b, max_degree=4)
    traj = sample_shao_wu_trajectories(mech, trajectory_count=1, generated_steps=TRAJ_STEPS, root_seed=7)[0, 0]
    order = mech.order

    # Codec view of the *selected* (protected) AST: this is the head's training pair.
    codec = CanonicalCodec(1, cfg.minimum_lag, cfg.maximum_lag, cfg.max_degree)
    prot_mech = SampledMechanism(protected_expr, b, DEMO_SEED, DEMO_INDEX)
    ex = codec.encode(prot_mech)
    names = {1: "ADD", 2: "MUL", 3: "PAR", 4: "TANH"}
    prefix = []
    for tok, val in zip(ex["prefix_ids"], ex["prefix_values"]):
        if tok in names:
            prefix.append({"tok": names[tok], "v": float(val) if tok == 3 else None})
        else:
            a = codec.token_atom(tok)
            prefix.append({"tok": f"LAG{a[1]}", "v": None, "atom": list(a)})
    tgt = []
    for tok, cval in zip(ex["target"], ex["coefficient_targets"]):
        if tok == 1:
            tgt.append({"tok": "BOS"})
        elif tok == 2:
            tgt.append({"tok": "EOS"})
        elif tok == 3:
            tgt.append({"tok": "END_TERM", "coef": float(cval)})
        else:
            a = codec.token_atom(tok)
            tgt.append({"tok": ("tanh " if len(a) == 3 else "") + f"x[t-{a[1]}]", "atom": list(a)})

    data = {
        "config": CONFIG,
        "seed": DEMO_SEED,
        "index": DEMO_INDEX,
        "operators": operator_count(sampled.expression),
        "raw_tree": tree_json(sampled.expression, b),
        "protected_tree": tree_json(protected_expr, b),
        "raw_formula": sampled.formula,
        "protected_formula": to_bound_formula(protected_expr, b),
        "final_formula": mech.formulas[0],
        "scale": float(mech.scale),
        "certificate": {"total_gain": float(mech.certificate.total_gain), "holds": bool(mech.certificate.holds),
                        "lag_gains": [float(g) for g in mech.certificate.lag_gains]},
        "target_gain": sw_cfg.target_gain, "safety": sw_cfg.safety_factor,
        "innovation_std": float(mech.innovation_std), "initial_radius": float(mech.initial_radius),
        "order": int(order),
        "trajectory": [float(v) for v in traj],
        "history_len": int(order),
        "raw_poly": poly_json(raw_poly),
        "protected_poly": poly_json(prot_poly),
        "final_poly": poly_json(final_poly),
        "same_trees": same,
        "same_poly": same_poly,
        "prefix": prefix,
        "target": tgt,
        "vocab_size": int(codec.vocab_size),
        "structures_total": str(sampler.total_structures),
        "size_counts": {str(k): str(v) for k, v in sampler.size_counts().items()},
    }
    out = ROOT / "report" / "slides" / "data" / "generator_demo.js"
    out.write_text("window.GEN_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("wrote", out.name, "scale", round(mech.scale, 4), "order", order, "terms raw/prot/final",
          len(raw_poly.terms), len(prot_poly.terms), len(final_poly.terms))
    print("raw:", sampled.formula)
    print("prot:", data["protected_formula"])
    print("prefix:", [p["tok"] for p in prefix])
    print("target:", [t["tok"] for t in tgt])
    print("traj head:", [round(float(v), 3) for v in traj[:5]], "std", round(float(traj.std()), 3))


if __name__ == "__main__":
    main()
