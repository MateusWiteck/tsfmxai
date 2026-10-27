"""Reproduce and verify the sampled examples in the Generation Overview.

Run from the repository root: python scripts/check_generation_overview.py
This is a report-specific check; it leaves the library and manuscripts unchanged.
"""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfmxai.generators.random_expression import RandomTreeConfig, sample_random_mechanism
from tsfmxai.generators.contraction import ContractionTreeConfig, build_contractive_mechanism
from tsfmxai.generators.shao_wu import (
    ShaoWuConfig, build_shao_wu_mechanism, saturate_product_lags, saturate_product_outputs,
)
from tsfmxai.generators.heaps import build_heaps_var_mechanism
from tsfmxai.generators.trajectory import evaluate_bound_expression


def main():
    base = RandomTreeConfig(minimum_operators=0, maximum_operators=0,
        time_probability=0., parameter_probability=0., lag_probability=1.,
        minimum_lag=1, maximum_lag=1, minimum_parameter_magnitude=.5,
        maximum_parameter_magnitude=.5, add_probability=.65,
        coefficient_gate_lags=False)
    # Expected functions use a newest-first history z: z[0] = x(t-1).
    # Configurations deliberately restrict the prior to produce readable cases.
    cases = [
        ("R1", base, 0, lambda z:z[0], lambda z:np.tanh(z[0]), [1.], 0.,
         (.855, .76, .76)),
        ("R2", replace(base, minimum_operators=1, maximum_operators=1,
         parameter_probability=1., add_probability=0., minimum_parameter_magnitude=2.,
         maximum_parameter_magnitude=2.), 0, lambda z:2*z[0], None, [2.], 0.,
         (.4275, .38, .38)),
        ("R3", replace(base, minimum_operators=1, maximum_operators=1,
         parameter_probability=1., add_probability=1.), 2, lambda z:z[0]-.5,
         None, [1.], -.5, (.95*(1-np.sqrt(3)*.05)/1.5, .76, .76)),
        ("R4", replace(base, minimum_operators=1, maximum_operators=1,
         maximum_lag=3, add_probability=1.), 0, lambda z:z[0]+z[2], None,
         [1.,0.,1.], 0., (.855/(1+1.25**2), .38, .38)),
        ("R5", replace(base, minimum_lag=12, maximum_lag=12), 0,
         lambda z:z[11], None, [0.]*11+[1.], 0., (.855/1.25**11, .76, .76)),
        ("R6", replace(base, minimum_operators=1, maximum_operators=1,
         maximum_lag=2, add_probability=0.), 0, lambda z:z[0]*z[1],
         lambda z:np.tanh(z[0])*np.tanh(z[1]), None, 0., (.38,.38,.95)),
        ("R7", replace(base, minimum_operators=1, maximum_operators=1,
         add_probability=0.), 0, lambda z:z[0]**2, lambda z:np.tanh(z[0])**2,
         None, 0., (.4275,.38,.95)),
        ("R8", replace(base, minimum_operators=3, maximum_operators=3,
         maximum_lag=3, add_probability=0.), 5, lambda z:z[0]**2*z[1]*z[2],
         lambda z:np.tanh(z[0])**2*np.tanh(z[1])*np.tanh(z[2]), None, 0.,
         (.855/(2+1.25+1.25**2), .19, .95)),
        ("R9", replace(base, minimum_operators=3, maximum_operators=3,
         maximum_lag=2, parameter_probability=.4, add_probability=.5), 1438,
         lambda z:.5*z[0]+z[0]*z[1],
         lambda z:.5*z[0]+np.tanh(z[0])*np.tanh(z[1]), None, 0.,
         (.855/(1.5+1.25), .304, .95)),
    ]
    rng = np.random.default_rng(20261027)
    histories = rng.uniform(-1,1,(32,12))
    for name, cfg, seed, original, factor, free, intercept, scales in cases:
        sampled = sample_random_mechanism(cfg, seed=seed)
        j = build_contractive_mechanism(sampled.expression, sampled.binding,
            ContractionTreeConfig(innovation_bound=np.sqrt(3)*.05))
        f = build_shao_wu_mechanism((saturate_product_lags(sampled.expression),),
            (sampled.binding,), ShaoWuConfig(lag_transform="product_lags"))
        o = build_shao_wu_mechanism((saturate_product_outputs(sampled.expression),),
            (sampled.binding,), ShaoWuConfig(lag_transform="product_output"))
        np.testing.assert_allclose([j.scale,f.scale,o.scale], scales, rtol=1e-13)
        assert j.certificate.holds and f.certificate.holds and o.certificate.holds
        for z in histories:
            evaluate = lambda e,b:evaluate_bound_expression(e,b,time=12,histories=z[::-1][None,:])
            np.testing.assert_allclose(evaluate(sampled.expression,sampled.binding),original(z),atol=1e-14)
            np.testing.assert_allclose(evaluate(j.expression,j.binding),scales[0]*original(z),atol=1e-14)
            expected_f = original(z) if free is not None else factor(z)
            expected_o = original(z) if free is not None else (
                .5*z[0]+np.tanh(z[0]*z[1]) if name=="R9" else np.tanh(original(z)))
            np.testing.assert_allclose(evaluate(f.expressions[0],f.bindings[0]),scales[1]*expected_f,atol=1e-14)
            np.testing.assert_allclose(evaluate(o.expressions[0],o.bindings[0]),scales[2]*expected_o,atol=1e-14)
        print(f"{name}: seed={seed}; input={sampled.formula}; scales={j.scale:.9g}, {f.scale:.9g}, {o.scale:.9g}")
        if free is not None:
            # Explicit illustrative convention: input AR coefficients become U_l.
            # Heaps itself receives free matrices, not an AST to be repaired.
            u = np.asarray(free)[:,None,None]
            h = build_heaps_var_mechanism(u,np.array([[.05**2]]))
            coefficients = np.asarray(h.coefficients)[:,0,0]
            mean = intercept/(1-coefficients.sum())
            h = build_heaps_var_mechanism(u,np.array([[.05**2]]),stationary_mean=np.array([mean]))
            assert h.certificate.holds
            expected = {
                "R1":[1/np.sqrt(2)], "R2":[2/np.sqrt(5)], "R3":[1/np.sqrt(2)],
                "R4":[1/np.sqrt(2),-.5,1/np.sqrt(2)], "R5":[0.]*11+[1/np.sqrt(2)],
            }[name]
            np.testing.assert_allclose(coefficients,expected,atol=1e-12)
            for z in histories:
                actual = evaluate_bound_expression(h.expressions[0],h.bindings[0],time=12,histories=z[::-1][None,:])
                np.testing.assert_allclose(actual,intercept+coefficients@z[:len(coefficients)],atol=1e-12)
            print("    Heaps:", h.formulas[0])
    print("Verified nine sampled ASTs, 27 transformations, and five matched Heaps constructions.")


if __name__ == "__main__":
    main()
