"""Diagnostics must distinguish algebraic errors from output ordering."""

from tsfmxai.evaluation.canonical_errors import classify_prediction, structural_features, swap_children
from tsfmxai.generators.random_expression import SampledMechanism
from tsfmxai.representations.expression import Add, Lag, MechanismBinding, Multiply, Parameter
from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec, END_TERM, EOS


def make_example():
    x, y = Lag(0, 'x'), Lag(0, 'y')
    expression = Multiply(Add(x, y), Add(x, y))
    mechanism = SampledMechanism(expression, MechanismBinding({}, {'x': 1, 'y': 2}), 0, 0)
    codec = CanonicalCodec()
    return codec, codec.encode(mechanism)


def test_expansion_and_repeated_factors():
    codec, example = make_example()
    features = structural_features(example)
    assert features['operator_count'] == 3
    assert features['expanded_terms_before_merge'] == 4
    assert features['merged_occurrences'] == 1
    assert features['target_terms'] == 3
    assert features['power_terms'] == 2
    assert features['double_sum_products'] == 1
    assert classify_prediction(example, example['target'][1:], codec)['exact']


def test_sorting_recovery_preserves_factor_multiplicity():
    codec, example = make_example()
    x, y = codec.atom_token(0, 1), codec.atom_token(0, 2)
    reverse = [y, y, END_TERM, y, x, END_TERM, x, x, END_TERM, EOS]
    result = classify_prediction(example, reverse, codec)
    assert not result['exact']
    assert result['error_category'] == 'ordering_or_duplicate_only'
    wrong_power = [x, END_TERM, x, y, END_TERM, y, y, END_TERM, EOS]
    result = classify_prediction(example, wrong_power, codec)
    assert not result['support_correct_after_sorting']
    assert result['multiplicity_only_error']


def test_unclosed_term_is_not_ordering_error():
    codec, example = make_example()
    result = classify_prediction(example, example['target'][1:-1] + [codec.atom_token(0, 1), EOS], codec)
    assert result['error_category'] == 'incomplete_syntax'
    assert not result['support_correct_after_sorting']


def test_two_commutative_swaps_restore_input():
    codec = CanonicalCodec()
    mechanism = SampledMechanism(Add(Parameter('a'), Multiply(Lag(0, 'x'), Lag(0, 'y'))),
                                MechanismBinding({'a': 2.}, {'x': 1, 'y': 2}), 0, 0)
    example = codec.encode(mechanism)
    swapped = swap_children(example)
    restored = swap_children(swapped)
    for key in ('prefix_ids', 'prefix_values', 'tree_ids', 'tree_values', 'left_children', 'right_children'):
        assert restored[key] == example[key]
    assert swapped['target'] == example['target']
    assert structural_features(example) == structural_features(swapped)
