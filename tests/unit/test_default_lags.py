"""Keep the generation and canonical-target defaults compatible."""

from tsfmxai.generators.random_expression import RandomTreeConfig
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.contraction import ContractionTreeConfig
from tsfmxai.generators.shao_wu import ShaoWuConfig
from tsfmxai.generators.heaps import HeapsVARConfig
from tsfmxai.evaluation.initialization import InitializationStudyConfig
from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec


def test_generator_and_codec_default_lag_range_agree():
    configs = [RandomTreeConfig(), CombinatorialTreeConfig(),
               ContractionTreeConfig().tree_config, ShaoWuConfig().tree_config,
               InitializationStudyConfig()]
    assert all(config.maximum_lag == 16 for config in configs)
    assert HeapsVARConfig().maximum_order == 16
    codec = CanonicalCodec()
    for transformation in ('raw', 'tanh'):
        atom = codec.token_atom(codec.atom_token(0, 16, transformation))
        assert atom[:2] == (0, 16)
