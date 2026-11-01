from copy import deepcopy
import math

import pytest

torch = pytest.importorskip('torch')

from tsfmxai.generators import SampledMechanism
from tsfmxai.representations import Add, Multiply, Parameter, Lag, MechanismBinding
from tsfmxai.symbolic_regression.canonical_data import (
    BOS, EOS, END_TERM, CanonicalCodec, generate_canonical_dataset, skeleton_digest,
    split_for_skeleton,
)
from tsfmxai.symbolic_regression.canonical_models import (
    CanonicalEncoderDecoder, collate_canonical, supervised_loss,
)
from tsfmxai.symbolic_regression.canonical_training import (
    evaluate_greedy, prepare_dataset, train_variant, finish_study, read_status,
)


def sample_formula():
    return SampledMechanism(
        Multiply(Add(Lag(0, 'a'), Parameter('p')), Add(Lag(1, 'b'), Parameter('q'))),
        MechanismBinding({'p': -2., 'q': 3.}, {'a': 1, 'b': 2}), 1, 0,
    )


def small_config():
    return {
        'generator': {'minimum_operators': 1, 'maximum_operators': 3, 'num_series': 1,
                      'minimum_lag': 1, 'maximum_lag': 3, 'max_degree': 3},
        'dataset_sizes': {'train': 32, 'validation': 8, 'test': 8},
        'data_seed': 13, 'split_seed': 31, 'training_seed': 44,
        'max_decode_steps': 24, 'batch_size': 16, 'epochs': 2, 'minimum_epochs': 1,
        'patience': 2, 'learning_rate': .001, 'weight_decay': .0001,
        'coefficient_weight': 1., 'gradient_clip': 1., 'validation_greedy_size': 4,
        'coefficient_atol': .01, 'coefficient_rtol': .01, 'min_delta': .0001,
        'model': {'dimension': 16, 'heads': 2, 'encoder_layers': 1, 'dropout': 0., 'max_nodes': 7},
        'variants': [{'name': 'tiny_tree_joint', 'encoder': 'tree_lstm', 'coefficients': True}],
    }


def test_roundtrip_sparse_targets_coefficients_and_tree_edges():
    codec = CanonicalCodec(num_series=2)
    example = codec.encode(sample_formula())
    decoded = codec.decode(example['target'][1:], example['coefficient_targets'][1:])
    assert decoded['valid']
    assert decoded['skeleton'] == ((), ((0, 1),), ((1, 2),), ((0, 1), (1, 2)))
    assert decoded['coefficients'] == (-6., 3., -2., 1.)
    assert all(child < i for i, child in enumerate(example['left_children']))
    assert all(child < i for i, child in enumerate(example['right_children']))
    assert example['prefix_values'].count(-2.) == 1


def test_zero_polynomial_and_invalid_output_detection():
    codec = CanonicalCodec()
    zero = SampledMechanism(Add(Parameter('a'), Parameter('b')),
                           MechanismBinding({'a': 1., 'b': -1.}, {}), 0, 0)
    assert codec.encode(zero)['target'] == [BOS, EOS]
    assert codec.decode([EOS])['valid']
    atom = codec.atom_token(0, 1)
    assert not codec.decode([atom, EOS])['valid']
    assert not codec.decode([atom, END_TERM, atom, END_TERM, EOS])['valid']
    assert not codec.decode([atom] * 5 + [END_TERM, EOS])['valid']
    assert not codec.decode([atom, END_TERM])['valid']
    assert not codec.decode([BOS, EOS])['valid']


def test_dataset_is_reproducible_and_split_by_whole_skeleton():
    config = small_config()
    first, summary, _ = generate_canonical_dataset(config)
    second, other, _ = generate_canonical_dataset(config)
    assert summary == other and first == second
    sets = {name: {skeleton_digest(e['skeleton']) for e in ds} for name, ds in first.items()}
    assert not (sets['train'] & sets['test'] or sets['train'] & sets['validation']
                or sets['test'] & sets['validation'])
    for name, ds in first.items():
        assert all(split_for_skeleton(e['skeleton'], config['split_seed']) == name for e in ds)


@pytest.mark.parametrize('encoder', ['transformer', 'tree_lstm'])
@pytest.mark.parametrize('joint', [False, True])
def test_forward_backward_and_free_decoding_ignore_targets(encoder, joint):
    torch.manual_seed(3)
    codec = CanonicalCodec(num_series=2)
    batch = collate_canonical([codec.encode(sample_formula())] * 2)
    model = CanonicalEncoderDecoder(codec.vocab_size, encoder, joint, dimension=16,
                                    heads=2, encoder_layers=1, dropout=0., max_nodes=7)
    model.eval()
    loss, _ = supervised_loss(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    original = model.generate(batch, max_steps=8)[0]
    modified = deepcopy(batch)
    modified['target'].fill_(0)
    modified['coefficient_targets'].fill_(123.)
    assert torch.equal(original, model.generate(modified, max_steps=8)[0])
    original_logits = model(batch)[0]
    changed_prefix = batch['target'][:, :-1].clone()
    changed_prefix[:, 2:] = EOS
    altered_logits = model(batch, changed_prefix)[0]
    assert torch.allclose(original_logits[:, :2], altered_logits[:, :2], atol=1e-6)


def test_oracle_predictions_have_perfect_evaluation_metrics():
    codec = CanonicalCodec(num_series=2)
    examples = [codec.encode(sample_formula())]
    class Oracle:
        predict_coefficients = True
        def eval(self):
            return self
        def generate(self, batch, max_steps):
            return batch['target'][:, 1:], batch['coefficient_targets'][:, 1:]
    metrics, _, _ = evaluate_greedy(Oracle(), examples, codec, small_config())
    assert metrics['skeleton_exact'] == metrics['joint_exact'] == metrics['term_f1'] == 1.
    assert metrics['coefficient_union_rmse'] == 0.


def test_small_training_saves_checkpoints_and_can_reopen_completed_run(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = small_config()
        data, summary, codec = prepare_dataset(config, tmp_path)
        result = train_variant(config['variants'][0], config, data, codec, tmp_path)
        assert result['state'] == 'completed'
        assert len(result['history']) == 2
        assert math.isfinite(result['test']['coefficient_union_rmse'])
        assert (tmp_path / 'tiny_tree_joint' / 'best.pt').exists()
        assert (tmp_path / 'tiny_tree_joint' / 'last.pt').exists()
        assert train_variant(config['variants'][0], config, data, codec, tmp_path) == result
        finish_study(config, tmp_path)
        assert read_status(tmp_path)['state'] == 'completed'
        _, cached, _ = prepare_dataset(config, tmp_path)
        assert summary == cached
    finally:
        torch.set_num_threads(previous_threads)
