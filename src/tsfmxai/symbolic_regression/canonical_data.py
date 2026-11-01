"""Supervised AST-to-canonical-sequence examples and skeleton-disjoint splits."""

from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite

from tsfmxai.generators import CombinatorialTreeConfig, CombinatorialTreeSampler, derive_sample_seed
from tsfmxai.representations import Add, Multiply, Parameter, Lag, canonicalize_polynomial

PAD, BOS, EOS, END_TERM = 0, 1, 2, 3
ATOM_OFFSET = 4


class CanonicalCodec:
    """Prefix AST input; sorted sparse factors separated by END_TERM output.

    Source token IDs 1/2/3 denote ADD/MUL/PARAM; IDs >=4 identify series-lag
    atoms. Numeric values are a separate source channel. Target token IDs
    1/2/3 denote BOS/EOS/END_TERM; IDs >=4 use the same atom dictionary.
    """

    def __init__(self, num_series=1, minimum_lag=1, maximum_lag=12, max_degree=4):
        self.num_series = num_series
        self.minimum_lag = minimum_lag
        self.maximum_lag = maximum_lag
        self.max_degree = max_degree
        self.lag_count = maximum_lag - minimum_lag + 1
        self.vocab_size = ATOM_OFFSET + num_series * self.lag_count

    def atom_token(self, series, lag):
        if not (0 <= series < self.num_series and self.minimum_lag <= lag <= self.maximum_lag):
            raise ValueError("Atom outside codec vocabulary.")
        return ATOM_OFFSET + series * self.lag_count + lag - self.minimum_lag

    def token_atom(self, token):
        if not ATOM_OFFSET <= token < self.vocab_size:
            raise ValueError("Token is not a lag atom.")
        series, offset = divmod(token - ATOM_OFFSET, self.lag_count)
        return (series, self.minimum_lag + offset)

    def encode(self, mechanism):
        prefix_ids, prefix_values = [], []
        tree_ids, tree_values, left_children, right_children = [], [], [], []

        def visit(node):
            if isinstance(node, Parameter):
                token, value = 3, float(mechanism.binding.parameter_values[node.slot])
            elif isinstance(node, Lag):
                token = self.atom_token(node.variable, mechanism.binding.lag_values[node.slot])
                value = 0.0
            elif isinstance(node, (Add, Multiply)):
                token, value = (1 if isinstance(node, Add) else 2), 0.0
            else:
                raise TypeError("Neural experiment supports polynomial ASTs only.")
            prefix_ids.append(token)
            prefix_values.append(value)
            left = visit(node.left) if isinstance(node, (Add, Multiply)) else -1
            right = visit(node.right) if isinstance(node, (Add, Multiply)) else -1
            index = len(tree_ids)
            tree_ids.append(token)
            tree_values.append(value)
            left_children.append(left)
            right_children.append(right)
            return index

        visit(mechanism.expression)
        polynomial = canonicalize_polynomial(mechanism.expression, mechanism.binding,
                                               max_degree=self.max_degree)
        target, coefficient_targets = [BOS], [0.0]
        for term, value in zip(polynomial.skeleton, polynomial.coefficients):
            target.extend(self.atom_token(*atom) for atom in term)
            coefficient_targets.extend(0.0 for _ in term)
            target.append(END_TERM)
            coefficient_targets.append(value)
        target.append(EOS)
        coefficient_targets.append(0.0)
        return {
            'prefix_ids': prefix_ids, 'prefix_values': prefix_values,
            'tree_ids': tree_ids, 'tree_values': tree_values,
            'left_children': left_children, 'right_children': right_children,
            'target': target, 'coefficient_targets': coefficient_targets,
            'skeleton': polynomial.skeleton, 'coefficients': polynomial.coefficients,
            'formula': mechanism.formula, 'sample_index': mechanism.sample_index,
            'operator_count': (len(tree_ids) - 1) // 2,
            'parameter_count': tree_ids.count(3),
            'degree': max(map(len, polynomial.skeleton), default=0),
        }

    def decode(self, tokens, coefficients=None):
        """Parse generated tokens after BOS; retain validity as a metric."""
        terms, values, current = [], [], []
        valid, ended = True, False
        for i, token in enumerate(tokens):
            token = int(token)
            if token == EOS:
                ended = True
                valid = valid and not current
                break
            if token == END_TERM:
                term = tuple(current)
                valid = valid and len(term) <= self.max_degree and term == tuple(sorted(term))
                if terms:
                    valid = valid and (len(terms[-1]), terms[-1]) < (len(term), term)
                terms.append(term)
                value = float(coefficients[i]) if coefficients is not None else 0.0
                valid = valid and isfinite(value)
                values.append(value)
                current = []
            elif ATOM_OFFSET <= token < self.vocab_size:
                current.append(self.token_atom(token))
            else:
                valid = False
        return {'skeleton': tuple(terms), 'coefficients': tuple(values),
                'valid': bool(valid and ended and not current), 'ended': ended}

    def target_text(self, tokens):
        names = {PAD: 'PAD', BOS: 'BOS', EOS: 'EOS', END_TERM: 'END_TERM'}
        return ' '.join(names[t] if t < ATOM_OFFSET else f'LAG{self.token_atom(t)}' for t in tokens)


def skeleton_digest(skeleton):
    return sha256(json.dumps(skeleton, separators=(',', ':')).encode()).hexdigest()


def split_for_skeleton(skeleton, split_seed):
    key = f'{split_seed}|{skeleton_digest(skeleton)}'.encode()
    bucket = int.from_bytes(sha256(key).digest()[:8], 'big') % 10_000
    return 'train' if bucket < 8000 else 'validation' if bucket < 9000 else 'test'


def generate_canonical_dataset(config):
    generator = CombinatorialTreeConfig(**config['generator'])
    sampler = CombinatorialTreeSampler(generator)
    codec = CanonicalCodec(generator.num_series, generator.minimum_lag,
                           generator.maximum_lag, generator.max_degree)
    sizes = config['dataset_sizes']
    datasets = {name: [] for name in sizes}
    keys = {name: set() for name in sizes}
    index = 0
    while any(len(datasets[name]) < size for name, size in sizes.items()):
        if index >= 50 * sum(sizes.values()):
            raise RuntimeError('Requested split quotas are unreachable with this generation budget.')
        seed = derive_sample_seed(config['data_seed'], index)
        example = codec.encode(sampler.sample(seed=seed, sample_index=index))
        partition = split_for_skeleton(example['skeleton'], config['split_seed'])
        if len(datasets[partition]) < sizes[partition]:
            if len(example['target']) - 1 > config['max_decode_steps']:
                raise ValueError('Target exceeds decoder length; increase max_decode_steps.')
            datasets[partition].append(example)
            keys[partition].add(skeleton_digest(example['skeleton']))
        index += 1
    assert not (keys['train'] & keys['validation'] or keys['train'] & keys['test']
                or keys['validation'] & keys['test'])
    fingerprint = sha256()
    for name, examples in datasets.items():
        fingerprint.update(name.encode())
        for example in examples:
            fingerprint.update(json.dumps(example, sort_keys=True, separators=(',', ':')).encode())
    summary = {
        'attempts': index, 'sizes': {k: len(v) for k, v in datasets.items()},
        'distinct_skeletons': {k: len(v) for k, v in keys.items()},
        'fingerprint': fingerprint.hexdigest(), 'split_intersections': 0,
        'max_source_nodes': max(len(e['tree_ids']) for ds in datasets.values() for e in ds),
        'max_target_steps': max(len(e['target']) - 1 for ds in datasets.values() for e in ds),
    }
    return datasets, summary, codec
