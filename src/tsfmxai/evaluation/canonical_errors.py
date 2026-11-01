"""Inspect canonical-decoding failures without updating model parameters."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from tsfmxai.symbolic_regression.canonical_data import BOS, EOS, END_TERM, CanonicalCodec
from tsfmxai.symbolic_regression.canonical_models import CanonicalEncoderDecoder, collate_canonical
from tsfmxai.symbolic_regression.canonical_training import atomic_json, format_prediction, utc_now


def structural_features(example):
    """Describe syntax and the distributive expansion before merging equal terms."""
    ids = example['prefix_ids']
    cursor = 0
    distribution_nodes = double_sum_products = 0

    def visit():
        nonlocal cursor, distribution_nodes, double_sum_products
        token = ids[cursor]
        cursor += 1
        if token >= 4:
            return Counter({(token,): 1}), 0
        if token == 3:
            return Counter({(): 1}), 0
        if token not in (1, 2):
            raise ValueError('Unsupported source token')
        left, left_depth = visit()
        right, right_depth = visit()
        if token == 1:
            terms = left + right
        else:
            distribution_nodes += int(sum(left.values()) > 1 or sum(right.values()) > 1)
            double_sum_products += int(sum(left.values()) > 1 and sum(right.values()) > 1)
            terms = Counter()
            for a, a_count in left.items():
                for b, b_count in right.items():
                    terms[tuple(sorted((*a, *b)))] += a_count * b_count
        return terms, 1 + max(left_depth, right_depth)

    raw, depth = visit()
    if cursor != len(ids):
        raise ValueError('Trailing source tokens')
    skeleton = example['skeleton']
    lags = [token for token in ids if token >= 4]
    coefficients = example['coefficients']
    parameters = [value for token, value in zip(ids, example['prefix_values']) if token == 3]
    return {
        'operator_count': ids.count(1) + ids.count(2), 'additions': ids.count(1),
        'multiplications': ids.count(2), 'parameter_count': len(parameters),
        'depth': depth, 'lag_leaves': len(lags), 'distinct_input_lags': len(set(lags)),
        'repeated_lag_leaves': len(lags) - len(set(lags)),
        'distribution_nodes': distribution_nodes, 'double_sum_products': double_sum_products,
        'expanded_terms_before_merge': sum(raw.values()),
        'merged_occurrences': sum(raw.values()) - len(raw),
        'cancelled_terms': len(raw) - len(skeleton),
        'target_terms': len(skeleton), 'target_tokens': len(example['target']) - 1,
        'degree': max(map(len, skeleton), default=0),
        'has_constant': int(() in skeleton),
        'power_terms': sum(len(term) > len(set(term)) for term in skeleton),
        'negative_parameters': sum(value < 0 for value in parameters),
        'min_abs_coefficient': min(map(abs, coefficients), default=0.),
    }


def _trim(tokens):
    tokens = list(map(int, tokens))
    return tokens[:tokens.index(EOS) + 1] if EOS in tokens else tokens


def classify_prediction(example, tokens, codec):
    tokens = _trim(tokens)
    decoded = codec.decode(tokens)
    truth = example['skeleton']
    # Retain multiplicity within each term (x*x differs from x); normalize order only.
    normalized = tuple(sorted(set(tuple(sorted(term)) for term in decoded['skeleton']),
                              key=lambda term: (len(term), term)))
    complete_syntax = bool(tokens and tokens[-1] == EOS and
                           (len(tokens) == 1 or tokens[-2] == END_TERM))
    expected, predicted = set(truth), set(normalized)
    exact = decoded['valid'] and decoded['skeleton'] == truth
    missing, extra = expected - predicted, predicted - expected
    same_support = complete_syntax and normalized == truth
    if exact:
        category = 'correct'
    elif same_support:
        category = 'ordering_or_duplicate_only'
    elif not complete_syntax:
        category = 'incomplete_syntax'
    elif missing == {()} and not extra:
        category = 'missing_constant_only'
    elif extra == {()} and not missing:
        category = 'extra_constant_only'
    elif missing and not extra:
        category = 'missing_terms_only'
    elif extra and not missing:
        category = 'extra_terms_only'
    else:
        category = 'missing_and_extra_terms'
    source_atoms = {codec.token_atom(t) for t in example['prefix_ids'] if t >= 4}
    predicted_atoms = {a for term in decoded['skeleton'] for a in term}
    factorial_support_equal = {frozenset(t) for t in truth} == {frozenset(t) for t in normalized}
    return {'exact': bool(exact), 'valid': bool(decoded['valid']), 'error_category': category,
            'support_correct_after_sorting': bool(same_support),
            'missing_terms': len(missing), 'extra_terms': len(extra),
            'duplicate_predicted_terms': len(decoded['skeleton']) - len(set(decoded['skeleton'])),
            'hallucinated_input_lags': len(predicted_atoms - source_atoms),
            'multiplicity_only_error': bool(not same_support and complete_syntax and factorial_support_equal),
            'ended': bool(decoded['ended']), 'predicted_tokens': len(tokens),
            'predicted_terms': len(decoded['skeleton']),
            'target_formula': format_prediction(truth),
            'predicted_formula': format_prediction(decoded['skeleton']),
            'missing_formula': format_prediction(tuple(sorted(missing, key=lambda t: (len(t), t)))) if missing else '',
            'extra_formula': format_prediction(tuple(sorted(extra, key=lambda t: (len(t), t)))) if extra else '',
            'prediction_ids': tokens}


def load_analysis_model(config_path, checkpoint_path):
    config = json.loads(Path(config_path).read_text(encoding='utf8'))
    g = config['generator']
    codec = CanonicalCodec(g['num_series'], g['minimum_lag'], g['maximum_lag'], g['max_degree'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = CanonicalEncoderDecoder(codec.vocab_size, encoder='transformer',
                                    predict_coefficients=False, **config['model'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, codec, config, checkpoint['epoch']


def as_example(row):
    result = dict(row)
    result['skeleton'] = tuple(tuple(tuple(a) for a in t) for t in json.loads(row['skeleton_json']))
    return result


@torch.inference_mode()
def audit_test_corpus(config_path, checkpoint_path, corpus_path, output_path, *, threads=4):
    """Cache per-example diagnostics for the full test set; weights remain frozen."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    model, codec, config, epoch = load_analysis_model(config_path, checkpoint_path)
    checkpoint_hash = sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    source_files = sorted((Path(corpus_path) / 'test').glob('part-*.parquet'))
    identity = {'checkpoint_sha256': checkpoint_hash, 'epoch': epoch,
                'config_sha256': sha256(Path(config_path).read_bytes()).hexdigest(),
                'analysis_code_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                'corpus_files': [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in source_files],
                'analysis_partition': 'sha256(error-study-v1|skeleton_hash) mod 10: 0 discovery, 1..9 confirmation',
                'threads': threads}
    identity = json.loads(json.dumps(identity))
    manifest = output_path / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf8')) != identity:
        raise ValueError('Analysis cache belongs to different checkpoint, code, or corpus')
    atomic_json(manifest, identity)
    seen = correct = valid = 0
    started = time.perf_counter()
    for shard_index, source in enumerate(source_files):
        destination = output_path / source.name
        if destination.exists():
            summary = pq.read_table(destination, columns=['exact', 'valid']).to_pydict()
            seen += len(summary['exact']); correct += sum(summary['exact']); valid += sum(summary['valid'])
            continue
        rows = pq.read_table(source).to_pylist()
        diagnostic = []
        for start in range(0, len(rows), config['batch_size']):
            examples = [as_example(row) for row in rows[start:start + config['batch_size']]]
            predictions, _ = model.generate(collate_canonical(examples), config['max_decode_steps'])
            predictions = predictions.tolist()
            for example, predicted in zip(examples, predictions):
                bucket = int.from_bytes(sha256(b'error-study-v1|' + example['skeleton_hash']).digest()[:8], 'big') % 10
                record = {'sample_index': example['sample_index'], 'skeleton_hash': example['skeleton_hash'],
                          'analysis_partition': 'discovery' if bucket == 0 else 'confirmation',
                          'input_formula': example['formula'], 'prefix_ids': example['prefix_ids'],
                          'prefix_values': example['prefix_values'], 'target_ids': example['target'],
                          **structural_features(example),
                          **classify_prediction(example, predicted, codec)}
                diagnostic.append(record)
        temporary = destination.with_suffix('.tmp')
        pq.write_table(pa.Table.from_pylist(diagnostic), temporary, compression='zstd')
        temporary.replace(destination)
        seen += len(diagnostic)
        correct += sum(r['exact'] for r in diagnostic); valid += sum(r['valid'] for r in diagnostic)
        atomic_json(output_path / 'status.json', {'state': 'evaluating', 'rows': seen,
            'correct': correct, 'valid': valid, 'shards': shard_index + 1, 'total_shards': len(source_files),
            'elapsed_seconds': time.perf_counter() - started, 'updated_at_utc': utc_now()})
    atomic_json(output_path / 'status.json', {'state': 'completed', 'rows': seen, 'correct': correct,
        'errors': seen - correct, 'valid': valid, 'exact': correct / seen,
        'elapsed_seconds': time.perf_counter() - started, 'updated_at_utc': utc_now()})
    return pd.concat([pq.read_table(p).to_pandas() for p in sorted(output_path.glob('part-*.parquet'))],
                     ignore_index=True)


def wilson(errors, n):
    if not n:
        return (float('nan'), float('nan'))
    p, z = errors / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - radius, center + radius


def grouped_errors(frame, column):
    result = []
    for value, group in frame.groupby(column, dropna=False):
        n, errors = len(group), int((~group['exact']).sum())
        low, high = wilson(errors, n)
        result.append({column: value, 'examples': n, 'errors': errors,
                       'error_pct': 100 * errors / n, 'error_low_pct': 100 * low,
                       'error_high_pct': 100 * high})
    return pd.DataFrame(result)


HYPOTHESES = {
    'six_operators': ('operator_count', 6),
    'degree_four': ('degree', 4),
    'at_least_six_terms': ('target_terms', 6),
    'at_least_25_output_tokens': ('target_tokens', 25),
    'at_least_three_distributions': ('distribution_nodes', 3),
    'product_of_two_sums': ('double_sum_products', 1),
    'at_least_two_merged_occurrences': ('merged_occurrences', 2),
    'repeated_lag_in_input': ('repeated_lag_leaves', 1),
    'power_in_target': ('power_terms', 1),
    'constant_in_target': ('has_constant', 1),
    'parameter_in_input': ('parameter_count', 1),
    'negative_parameter': ('negative_parameters', 1),
    'deep_tree': ('depth', 5),
}


def hypothesis_table(frame, selected=None):
    """Risk differences with denominators; independent discovery/confirmation labels."""
    records = []
    for name, (column, threshold) in HYPOTHESES.items():
        if selected is not None and name not in selected:
            continue
        flag = frame[column] >= threshold
        yes, no = frame[flag], frame[~flag]
        if not len(yes) or not len(no):
            continue
        e1, e0 = int((~yes['exact']).sum()), int((~no['exact']).sum())
        r1, r0 = e1 / len(yes), e0 / len(no)
        se = math.sqrt(r1 * (1 - r1) / len(yes) + r0 * (1 - r0) / len(no))
        records.append({'hypothesis': name, 'with_n': len(yes), 'without_n': len(no),
                        'with_error_pct': 100 * r1, 'without_error_pct': 100 * r0,
                        'difference_pp': 100 * (r1 - r0),
                        'difference_low_pp': 100 * (r1 - r0 - 1.96 * se),
                        'difference_high_pp': 100 * (r1 - r0 + 1.96 * se),
                        'relative_error': r1 / r0 if r0 else float('inf')})
    return pd.DataFrame(records)


def swap_children(example):
    """Swap operands at every +/* node; preserve leaves, coefficients and node count."""
    index = 0

    def parse():
        nonlocal index
        token, value = example['prefix_ids'][index], example['prefix_values'][index]
        index += 1
        return (token, value, parse(), parse()) if token in (1, 2) else (token, value)

    root = parse()
    if index != len(example['prefix_ids']):
        raise ValueError('Trailing source tokens')
    prefix_ids, prefix_values, tree_ids, tree_values, left_children, right_children = [], [], [], [], [], []

    def emit(node):
        token, value = node[:2]
        prefix_ids.append(token); prefix_values.append(value)
        left = emit(node[3]) if token in (1, 2) else -1
        right = emit(node[2]) if token in (1, 2) else -1
        position = len(tree_ids)
        tree_ids.append(token); tree_values.append(value)
        left_children.append(left); right_children.append(right)
        return position

    emit(root)
    return {**example, 'prefix_ids': prefix_ids, 'prefix_values': prefix_values,
            'tree_ids': tree_ids, 'tree_values': tree_values,
            'left_children': left_children, 'right_children': right_children}


def diagnostic_example(row, codec):
    """Recover source inputs from a diagnostic row; coefficients are unused for decoding."""
    ids, values = list(row['prefix_ids']), list(row['prefix_values'])
    target = list(row['target_ids'])
    skeleton = codec.decode(target[1:])['skeleton']
    initial = {'prefix_ids': ids, 'prefix_values': values, 'target': target,
               'coefficient_targets': [0.] * len(target), 'skeleton': skeleton,
               'coefficients': [1.] * len(skeleton), 'formula': row['input_formula']}
    # Two swaps restore the original syntax and generate consistent postorder arrays.
    return swap_children(swap_children(initial))


@torch.inference_mode()
def commutative_probe(frame, model, codec, config, *, count=10000, seed=20261103):
    """A fixed sample, independent of success, tests equivalent input arrangements."""
    sample = frame.sample(n=min(count, len(frame)), random_state=seed)
    records = []
    for start in range(0, len(sample), config['batch_size']):
        block = sample.iloc[start:start + config['batch_size']]
        original = [diagnostic_example(row, codec) for row in block.to_dict('records')]
        swapped = [swap_children(example) for example in original]
        predictions, _ = model.generate(collate_canonical(swapped), config['max_decode_steps'])
        for row, example, prediction in zip(block.to_dict('records'), swapped, predictions.tolist()):
            result = classify_prediction(example, prediction, codec)
            records.append({'sample_index': row['sample_index'], 'original_exact': row['exact'],
                'swapped_exact': result['exact'],
                'changed_prediction': list(row['prediction_ids']) != result['prediction_ids'],
                'original_formula': row['input_formula'],
                'target_formula': row['target_formula'], 'original_prediction': row['predicted_formula'],
                'swapped_prediction': result['predicted_formula']})
    return pd.DataFrame(records)
