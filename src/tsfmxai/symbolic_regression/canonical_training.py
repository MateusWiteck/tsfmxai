"""Resumable CPU training and free-decoding evaluation for canonical AST targets."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec, generate_canonical_dataset
from tsfmxai.symbolic_regression.canonical_models import (
    CanonicalEncoderDecoder, collate_canonical, supervised_loss,
)
from tsfmxai.representations.canonical_polynomial import format_monomial


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def read_status(run_dir):
    path = Path(run_dir) / 'status.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def update_status(run_dir, **updates):
    status = read_status(run_dir)
    status.update(updates)
    status['updated_at_utc'] = utc_now()
    atomic_json(Path(run_dir) / 'status.json', status)
    return status


def config_digest(config):
    return sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare_dataset(config, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    digest = config_digest(config)
    previous = read_status(run_dir)
    if previous and previous.get('config_digest') != digest:
        raise ValueError('Run directory already belongs to a different experiment configuration.')
    update_status(run_dir, state='preparing_data', config_digest=digest, kernel_pid=os.getpid())
    cache = run_dir / 'dataset.pt'
    if cache.exists():
        # Only this local experiment creates this trusted artifact.
        saved = torch.load(cache, weights_only=False, map_location='cpu')
        if saved['config_digest'] != digest:
            raise ValueError('Cached dataset configuration does not match.')
        datasets, summary = saved['datasets'], saved['summary']
        g = config['generator']
        codec = CanonicalCodec(g['num_series'], g['minimum_lag'], g['maximum_lag'], g['max_degree'])
    else:
        datasets, summary, codec = generate_canonical_dataset(config)
        temporary = cache.with_suffix('.tmp')
        torch.save({'datasets': datasets, 'summary': summary, 'config_digest': digest}, temporary)
        os.replace(temporary, cache)
    update_status(run_dir, state='data_ready', dataset=summary)
    return datasets, summary, codec


def _averages(totals, coefficient_weight):
    ce = totals['token_sum'] / max(1, totals['token_count'])
    coef = totals['coefficient_sum'] / max(1, totals['term_count'])
    return {'loss': ce + coefficient_weight * coef, 'token_ce': ce,
            'coefficient_huber': coef,
            'teacher_token_accuracy': totals['token_correct'] / max(1, totals['token_count']),
            'teacher_coefficient_mae': totals['coefficient_mae_sum'] / max(1, totals['term_count'])}


@torch.no_grad()
def evaluate_teacher(model, examples, config, device="cpu"):
    model.eval()
    totals = defaultdict(float)
    loader = DataLoader(examples, batch_size=config['batch_size'], collate_fn=collate_canonical)
    for batch in loader:
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        _, metrics = supervised_loss(model, batch, config['coefficient_weight'])
        for key, value in metrics.items():
            totals[key] += value
    averages = _averages(totals, config['coefficient_weight'])
    if not model.predict_coefficients:
        averages['teacher_coefficient_mae'] = None
        averages['coefficient_huber'] = None
    return averages


def format_prediction(skeleton, coefficients=None):
    if coefficients is None:
        return ' + '.join(format_monomial(term) for term in skeleton) or '0'
    return ' + '.join(f'({value:.4g})*{format_monomial(term)}'
                      for term, value in zip(skeleton, coefficients)) or '0'


@torch.no_grad()
def evaluate_greedy(model, examples, codec, config, collect_examples=False, device="cpu"):
    model.eval()
    loader = DataLoader(examples, batch_size=config['batch_size'], collate_fn=collate_canonical)
    correct = valid = tp = fp = fn = joint_correct = 0
    union_squared = union_count = matched_absolute = matched_count = 0
    rows, illustrations = [], []
    seen_illustrations = {True: 0, False: 0}
    offset = 0
    started = time.perf_counter()
    for batch in loader:
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        generated, values = model.generate(batch, config['max_decode_steps'])
        for j in range(len(generated)):
            actual = examples[offset + j]
            decoded = codec.decode(generated[j].tolist(), values[j].tolist() if values is not None else None)
            truth_set, predicted_set = set(actual['skeleton']), set(decoded['skeleton'])
            exact = decoded['valid'] and decoded['skeleton'] == actual['skeleton']
            correct += exact
            valid += decoded['valid']
            tp += len(truth_set & predicted_set)
            fp += len(predicted_set - truth_set)
            fn += len(truth_set - predicted_set)
            joint = False
            if model.predict_coefficients:
                # Invalid expressions are evaluated as an empty coefficient map.
                predicted = dict(zip(decoded['skeleton'], decoded['coefficients'])) if decoded['valid'] else {}
                truth = dict(zip(actual['skeleton'], actual['coefficients']))
                for term in predicted.keys() | truth.keys():
                    error = predicted.get(term, 0.) - truth.get(term, 0.)
                    union_squared += error * error
                    union_count += 1
                for term in predicted.keys() & truth.keys():
                    matched_absolute += abs(predicted[term] - truth[term])
                    matched_count += 1
                joint = exact and all(math.isclose(predicted[t], v,
                    abs_tol=config['coefficient_atol'], rel_tol=config['coefficient_rtol'])
                    for t, v in truth.items())
                joint_correct += joint
            rows.append({'operator_count': actual['operator_count'], 'degree': actual['degree'],
                         'parameter_count': actual['parameter_count'], 'terms': len(truth_set),
                         'exact': int(exact), 'valid': int(decoded['valid'])})
            if collect_examples and seen_illustrations[bool(exact)] < 6:
                seen_illustrations[bool(exact)] += 1
                illustrations.append({'input_formula': actual['formula'],
                    'target': format_prediction(actual['skeleton'], actual['coefficients'] if model.predict_coefficients else None),
                    'predicted': format_prediction(decoded['skeleton'], decoded['coefficients'] if model.predict_coefficients else None),
                    'valid': decoded['valid'], 'skeleton_exact': bool(exact),
                    'joint_exact': bool(joint) if model.predict_coefficients else None})
        offset += len(generated)
    groups = []
    for field in ('operator_count', 'degree', 'parameter_count', 'terms'):
        for value in sorted({r[field] for r in rows}):
            subset = [r for r in rows if r[field] == value]
            groups.append({'group': field, 'value': value, 'count': len(subset),
                           'skeleton_exact': sum(r['exact'] for r in subset) / len(subset)})
    result = {'count': len(examples), 'skeleton_exact': correct / max(1, len(examples)),
              'valid_rate': valid / max(1, len(examples)),
              'term_precision': tp / max(1, tp + fp), 'term_recall': tp / max(1, tp + fn),
              'term_f1': 2 * tp / max(1, 2 * tp + fp + fn),
              'joint_exact': joint_correct / len(examples) if model.predict_coefficients else None,
              'coefficient_union_rmse': math.sqrt(union_squared / max(1, union_count)) if model.predict_coefficients else None,
              'coefficient_matched_mae': matched_absolute / matched_count if matched_count else None,
              'matched_coefficient_count': matched_count,
              'seconds': time.perf_counter() - started}
    return result, illustrations, groups


def _checkpoint(path, data):
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def train_variant(variant, config, datasets, codec, run_dir):
    """Train one variant, saving resumable state after every complete epoch."""
    run_dir = Path(run_dir)
    name = variant['name']
    config_hash = config_digest(config)
    status = read_status(run_dir)
    if status.get('config_digest') != config_hash:
        raise ValueError('Prepare this configuration and dataset before training.')
    previous = status.get('variants', {}).get(name, {})
    if previous.get('state') == 'completed':
        return previous
    variant_dir = run_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = variant_dir / 'best.pt', variant_dir / 'last.pt'
    training_seed = config['training_seed']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(training_seed)
        model = CanonicalEncoderDecoder(codec.vocab_size, encoder=variant['encoder'],
            predict_coefficients=variant['coefficients'], **config['model'])
        warm_start_path = config.get('warm_start', {}).get(name)
        if warm_start_path and not last_path.exists():
            source = torch.load(warm_start_path, weights_only=False, map_location='cpu')
            if source['variant'] != variant:
                raise ValueError('Warm-start checkpoint variant mismatch.')
            model.load_state_dict(source['model'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3,
                                                               min_lr=1e-5)
        shuffle_rng = torch.Generator().manual_seed(training_seed + 1)
        history, best_loss, best_epoch, bad_epochs, start_epoch, elapsed = [], math.inf, 0, 0, 1, 0.
        if last_path.exists():
            checkpoint = torch.load(last_path, weights_only=False, map_location='cpu')
            if checkpoint['config_digest'] != config_hash or checkpoint['variant'] != variant:
                raise ValueError('Checkpoint configuration mismatch.')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            torch.set_rng_state(checkpoint['torch_rng'])
            shuffle_rng.set_state(checkpoint['shuffle_rng'])
            history, best_loss, best_epoch = checkpoint['history'], checkpoint['best_loss'], checkpoint['best_epoch']
            bad_epochs, start_epoch, elapsed = checkpoint['bad_epochs'], checkpoint['epoch'] + 1, checkpoint['elapsed']
        loader = DataLoader(datasets['train'], batch_size=config['batch_size'], shuffle=True,
                            generator=shuffle_rng, collate_fn=collate_canonical, num_workers=0)
        n_parameters = sum(p.numel() for p in model.parameters())
        started = time.perf_counter()
        try:
            for epoch in range(start_epoch, config['epochs'] + 1):
                if bad_epochs >= config['patience'] and epoch > config['minimum_epochs']:
                    break
                model.train()
                totals = defaultdict(float)
                update_status(run_dir, state='training', active_variant=name, epoch=epoch,
                              max_epochs=config['epochs'], batch=0, batches_per_epoch=len(loader))
                for batch_index, batch in enumerate(loader, 1):
                    optimizer.zero_grad(set_to_none=True)
                    loss, metrics = supervised_loss(model, batch, config['coefficient_weight'])
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Training loss became non-finite.')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                    optimizer.step()
                    for key, value in metrics.items():
                        totals[key] += value
                    if batch_index % 50 == 0:
                        update_status(run_dir, batch=batch_index)
                train = _averages(totals, config['coefficient_weight'])
                validation = evaluate_teacher(model, datasets['validation'], config)
                greedy, _, _ = evaluate_greedy(model,
                    datasets['validation'][:config['validation_greedy_size']], codec, config)
                scheduler.step(validation['loss'])
                row = {'epoch': epoch, 'train_loss': train['loss'], 'validation_loss': validation['loss'],
                       'validation_token_accuracy': validation['teacher_token_accuracy'],
                       'validation_greedy_exact': greedy['skeleton_exact'],
                       'validation_greedy_f1': greedy['term_f1'],
                       'learning_rate': optimizer.param_groups[0]['lr']}
                history.append(row)
                if validation['loss'] < best_loss - config['min_delta']:
                    best_loss, best_epoch, bad_epochs = validation['loss'], epoch, 0
                    _checkpoint(best_path, {'model': model.state_dict(), 'variant': variant,
                                           'config_digest': config_hash, 'epoch': epoch})
                else:
                    bad_epochs += 1
                _checkpoint(last_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
                    'shuffle_rng': shuffle_rng.get_state(), 'history': history, 'best_loss': best_loss,
                    'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'epoch': epoch,
                    'elapsed': elapsed + time.perf_counter() - started, 'variant': variant,
                    'config_digest': config_hash})
                status = read_status(run_dir)
                variants = status.get('variants', {})
                variants[name] = {'state': 'training', 'variant': variant, 'history': history,
                                  'best_epoch': best_epoch, 'parameters': n_parameters}
                update_status(run_dir, variants=variants, last_epoch=row)
                print(f"{name}: epoch {epoch}/{config['epochs']} train={train['loss']:.4f} "
                      f"val={validation['loss']:.4f} greedy_exact={greedy['skeleton_exact']:.3f}", flush=True)
            best = torch.load(best_path, weights_only=False, map_location='cpu')
            model.load_state_dict(best['model'])
            update_status(run_dir, state='evaluating_test', active_variant=name)
            test, examples, groups = evaluate_greedy(model, datasets['test'], codec, config, True)
            teacher = evaluate_teacher(model, datasets['test'], config)
            result = {'state': 'completed', 'variant': variant, 'history': history,
                      'best_epoch': best['epoch'], 'parameters': n_parameters,
                      'warm_start_checkpoint': warm_start_path,
                      'training_seconds': elapsed + time.perf_counter() - started,
                      'test': test, 'teacher_test': teacher, 'examples': examples, 'groups': groups,
                      'checkpoint': str(best_path.resolve())}
            status = read_status(run_dir)
            variants = status.get('variants', {})
            variants[name] = result
            update_status(run_dir, state='variant_completed', variants=variants)
            return result
        except Exception as error:
            update_status(run_dir, state='failed', active_variant=name,
                          error=f'{type(error).__name__}: {error}')
            raise


def finish_study(config, run_dir):
    status = read_status(run_dir)
    if not all(status.get('variants', {}).get(v['name'], {}).get('state') == 'completed'
               for v in config['variants']):
        raise RuntimeError('Some variants are still unfinished.')
    return update_status(run_dir, state='completed', active_variant=None, completed_at_utc=utc_now())
