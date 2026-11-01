"""Frozen-checkpoint error audit; never starts or resumes training."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from tsfmxai.evaluation.canonical_errors import audit_test_corpus
from tsfmxai.symbolic_regression.canonical_training import atomic_json, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='outputs/canonicalization/ast_5m_unique_eval_20261030')
    parser.add_argument('--config', default='configs/canonicalization_5m_ast_parquet_unique_eval.json')
    parser.add_argument('--output', default='outputs/canonicalization/error_analysis_20261031')
    args = parser.parse_args()
    run_dir, output = ROOT / args.run_dir, ROOT / args.output
    try:
        frame = audit_test_corpus(ROOT / args.config,
            run_dir / 'checkpoints/transformer_skeleton/best.pt', run_dir / 'parquet', output)
        print('rows', len(frame), 'errors', (~frame.exact).sum(), 'accuracy', frame.exact.mean(), flush=True)
    except Exception as error:
        atomic_json(output / 'failure.json', {'error': repr(error), 'at_utc': utc_now()})
        raise


if __name__ == '__main__':
    main()
