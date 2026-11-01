"""Execute one notebook in place, saving completed cells and runtime status.

Launch this script in a detached process to keep long experiments running.
There is one notebook artifact; logs and checkpoints belong in the ignored
run directory. If execution fails, the traceback and completed outputs remain.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

import nbformat
from nbclient import NotebookClient


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    notebook_path = Path(args.notebook).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    notebook = nbformat.read(notebook_path, as_version=4)
    started = datetime.now(timezone.utc).isoformat()

    def status(state, **extra):
        record = {'state': state, 'runner_pid': os.getpid(), 'notebook': str(notebook_path),
                  'started_at_utc': started, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                  **extra}
        path = run_dir / 'runner_status.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
        os.replace(temporary, path)

    def save(**kwargs):
        temporary = notebook_path.with_suffix('.tmp')
        nbformat.write(notebook, temporary)
        os.replace(temporary, notebook_path)

    def starting(cell, cell_index, **kwargs):
        status('running', cell=cell_index, first_line=cell.source.splitlines()[0] if cell.source else '')
        print(f'Executing cell {cell_index}: {cell.source[:100]}', flush=True)

    status('starting')
    try:
        NotebookClient(notebook, timeout=None, kernel_name='python3',
                       on_cell_start=starting, on_cell_executed=save).execute(cwd=str(notebook_path.parent))
        save()
        status('completed')
        print('Notebook completed.', flush=True)
    except BaseException:
        save()
        status('failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
