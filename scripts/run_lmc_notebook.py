"""Execute the LMC notebook with live stdout and save an executed copy."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import traceback

import nbformat


class Tee:
    def __init__(self, stream):
        self.stream = stream
        self.parts = []

    def write(self, value):
        self.stream.write(value)
        self.stream.flush()
        self.parts.append(value)

    def flush(self):
        self.stream.flush()


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    source = root / "experiments/generator_model_evaluation/01_chronos2_lmc_synth.ipynb"
    notebook = nbformat.read(source, as_version=4)
    namespace = {"__name__": "__main__", "__file__": str(source)}
    for number, cell in enumerate(notebook.cells, 1):
        if cell.cell_type != "code":
            continue
        stdout = Tee(sys.stdout)
        stderr = Tee(sys.stderr)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(cell.source, f"{source}:cell{number}", "exec"), namespace)
        except Exception:
            traceback.print_exc(file=stderr)
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="".join(stdout.parts)),
                            nbformat.v4.new_output("stream", name="stderr", text="".join(stderr.parts))]
            cell.execution_count = number
            raise
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="".join(stdout.parts))]
        if stderr.parts:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text="".join(stderr.parts)))
        cell.execution_count = number
    run = namespace["RUN"]
    output = root / "outputs/chronos2_lmc" / run / "executed.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print("Executed notebook:", output, flush=True)


if __name__ == "__main__":
    main()
