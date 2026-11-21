"""Create a single executed notebook containing the complete experiment record."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" /
                        "generator_model_evaluation" / "generator_selection" / "kernel_synth_hybrid_assessment" /
                        "01_batch_generation_speed.ipynb")
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    source_paths = (
        "src/tsfmxai/generators/hybrid.py", "src/tsfmxai/generators/kernel_synth.py",
        "src/tsfmxai/generators/multiseries.py", "src/tsfmxai/datasets/hybrid.py",
        "src/tsfmxai/datasets/multiseries.py", "src/tsfmxai/datasets/online.py",
        "scripts/benchmark_hybrid_batches.py",
    )
    payload["provenance"] = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                             text=True).strip(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in source_paths},
    }
    payload["question"] = "How does KernelSynth channel probability affect delivered batch time?"
    payload["hypothesis"] = (
        "More KernelSynth channels may increase sustained batch time because GP "
        "factorization is more expensive than AST execution."
    )
    payload["scientific_summary"] = (
        "Exploratory CPU-only generation benchmark. The kernel fraction is sampled per "
        "channel; reported realized fractions are measured. Factorization is cached per "
        "worker, but GP trajectories are fresh. The legacy AST builder also samples "
        "unused ASTs for KernelSynth channels. No Chronos forward or GPU stall is measured. "
        "This is not a speed comparison with TimePFN LMC: that runner generates "
        "1024-point trajectories, caches them, extracts multiple windows, and its "
        "published cold timing also includes a GPU update. The present 208-point "
        "KernelSynth-like sampler uses the official kernel bank but is not the pinned "
        "TimePFN implementation."
    )
    payload["reproduction"] = (
        "From repository root with the recorded Python environment: "
        "python scripts/benchmark_hybrid_batches.py --output outputs/hybrid_benchmark.json "
        "--workers 8 --warmup 16 --measured 64; "
        "python scripts/record_hybrid_benchmark.py outputs/hybrid_benchmark.json"
    )
    rows = payload["results"]
    minimum = min(row["sustained_seconds_per_batch"] for row in rows)
    maximum = max(row["sustained_seconds_per_batch"] for row in rows)
    payload["scientific_summary"] += (
        f" Observed sustained seconds/batch ranged from {minimum:.4f} to {maximum:.4f}."
    )
    record_source = "# Complete executed experiment record\nimport json\nrecord = json.loads(" + repr(
        json.dumps(payload, ensure_ascii=False)) + ")\n" + (
        "for row in record['results']:\n"
        "    print(f\"p={row['kernel_probability']:.0%}, realized={row['actual_kernel_fraction']:.1%}, "
        "seconds/batch={row['sustained_seconds_per_batch']:.3f}, "
        "p95 wait={row['p95_seconds']:.3f}\")"
    )
    chart_source = """import matplotlib.pyplot as plt
rows = record['results']
x = [100 * r['kernel_probability'] for r in rows]
y = [r['sustained_seconds_per_batch'] for r in rows]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x, y, marker='o', linewidth=2)
for xi, yi in zip(x, y):
    ax.annotate(f'{yi:.3f} s', (xi, yi), xytext=(0, 9),
                textcoords='offset points', ha='center')
ax.set(xlabel='Probabilidade de canal KernelSynth (%)',
       ylabel='Segundos por batch entregue',
       title='Gerador híbrido: tempo de geração com 8 workers')
ax.set_xticks(x)
ax.set_ylim(0, max(y) * 1.18)
ax.grid(alpha=.25)
fig.tight_layout()
plt.show()"""
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(
            "# KernelSynth como canal exógeno no gerador multivariado\n\n"
            "Este ensaio mede a entrega de batches de **16 exemplos × 192 passos × 6 canais** "
            "(contexto 96, futuro 96). Cada canal é AST ou KernelSynth, sorteado "
            "independentemente. Os canais KernelSynth não têm alvo da equation head; "
            "os AST podem depender dos seus lags. A figura usa a taxa sustentada depois "
            "de 16 batches de aquecimento, em 64 batches medidos por ponto, com oito "
            "workers fixos. A fila usa prefetch 2. Os tempos são desta máquina, sem GPU."
        ),
        nbformat.v4.new_code_cell(record_source),
        nbformat.v4.new_code_cell(chart_source),
        nbformat.v4.new_markdown_cell(
            "## Interpretação\n\nA curva mede **tempo por batch entregue** com o consumidor "
            "pedindo batches continuamente. A espera de `next()` pode ser pequena "
            "quando há itens prefetched; por isso usamos o tempo total dos 64 batches "
            "medidos dividido por 64. A preparação dos 16 batches iniciais e o "
            "início dos workers aparecem separadamente no registro. "
            "Cada ponto sorteia uma fração efetiva próxima, mas não idêntica, à "
            "probabilidade configurada. Este teste não mede a qualidade do forecasting.\n\n"
            "**Não comparar diretamente com o tempo LMC anterior:** aqui usamos "
            "trajetórias de aproximadamente 208 pontos, uma janela por sistema, "
            "e nenhum passo de treino. O runner TimePFN usa trajetórias de 1.024 "
            "pontos, cache e múltiplas janelas por trajetória. Sua medida fria "
            "inclui atualização na GPU. Este caminho reproduz o banco de kernels "
            "em um sampler próprio, não o código fixado do TimePFN."
        ),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    executed = NotebookClient(notebook, timeout=120, kernel_name="python3").execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
