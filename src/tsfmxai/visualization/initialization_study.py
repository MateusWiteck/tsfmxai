"""Notebook displays for the three controlled initialization experiments.

Requires the experiment extras pandas and matplotlib. Rates are descriptive;
replicates within one mechanism are dependent and are not independent trials.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def initialization_tables(results):
    frame = pd.DataFrame([row for result in results.values() for row in result["rows"]])
    skipped = pd.DataFrame([row for result in results.values() for row in result["skipped"]])
    if frame.empty:
        return pd.DataFrame(), skipped
    keys = ["method", "case", "horizon"]
    summary = frame.groupby(keys, sort=False).agg(
        mechanisms=("mechanism", "nunique"), trajectories=("trajectory", "size"),
        explosions=("exploded", "sum"), near_zero=("near_zero", "sum"),
        persistent_flat=("persistent_flat_tail", "sum"),
        flat_nonzero=("flat_nonzero", "sum"),
        amplitude_decline=("amplitude_decline_near_zero", "sum"),
        max_initial_abs=("initial_max_abs", "max"),
        max_amplitude_l2_bound=("amplitude_l2_bound", "max"),
    ).reset_index()
    for metric in ("explosions", "near_zero", "persistent_flat", "flat_nonzero", "amplitude_decline"):
        summary[metric + "_pct"] = 100 * summary[metric] / summary.trajectories
    if "actual_order" in frame:
        detail = frame.groupby(keys, sort=False).agg(
            minimum_actual_lag=("actual_order", "min"), maximum_actual_lag=("actual_order", "max"),
            mean_initialization_seconds=("initialization_seconds", "mean"),
            mean_simulation_seconds=("simulation_seconds", "mean"),
        ).reset_index()
        summary = summary.merge(detail, on=keys, how="left", sort=False)
    skipped = pd.DataFrame([row for result in results.values() for row in result["skipped"]])
    return summary, skipped


def forgetting_table(results):
    frame = pd.DataFrame([row for result in results.values() for row in result["rows"]])
    if frame.empty:
        return pd.DataFrame()
    summary = frame.groupby(["method", "case", "horizon"], sort=False).agg(
        mechanisms=("mechanism", "nunique"), pairs=("trajectory", "size"),
        pair_explosions=("pair_exploded", "sum"), surviving_pairs=("pair_success", "sum"),
        forgotten_pairs=("persistent_forgetting", "sum"),
        median_final_state_gap=("state_gap", "median"),
        median_settling_step_achieved=("observed_settling_step", "median"),
        weighted_checks=("weighted_bound_passed", "count"),
    ).reset_index()
    summary["pair_explosion_pct"] = 100 * summary.pair_explosions / summary.pairs
    summary["forgotten_pct_all_pairs"] = 100 * summary.forgotten_pairs / summary.pairs
    summary["forgotten_pct_survivors"] = 100 * summary.forgotten_pairs / summary.surviving_pairs.replace(0, np.nan)
    return summary


def plot_initialization_result(result, title, config):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for index, axis in enumerate(axes.flat[:3]):
        for example in result["examples"]:
            if example["mechanism"] != index:
                continue
            history, path = example["history"][0], example["path"][0]
            line, = axis.plot(np.arange(1, len(path)+1), path, label=example["case"], linewidth=1)
            axis.plot(np.arange(1-len(history), 1), history, color=line.get_color(), linestyle=":")
        axis.set(title=f"Fixed mechanism {index}; trajectory 0", xlabel="Generated step (history ≤ 0)", ylabel="Value")
        axis.set_yscale("symlog", linthresh=.2)
        axis.axvline(0, color="gray", linewidth=.5)
    axes[0, 0].legend(fontsize=7)
    summary, _ = initialization_tables({title: result})
    last = summary[summary.horizon == max(config.horizons)]
    x = np.arange(len(last))
    for j, (metric, label) in enumerate((("explosions_pct", "Explosion"), ("near_zero_pct", "Near zero"), ("persistent_flat_pct", "Persistent flat tail"))):
        axes[1, 1].bar(x + (j-1)*.24, last[metric], width=.24, label=label)
    axes[1, 1].set_xticks(x, last.case, rotation=30, ha="right")
    axes[1, 1].set(ylabel="% of all attempted trajectories", ylim=(0, 105), title=f"h = {max(config.horizons)}; overlapping labels")
    axes[1, 1].legend(fontsize=7)
    fig.suptitle(title)
    return fig


def plot_history_comparison(results, title, config, *, zoom_steps=None):
    """Columns are generator variants; rows are predetermined mechanism IDs.

    Every saved trajectory is drawn. Paired paths use the same color with
    solid/dashed lines; history is dotted. Grey shading marks the used lag state.
    """
    groups = list(results)
    count = min(config.example_mechanisms, config.mechanisms)
    fig, axes = plt.subplots(count, len(groups), figsize=(22, 2.8 * count), squeeze=False,
                             constrained_layout=True)
    for col, group in enumerate(groups):
        result = results[group]
        for row in range(count):
            ax = axes[row, col]
            ax.set_title(f"{group}\nmechanism {row}", fontsize=9)
            samples = [e for e in result["examples"] if e["mechanism"] == row]
            if not samples:
                ax.text(.5, .5, "Not applicable\n" + (result["skipped"][0]["reason"]
                        if result["skipped"] else "No saved examples"),
                        ha="center", va="center", transform=ax.transAxes, wrap=True, fontsize=8)
                ax.set_axis_off()
                continue
            order = samples[0]["actual_order"]
            ax.axvspan(.5-order, .5, color="gray", alpha=.12)
            ax.axvline(.5, color="black", linewidth=.7)
            for e in samples:
                color = f"C{e['trajectory'] % 10}"
                for suffix, style in (("", "-"), ("_b", "--")):
                    if "path"+suffix not in e:
                        continue
                    h, path = e["history"+suffix][0], e["path"+suffix][0]
                    ax.plot(np.arange(1-len(h), 1), h, color=color, linestyle=":" if not suffix else "--",
                            alpha=.9, linewidth=.9, marker="o" if len(h) == 1 else None,
                            markersize=5, zorder=3)
                    label = f"r{e['trajectory']}" + (" B" if suffix else " A")
                    ax.plot(np.arange(1, len(path)+1), path, color=color, linestyle=style,
                            linewidth=.9, alpha=.8, label=label)
            if zoom_steps is not None:
                ax.set_xlim(max(1-len(samples[0]["history"][0]), -2*order), zoom_steps)
            ax.set_yscale("symlog", linthresh=.2)
            ax.set_title(f"{group}\nmechanism {row}; actual memory p={order}", fontsize=9)
            ax.set_xlabel("History t ≤ 0 | generated t ≥ 1", fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel("Value (symlog)")
            if row == 0:
                ax.legend(fontsize=6, ncol=2)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_startup_histories(results, title):
    """Linear-scale view makes cosine truncation, offsets and used memory visible."""
    fig, axes = plt.subplots(1, len(results), figsize=(22, 3.2), squeeze=False,
                             constrained_layout=True)
    for ax, (group, result) in zip(axes.flat, results.items()):
        ax.set_title(group)
        samples = [e for e in result["examples"] if e["mechanism"] == 0]
        if not samples:
            ax.text(.5, .5, "Not applicable", ha="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        for e in samples:
            h = e["history"][0]
            ax.plot(np.arange(1-len(h), 1), h, label=f"r{e['trajectory']}", alpha=.9,
                    marker="o" if len(h) == 1 else None, markersize=6, zorder=3)
        order = samples[0]["actual_order"]
        ax.axvspan(.5-order, .5, color="gray", alpha=.15, zorder=0)
        ax.axvline(0, color="black", lw=.6, zorder=1)
        if all(e["history"].shape[1] == 1 for e in samples):
            ax.set_xlim(-1, 1)
            ax.set_xticks([0])
            ax.text(.03, .95, "1 sample per trajectory", transform=ax.transAxes,
                    va="top", fontsize=8)
        values = np.concatenate([e["history"][0] for e in samples])
        if np.ptp(values) == 0:
            margin = max(.1, .1 * abs(float(values[0])))
            ax.set_ylim(values[0] - margin, values[0] + margin)
        ax.set(title=f"{group}\nmechanism 0; p={order}; N={len(h)}",
               xlabel="Initial-history time", ylabel="Value")
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(labelsize=8)
    fig.suptitle(title + " | shaded: lag state actually used")
    return fig


def plot_case_rates(results, title, config):
    summary, _ = initialization_tables(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), constrained_layout=True)
    for ax, (metric, label) in zip(axes, (("explosions_pct", "Explosion"),
                                        ("near_zero_pct", "Near zero"),
                                        ("persistent_flat_pct", "Persistent flat tail"))):
        if not summary.empty:
            for group, block in summary.groupby("method", sort=False):
                ax.plot(block.horizon, block[metric], "o-", label=group)
        ax.set(title=label, xlabel="Generated horizon", ylabel="% of all valid attempts", ylim=(-2, 102))
        ax.set_xticks(config.horizons)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    return fig


def plot_pair_gaps(results, title, config):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), constrained_layout=True)
    for group, result in results.items():
        if not result["curves"]:
            continue
        states = np.stack([c["state_gap"] for c in result["curves"]])
        initial = states[:, :1]
        ratios = np.divide(states, initial, out=np.full_like(states, np.nan), where=initial > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q = np.nanquantile(ratios, [.25, .5, .75], axis=0)
        line, = axes[0].plot(np.maximum(q[1], 1e-16), label=group)
        axes[0].fill_between(np.arange(states.shape[1]), np.maximum(q[0], 1e-16),
                             np.maximum(q[2], 1e-16), color=line.get_color(), alpha=.12)
        axes[1].plot(np.isfinite(states).sum(axis=0), label=group)
        summary = forgetting_table({group: result})
        axes[2].plot(summary.horizon, summary.forgotten_pct_all_pairs, "o-", label=group)
    axes[0].set(yscale="log", xlabel="Generated step", ylabel="State gap / initial state gap",
                title="Median and IQR among observable pairs")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="Generated step", ylabel="Observable pairs", title="Denominator after failures")
    axes[2].set(xlabel="Horizon", ylabel="% of all valid attempted pairs", ylim=(-2, 102),
                title="Persistent forgetting: final 24 steps ≤ 1e-6")
    fig.suptitle(title)
    return fig


def plot_condition_summary(frame, title, *, forgetting=False):
    """Compare conditions without pooling sample sizes or treating missing as zero."""
    metrics = (("pair_explosion_pct", "Pair explosions"),
               ("forgotten_pct_all_pairs", "Persistent forgetting")) if forgetting else (
                   ("explosions_pct", "Explosion"), ("near_zero_pct", "Near zero"),
                   ("persistent_flat_pct", "Persistent flat tail"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4.2), squeeze=False,
                             constrained_layout=True)
    last = frame[frame.horizon == frame.horizon.max()]
    conditions = list(dict.fromkeys(last.block))
    for ax, (metric, label) in zip(axes.flat, metrics):
        for group, subset in last.groupby("method", sort=False):
            values = subset.set_index("block")[metric].reindex(conditions)
            ax.plot(np.arange(len(conditions)), values, "o-", label=group, ms=3)
        ax.set_xticks(np.arange(len(conditions)), conditions, rotation=65, ha="right", fontsize=6)
        ax.set(title=label, ylabel="% of valid attempted cases", ylim=(-2, 102))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    return fig


def plot_forgetting_result(result, title, config):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    cases = list(dict.fromkeys(c["case"] for c in result["curves"]))
    for case in cases:
        curves = [c for c in result["curves"] if c["case"] == case]
        states = np.stack([c["state_gap"] for c in curves])
        ratios = states / states[:, :1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(ratios, axis=0)
        axes[0].plot(np.maximum(median, 1e-16), label=case)
        axes[1].plot(np.isfinite(states).sum(axis=0), label=case)
    axes[0].set(yscale="log", xlabel="Generated step", ylabel="Median state-gap / initial gap",
                title="Conditional on pairs still observable")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="Generated step", ylabel="Observable pairs", title="Failures become missing distances")
    e = next(e for e in result["examples"] if e["mechanism"] == 0 and e["case"] == "opposite constants")
    for key, label in (("path", "+ constant history"), ("path_b", "− constant history")):
        axes[2].plot(np.arange(1, e[key].shape[1]+1), e[key][0], label=label, linewidth=1)
    axes[2].set(yscale="symlog", xlabel="Generated step", ylabel="Value", title="Mechanism 0, trajectory 0; identical noise")
    axes[2].legend(fontsize=7)
    fig.suptitle(title)
    return fig
