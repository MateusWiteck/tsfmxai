"""Reproducible KernelSynth and retained LMC-Synth experiment fixtures."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)


OFFICIAL_PERIODS = (
    24, 48, 96, 24 * 7, 48 * 7, 96 * 7, 7, 14, 30, 60, 365,
    365 * 2, 4, 26, 52, 4, 6, 12, 4, 4 * 10, 10,
)


def official_kernel_bank(length: int = 1024) -> list[dict[str, Any]]:
    """Return the 33 ordered KernelSynth bank entries used by Chronos."""
    bank: list[dict[str, Any]] = []
    for index, period in enumerate(OFFICIAL_PERIODS, start=1):
        bank.append({
            "id": f"periodic_{index:02d}_p{period}",
            "formula": f"ExpSineSquared(length_scale=1, periodicity={period}/{length})",
            "family": "periodic",
            "periods": [int(period)],
            "kernel": ExpSineSquared(length_scale=1.0, periodicity=period / length),
        })
    for sigma in (0.0, 1.0, 10.0):
        bank.append({"id": f"linear_sigma{sigma:g}", "formula": f"DotProduct(sigma_0={sigma:g})",
                     "family": "linear", "periods": [], "kernel": DotProduct(sigma_0=sigma)})
    for scale in (0.1, 1.0, 10.0):
        bank.append({"id": f"rbf_scale{scale:g}", "formula": f"RBF(length_scale={scale:g})",
                     "family": "rbf", "periods": [], "kernel": RBF(length_scale=scale)})
    for alpha in (0.1, 1.0, 10.0):
        bank.append({"id": f"rq_alpha{alpha:g}",
                     "formula": f"RationalQuadratic(length_scale=1, alpha={alpha:g})",
                     "family": "rational_quadratic", "periods": [],
                     "kernel": RationalQuadratic(length_scale=1.0, alpha=alpha)})
    for level in (0.1, 1.0):
        bank.append({"id": f"white_level{level:g}", "formula": f"WhiteKernel(noise_level={level:g})",
                     "family": "white", "periods": [], "kernel": WhiteKernel(noise_level=level)})
    bank.append({"id": "constant_1", "formula": "ConstantKernel(constant_value=1)",
                 "family": "constant", "periods": [], "kernel": ConstantKernel(constant_value=1.0)})
    if len(bank) != 33:
        raise AssertionError("The official KernelSynth bank must have 33 entries.")
    return bank


def covariance_matrix(kernel: Any, length: int = 1024) -> np.ndarray:
    time = np.linspace(0.0, 1.0, length)[:, None]
    matrix = np.asarray(kernel(time), dtype=float)
    return (matrix + matrix.T) / 2.0


def stable_factor(matrix: np.ndarray) -> np.ndarray:
    scale = max(float(np.mean(np.diag(matrix))), 1.0)
    identity = np.eye(len(matrix))
    for exponent in range(-10, -3):
        try:
            return np.linalg.cholesky(matrix + (10.0**exponent) * scale * identity)
        except np.linalg.LinAlgError:
            continue
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return vectors @ np.diag(np.sqrt(np.maximum(values, 1e-8 * scale)))


def draw_from_factor(factor: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((count, factor.shape[0])) @ factor.T


def standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(np.mean(train))
    scale = float(np.std(train) + 1e-12)
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def retained_kernel_cases(*, global_seed: int = 20260930, length: int = 1024,
                          composition_count: int = 20) -> list[dict[str, Any]]:
    """Build the 33 base cases and the retained composite cases."""
    bank = official_kernel_bank(length)
    base_covariances = [covariance_matrix(item["kernel"], length) for item in bank]
    cases = [{"id": item["id"], "formula": item["formula"], "category": "base",
              "family": item["family"], "periods": list(item["periods"]),
              "covariance": base_covariances[index]}
             for index, item in enumerate(bank)]
    rng = np.random.default_rng(global_seed + 10_000)
    for number in range(composition_count):
        count = 2 + number % 4
        indices = rng.integers(0, len(bank), size=count)
        operators = rng.choice(["+", "*"], size=count - 1)
        first = int(indices[0])
        matrix = base_covariances[first].copy()
        formula = f"({bank[first]['formula']})"
        periods = list(bank[first]["periods"])
        for operator, index_value in zip(operators, indices[1:]):
            index = int(index_value)
            item = bank[index]
            matrix = matrix + base_covariances[index] if operator == "+" else matrix * base_covariances[index]
            formula = f"({formula} {operator} ({item['formula']}))"
            periods.extend(item["periods"])
        cases.append({"id": f"composite_{number + 1:02d}", "formula": formula,
                      "category": "composite", "family": "composite",
                      "periods": sorted(set(map(int, periods))), "covariance": matrix})
    return cases


def draw_retained_kernel_case(case: dict[str, Any], *, case_index: int,
                              global_seed: int = 20260930, n_train: int = 16,
                              n_test: int = 16) -> tuple[np.ndarray, np.ndarray]:
    factor = stable_factor(np.asarray(case["covariance"], dtype=float))
    rng = np.random.default_rng(global_seed + 100_000 + case_index)
    raw_train = draw_from_factor(factor, n_train, rng)
    raw_test = draw_from_factor(factor, n_test, rng)
    train, test, _, _ = standardize_train_test(raw_train, raw_test)
    return train, test


def retained_lmc_mechanisms(*, global_seed: int = 20260930, length: int = 1024
                            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray]]:
    bank = official_kernel_bank(length)
    latent_factors = [stable_factor(covariance_matrix(item["kernel"], length)) for item in bank]
    rng = np.random.default_rng(global_seed + 20_000)
    mechanisms = []
    for number in range(11):
        indices = list(range(3 * number, 3 * number + 3))
        alpha = (0.2, 0.5, 1.0, 2.0)[number % 4]
        mechanisms.append({"id": f"lmc_{number + 1:02d}", "latent_indices": indices,
                           "latent_ids": [bank[index]["id"] for index in indices],
                           "latent_formulas": [bank[index]["formula"] for index in indices],
                           "latent_periods": [list(bank[index]["periods"]) for index in indices],
                           "dirichlet_alpha": alpha,
                           "mixing": rng.dirichlet(alpha * np.ones(3), size=4)})
    return mechanisms, bank, latent_factors


def draw_retained_lmc_case(mechanism: dict[str, Any], latent_factors: list[np.ndarray], *,
                           mechanism_index: int, global_seed: int = 20260930,
                           n_train: int = 16, n_test: int = 16) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(global_seed + 200_000 + mechanism_index)
    raw_train_latents = np.stack([draw_from_factor(latent_factors[index], n_train, rng)
                                  for index in mechanism["latent_indices"]], axis=1)
    raw_test_latents = np.stack([draw_from_factor(latent_factors[index], n_test, rng)
                                 for index in mechanism["latent_indices"]], axis=1)
    latent_means = raw_train_latents.mean(axis=(0, 2))
    latent_scales = raw_train_latents.std(axis=(0, 2)) + 1e-12
    train_latents = (raw_train_latents - latent_means[None, :, None]) / latent_scales[None, :, None]
    test_latents = (raw_test_latents - latent_means[None, :, None]) / latent_scales[None, :, None]
    mixing = np.asarray(mechanism["mixing"], dtype=float)
    raw_train_channels = np.einsum("cq,nqt->ntc", mixing, raw_train_latents)
    raw_test_channels = np.einsum("cq,nqt->ntc", mixing, raw_test_latents)
    channel_means = raw_train_channels.mean(axis=(0, 1))
    channel_scales = raw_train_channels.std(axis=(0, 1)) + 1e-12
    return {
        "raw_train_latents": raw_train_latents, "raw_test_latents": raw_test_latents,
        "train_latents": train_latents, "test_latents": test_latents,
        "latent_means": latent_means, "latent_scales": latent_scales,
        "train_channels": (raw_train_channels - channel_means) / channel_scales,
        "test_channels": (raw_test_channels - channel_means) / channel_scales,
        "channel_means": channel_means, "channel_scales": channel_scales,
    }
