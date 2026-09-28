import pickle
import numpy as np
from scipy.special import binom
from tqdm import tqdm

from exogenous import ExogenousDataGenerator


def binary(num: int, digits: int = 7):
    return [bit == "1" for bit in bin(num)[2:].zfill(digits)]


def generate_load(load_profile, day_of_week, hour_of_day, holidays):
    load = np.zeros(336)
    is_holiday = holidays[0] if holidays is not None else np.random.uniform() < 0.1
    for i in range(336):
        if hour_of_day == 24:
            hour_of_day = 0
            day_of_week = (day_of_week + 1) % 7
            is_holiday = holidays[i] if holidays is not None else np.random.uniform() < 0.1
        if is_holiday or day_of_week == 6:
            load[i] = load_profile[2][hour_of_day]
        elif day_of_week == 5:
            load[i] = load_profile[1][hour_of_day]
        else:
            load[i] = load_profile[0][hour_of_day]
        hour_of_day += 1
    return load


def generate_from_components(components):
    load_profile = components["load_profile"]
    holidays = components["holidays"]
    temperature = components["temperature"]
    load = generate_load(load_profile, components["dayofweek"], components["hourofday"], holidays)
    load = load * (0.5 + 0.5 * temperature)
    load = load + components["noise"]
    return load


def generate_alternative_sample(generator, components, known_components):
    if "load_profile" not in known_components:
        const_base_load = np.random.uniform(-0.5, 0.5)
        if "month" in known_components:
            month = components["month"]
        else:
            month = np.random.randint(0, 12)
        base_load = const_base_load + generator._month_base_load(month)
        day_pattern = generator._sample_day()
        normal_day = base_load + generator._add_noise(day_pattern, 0.1)
        saturday_factor = np.random.uniform(0.5, 0.9)
        special_day_factor = np.random.uniform(0.2, saturday_factor)
        saturday = generator._add_noise(saturday_factor * normal_day, 0.1)
        special_day = generator._add_noise(special_day_factor * normal_day, 0.1)
        load_profile = (normal_day, saturday, special_day)
        day_of_week = components["dayofweek"] if "dayofweek" in known_components else np.random.randint(0, 7)
        hour_of_day = components["hourofday"] if ("hourofday" in known_components or "dayofweek" in known_components or ("holidays" in known_components and np.sum(components["holidays"]) > 0)) else np.random.randint(0, 24)
        holidays = components["holidays"] if "holidays" in known_components else None
        load = generate_load(load_profile, day_of_week, hour_of_day, holidays)
    else:
        load_profile = components["load_profile"]
        holidays = components["holidays"] if "holidays" in known_components else None
        load = generate_load(load_profile, components["dayofweek"], components["hourofday"], holidays)
    if "temperature" in known_components:
        temperature = components["temperature"]
    else:
        if "load_profile" in known_components:
            temperature = np.zeros(336)
            temperature[:168] = components["temperature"][:168]
            steps = np.random.normal(0, 0.02, size=168)
            temperature[168:] = temperature[167] + np.cumsum(steps)
        else:
            temperature = generator._generate_temperature()
    load = load * (0.5 + 0.5 * temperature)
    load += components["noise"]
    return load[-168:]


if __name__ == "__main__":
    plot_dir = "plots/ground_truth/"

    component_names = ["month", "hourofday", "dayofweek", "load_profile", "holidays", "temperature"]

    with open("data/synthetic_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    dataset = dataset[-10000:]  # test data

    generator = ExogenousDataGenerator(noise_features=2)

    component_importances = {comp: [] for comp in component_names}

    np.random.seed(42)
    #sample_indices = sorted(np.random.choice(list(range(len(dataset))), 1000, replace=False))
    sample_indices = [35, 2753, 3154]
    ground_truth_explanations = []

    for sample_i in tqdm(sample_indices):
        sample = dataset[sample_i]
        x_enc, x_dec, y, components = sample

        load = generate_from_components(components)

        coalitions = {}

        for i in range(2**len(component_names)):
            mask = binary(i, digits=len(component_names))
            known_components = set(component_names[index] for index, mask_bit in enumerate(mask) if mask_bit)
            alt_samples = []
            for _ in range(1000):
                alt_sample = generate_alternative_sample(generator, components, known_components)
                alt_samples.append(alt_sample)
            alt_sample = np.mean(alt_samples, axis=0)
            coalitions[tuple(sorted(known_components))] = alt_sample

        n_components = len(component_names)
        var_impacts = {var: np.zeros(168) for var in component_names}
        for coalition, alt_sample in coalitions.items():
            for var in coalition:
                coalition_size = len(coalition) - 1
                weight = 1 / binom(n_components - 1, coalition_size)
                smaller_coalition = set(coalition)
                smaller_coalition.remove(var)
                smaller_coalition = tuple(sorted(smaller_coalition))
                marginal = coalitions[coalition] - coalitions[smaller_coalition]
                var_impacts[var] += marginal * weight / n_components

        for var, impact in var_impacts.items():
            importance = np.mean(np.abs(impact))
            print(f"{var} importance: {importance}")
            component_importances[var].append(importance)

        start_day = int(components["dayofweek"])
        start_hour = int(components["hourofday"])

        probe = np.sum([var_impacts[var] for var in component_names], axis=0) + coalitions[tuple()]
        print("difference:", np.abs(probe - y).sum())

        ground_truth_explanations.append({
            "time": sample_i,
            "x_enc": x_enc,
            "x_dec": x_dec,
            "y": y,
            "prediction": y,
            "dummy_prediction": components["noise"],
            "feature_names": ["HOD", "DOW", "month", "holiday", "T", "load"],
            "shapley_values": {
                0: var_impacts["hourofday"],
                1: var_impacts["dayofweek"],
                2: var_impacts["month"],
                3: var_impacts["holidays"],
                4: var_impacts["temperature"],
                5: var_impacts["load_profile"]
            }
        })

    with open("ground_truth_explanations.pkl", "wb") as f:
        pickle.dump(ground_truth_explanations, f)

    sum_importances = np.sum([np.mean(importances) for importances in component_importances.values()])
    for var, importances in component_importances.items():
        importance = np.mean(importances)
        percentage = importance / sum_importances * 100
        print(f"{var} importance: {importance:.4f} ({percentage:.2f}%)")
