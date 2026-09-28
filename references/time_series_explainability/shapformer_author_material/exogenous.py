import numpy as np
import torch


CATEGORICAL_FEATURES = {
    "hourofday",
    "dayofweek",
    "month",
    "holiday"
}


class ExogenousDataGenerator:
    def __init__(self, noise=0.05, noise_features: int = 0):
        self.noise = noise
        self.noise_features = noise_features
        self.features = [
            "load",
            "hourofday",
            "dayofweek",
            "month",
            "holiday",
            "temperature"
        ]
        for i in range(1, noise_features + 1):
            self.features.append(f"noise{i}")

    def _base_day_load(self):
        hours = np.arange(24)
        x = hours * 2 * np.pi / 24
        load = np.sin(x - 0.5 * np.pi)
        return load

    def _sample_day(self):
        factor = np.random.uniform(0.5, 1.0)
        alpha = np.random.uniform(-0.5, 0.5)
        beta = np.random.uniform(-0.5, 0.5)
        load = self._base_day_load()
        hours = np.arange(24)
        x = hours * 2 * np.pi / 24
        load += alpha * np.sin(x)
        load += beta * np.cos(x)
        load *= factor
        return load

    def _add_noise(self, x, level):
        return x + np.random.normal(0, level, size=x.shape)

    def _generate_temperature(self):
        start = np.random.uniform()
        vals = np.random.normal(0, 0.02, size=336)
        vals = start + np.cumsum(vals)
        return vals

    def _month_base_load(self, month):
        month_base_load = 0.1 * np.sin(month * 2 * np.pi / 12)
        return month_base_load

    def _sample_holiday(self):
        return np.random.uniform() < 0.1

    def _generate_load(self, start_hour, start_dayofweek, month):
        const_base_load = np.random.uniform(-0.5, 0.5)
        base_load = const_base_load + self._month_base_load(month)
        day_pattern = self._sample_day()
        normal_day = base_load + self._add_noise(day_pattern, 0.1)
        saturday_factor = np.random.uniform(0.5, 0.9)
        special_day_factor = np.random.uniform(0.2, saturday_factor)
        saturday = self._add_noise(saturday_factor * normal_day, 0.1)
        special_day = self._add_noise(special_day_factor * normal_day, 0.1)
        load_profile = (normal_day, saturday, special_day)
        load = np.zeros(336)
        holidays = np.zeros(336, dtype=int)
        hour = start_hour
        dow = start_dayofweek
        is_holiday = self._sample_holiday()
        for i in range(336):
            if hour == 24:
                hour = 0
                is_holiday = self._sample_holiday()
                dow = (dow + 1) % 7
            if is_holiday:
                holidays[i] = 1
            if dow == 6 or is_holiday:
                load[i] = special_day[hour]
            elif dow == 5:
                load[i] = saturday[hour]
            else:
                load[i] = normal_day[hour]
            hour += 1
        return load, holidays, const_base_load, load_profile

    def generate(self, item=None):
        start_hour = np.random.randint(0, 24)
        start_dayofweek = np.random.randint(0, 7)
        hour_of_week = (start_dayofweek * 24 + start_hour + np.arange(0, 336)) % 168
        day = hour_of_week // 24
        hour = hour_of_week % 24
        month = np.random.randint(0, 12)
        load, holidays, base_load, load_profile = self._generate_load(start_hour, start_dayofweek, month)
        month = np.repeat(month, 336)
        temperature = self._generate_temperature()
        load = load * (0.5 + 0.5 * temperature)
        noise = self._add_noise(np.zeros_like(load), self.noise)
        load = load + noise
        x_past = {
            "load": load[:168],
            "hourofday": hour[:168],
            "dayofweek": day[:168],
            "month": month[:168],
            "holiday": holidays[:168],
            "temperature": temperature[:168]
        }
        x_future = {
            "hourofday": hour[168:],
            "dayofweek": day[168:],
            "month": month[168:],
            "holiday": holidays[168:],
            "temperature": temperature[168:]
        }
        components = {
            "base_load": base_load,
            "month": month[0],
            "hourofday": start_hour,
            "dayofweek": start_dayofweek,
            "load_profile": load_profile,
            "holidays": holidays,
            "temperature": temperature,
            "noise": noise
        }
        for noise_i in range(1, self.noise_features + 1):
            x_past[f"noise{noise_i}"] = np.random.uniform(size=168)
            x_future[f"noise{noise_i}"] = np.random.uniform(size=168)
        y = load[168:]
        return x_past, x_future, y, components

    def get_features(self):
        return self.features


class ExogenousFeatureDataset:
    def __init__(self, generator, device: str, n_samples: int = None):
        self.generator = generator
        self.n = len(self.generator) if n_samples is None else n_samples
        self.x_enc, self.x_dec, self.y = self._generate()
        self._to(device)

    def _generate(self):
        features = self.generator.get_features()

        X_enc = {feature: [] for feature in features}
        X_dec = {feature: [] for feature in features[1:]}
        Y = []

        for i in range(self.n):
            x_enc, x_dec, y = self.generator.generate(i)[:3]

            X_enc["load"].append(torch.Tensor(x_enc["load"]))

            for feature in features[1:]:
                if feature in CATEGORICAL_FEATURES:
                    x_enc_tensor = torch.IntTensor(x_enc[feature])
                    x_dec_tensor = torch.IntTensor(x_dec[feature])
                else:
                    x_enc_tensor = torch.Tensor(x_enc[feature])
                    x_dec_tensor = torch.Tensor(x_dec[feature])
                X_enc[feature].append(x_enc_tensor)
                X_dec[feature].append(x_dec_tensor)

            Y.append(torch.Tensor(y))

        for feature in features:
            X_enc[feature] = torch.stack(X_enc[feature])
        for feature in features[1:]:
            X_dec[feature] = torch.stack(X_dec[feature])
        Y = torch.stack(Y)

        return X_enc, X_dec, Y

    def _to(self, device: str):
        for ft in self.x_enc:
            self.x_enc[ft] = self.x_enc[ft].to(device)
        for ft in self.x_dec:
            self.x_dec[ft] = self.x_dec[ft].to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        x_enc = {ft: self.x_enc[ft][item].unsqueeze(0) for ft in self.x_enc}
        x_dec = {ft: self.x_dec[ft][item].unsqueeze(0) for ft in self.x_dec}
        y = self.y[item].unsqueeze(0)
        return x_enc, x_dec, y

    def get_date(self, item):
        return self.generator.get_date(item)


def create_batch(dataset, sample_indices):
    x_enc = []
    x_dec = []
    y = []
    for i in sample_indices:
        x_enc_i, x_dec_i, y_i = dataset[i]
        x_enc.append(x_enc_i)
        x_dec.append(x_dec_i)
        y.append(y_i)
    x_enc = {key: torch.concatenate([x[key] for x in x_enc], dim=0) for key in x_enc[0].keys()}
    x_dec = {key: torch.concatenate([x[key] for x in x_dec], dim=0) for key in x_dec[0].keys()}
    y = torch.concatenate(y, dim=0)
    return x_enc, x_dec, y
