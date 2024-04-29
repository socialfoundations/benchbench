import random
import numpy as np
import pandas as pd


def load_random_benchmark(seed=0, num_task=100, num_model=100):
    np.random.seed(seed)
    random.seed(seed)
    data = np.random.random([num_model, num_task]) * 100
    data = pd.DataFrame(data)
    cols = list(data.columns)
    return data, cols


def load_constant_benchmark(seed=0, num_task=100, num_model=100):
    np.random.seed(seed)
    random.seed(seed)
    rd = np.random.random([num_model, 1])
    data = np.concatenate([rd.copy() for _ in range(num_task)], axis=1) * 100
    data = pd.DataFrame(data)
    cols = list(data.columns)
    return data, cols


def load_interpolation_benchmark(seed=0, mix_ratio=0.0, num_task=100, num_model=100):
    num_random = int(mix_ratio * num_task + 0.5)
    num_constant = int((1 - mix_ratio) * num_task + 0.5)
    if num_random == 0:
        return load_constant_benchmark(
            seed=seed, num_task=num_constant, num_model=num_model
        )
    elif num_constant == 0:
        return load_random_benchmark(
            seed=seed, num_task=num_random, num_model=num_model
        )
    else:
        random = load_random_benchmark(
            seed=seed, num_task=num_random, num_model=num_model
        )[0]
        constant = load_constant_benchmark(
            seed=seed, num_task=num_constant, num_model=num_model
        )[0]
        data = pd.DataFrame(np.concatenate([random.values, constant.values], axis=1))
        cols = list(data.columns)
        return data, cols
