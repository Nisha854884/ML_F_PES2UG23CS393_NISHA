import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    """
    target = data[:, -1]  # last column = target
    classes, counts = np.unique(target, return_counts=True)
    probs = counts / counts.sum()
    # Avoid log2(0) by ensuring probs > 0
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
    return float(entropy)


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    """
    values, counts = np.unique(data[:, attribute], return_counts=True)
    total = len(data)
    avg_info = 0.0

    for v, c in zip(values, counts):
        subset = data[data[:, attribute] == v]
        subset_entropy = get_entropy_of_dataset(subset)
        avg_info += (c / total) * subset_entropy

    return float(avg_info)


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    """
    entropy_dataset = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    info_gain = entropy_dataset - avg_info
    return round(info_gain, 4)


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.
    """
    gains = {}
    for attr in range(data.shape[1] - 1):  # exclude target column
        gains[attr] = get_information_gain(data, attr)

    best_attr = max(gains, key=gains.get)
    return gains, best_attr
