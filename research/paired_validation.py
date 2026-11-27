"""Paired bootstrap for the change in AP@20% within one scoring month."""

import numpy as np


def compare_predictions(labels, earlier, candidate, repetitions=1000, seed=2026):
    labels = np.asarray(labels, dtype=np.uint8)
    previous_order = np.argsort(-np.asarray(earlier), kind='stable')
    candidate_order = np.argsort(-np.asarray(candidate), kind='stable')
    n = len(labels)
    k = int(np.ceil(.2 * n))
    positions = np.arange(1, k + 1)
    rng = np.random.default_rng(seed)
    difference = np.empty(repetitions)
    for i in range(repetitions):
        copies = np.bincount(rng.integers(0, n, size=n), minlength=n)
        denominator = max(1, min(k, int(copies @ labels)))
        def score(order):
            target = np.repeat(labels[order], copies[order])[:k]
            return ((target.cumsum() / positions) @ target) / denominator
        difference[i] = score(candidate_order) - score(previous_order)
    return np.quantile(difference, [.025, .975])
