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


def public_sized_comparison(labels, earlier, candidate, fraction=.2, repetitions=1000, seed=2026):
    """Evaluate two rankings on identical random public-sized customer subsets."""
    labels = np.asarray(labels, dtype=np.uint8)
    if len(earlier) != len(labels) or len(candidate) != len(labels):
        raise ValueError('Both rankings must cover the same customers as the labels.')
    n = len(labels)
    size = int(np.ceil(fraction*n))
    k = int(np.ceil(.2*size))
    positions = np.arange(1,k+1)
    rankings = [np.argsort(-np.asarray(scores),kind='stable') for scores in [earlier,candidate]]
    rng = np.random.default_rng(seed)
    scores = np.empty((repetitions,2))
    selected = np.zeros(n,dtype=bool)
    for i in range(repetitions):
        selected.fill(False)
        selected[rng.choice(n,size,replace=False)] = True
        denominator = max(1,min(k,int(labels[selected].sum())))
        for j,ordering in enumerate(rankings):
            target = labels[ordering[selected[ordering]][:k]]
            scores[i,j] = (target.cumsum()/positions) @ target / denominator
    difference = scores[:,1]-scores[:,0]
    return dict(subset_size=size, top_k=k, candidate_mean=scores[:,1].mean(),
                candidate_p025=np.quantile(scores[:,1],.025), candidate_p975=np.quantile(scores[:,1],.975),
                delta_mean=difference.mean(), delta_p025=np.quantile(difference,.025),
                delta_p975=np.quantile(difference,.975), fraction_improved=(difference>0).mean())
