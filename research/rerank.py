"""Comparisons against a fixed customer ranking and corrections trained on OOF scores."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit, logit

from research.window_models import forecast


def compare(labels, candidate, reference, order_count):
    ids = labels.index
    ranked = np.argsort(-candidate.reindex(ids).to_numpy(), kind='stable')
    old = np.argsort(-reference.reindex(ids).to_numpy(), kind='stable')
    k = int(np.ceil(.2 * len(ids)))
    top, previous = ranked[:k], old[:k]
    y = labels.to_numpy()
    added = np.setdiff1d(top, previous)
    dropped = np.setdiff1d(previous, top)
    hit = y[top]
    ap = ((hit.cumsum() / np.arange(1, k + 1)) @ hit) / max(1, min(k, y.sum()))
    medium = order_count.reindex(ids).between(7, 15).to_numpy()
    return dict(ap20=ap, buyers=int(hit.sum()), changed=len(added), added_buyers=int(y[added].sum()),
                dropped_buyers=int(y[dropped].sum()), false_7_15=int(((hit == 0) & medium[top]).sum()))


def rerank(fit, y, query, oof, prior, dates, method):
    # Rows have predictions from a model that had no access to their own label.
    percentile = oof.groupby(dates).rank(pct=True)
    weights = .25 + .75 * percentile.ge(.60)
    if method == 'weighted':
        return forecast(fit, y, query, weights=weights)
    model = LGBMClassifier(n_estimators=180, learning_rate=.035, num_leaves=9, min_child_samples=120,
                           reg_lambda=12, colsample_bytree=.65, verbosity=-1, random_state=42, n_jobs=4)
    base = logit(np.clip(oof.to_numpy(), 1e-5, 1 - 1e-5))
    model.fit(fit.to_numpy(), y, init_score=base, sample_weight=weights)
    correction = model.booster_.predict(query.to_numpy(), raw_score=True, num_threads=4)
    return expit(logit(np.clip(np.asarray(prior), 1e-5, 1 - 1e-5)) + correction)
