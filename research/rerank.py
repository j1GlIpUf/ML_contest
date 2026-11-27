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


def medium_order_correction(fit, y, oof, query, prior):
    """Estimate a correction for 7–15 orders from earlier out-of-sample residuals."""
    def strata(frame, score):
        due = -frame.rhythm_order_all_median_delta
        phase = np.select([due.lt(-14), due.lt(15), due.le(75), due.le(104)], range(4), default=4)
        repeats = frame.motif_mode_3_support.fillna(0).clip(upper=2)
        band = np.digitize(score, [.5, .7, .85])
        return pd.DataFrame({'phase': phase, 'repeats':repeats, 'band':band}, index=frame.index)
    selected = fit.n_orders.between(7, 15)
    training = strata(fit, oof).loc[selected].copy()
    training['error'] = (y - oof).loc[selected]
    training['variance'] = (oof * (1 - oof)).loc[selected]
    target = strata(query, prior)
    adjustment = pd.Series(0., index=query.index)
    for keys in [['phase','band'], ['phase','repeats','band']]:
        rates = training.groupby(keys)[['error','variance']].sum()
        delta = (rates.error / (rates.variance + 10)).clip(-.8,.8)
        adjustment += .5 * delta.reindex(pd.MultiIndex.from_frame(target[keys])).fillna(0).to_numpy()
    adjustment *= query.n_orders.between(7, 15)
    return expit(logit(prior.clip(1e-5, 1-1e-5)) + adjustment)


def boundary_pairs(fit, y, oof, dates, query, prior):
    """Train a residual ranker on plausible customer swaps around the marketing cutoff."""
    import lightgbm as lgb
    rng = np.random.default_rng(73)
    labels = y.to_numpy()
    base = logit(np.clip(np.asarray(oof), 1e-5, 1 - 1e-5))
    positives, negatives, masses = [], [], []
    for date in pd.unique(dates):
        ix = np.flatnonzero(np.asarray(dates) == date)
        rank = ix[np.argsort(-base[ix], kind='stable')]
        k = int(np.ceil(.2 * len(ix)))
        cutoff_neg = rank[:2*k][labels[rank[:2*k]] == 0]
        for i, row in enumerate(rank[:3*k]):
            if not labels[row]:
                continue
            local = rank[max(0,i-k//3):min(3*k,i+k//3)]
            local_neg = local[labels[local] == 0]
            pool = np.r_[rng.choice(cutoff_neg, 8),rng.choice(local_neg, 8) if len(local_neg) else []].astype(int)
            positives.extend([row]*len(pool))
            negatives.extend(pool)
            masses.extend([1 / np.sqrt(.2 + i/k)] * len(pool))
    pos,neg = np.array(positives),np.array(negatives)
    w = np.asarray(masses)
    w *= len(fit) / w.sum()
    def objective(raw, dataset):
        s = raw + base
        chance = expit(s[neg] - s[pos])
        grad = np.bincount(neg,w*chance,minlength=len(fit))-np.bincount(pos,w*chance,minlength=len(fit))
        curv = w*chance*(1-chance)
        hess = np.bincount(neg,curv,minlength=len(fit))+np.bincount(pos,curv,minlength=len(fit))
        point = expit(s)
        return grad + .5*(point-labels), hess + .5*point*(1-point)
    model = lgb.train(dict(objective=objective, learning_rate=.035,num_leaves=8,
                    min_data_in_leaf=125,lambda_l2=20,feature_fraction=.65,
                    num_threads=4,verbosity=-1,seed=73), lgb.Dataset(fit.to_numpy(),label=labels),
                    num_boost_round=170)
    delta = model.predict(query.to_numpy(),num_threads=4)
    return expit(logit(np.clip(np.asarray(prior),1e-5,1-1e-5))+delta)
