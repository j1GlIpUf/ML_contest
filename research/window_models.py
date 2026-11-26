"""Time-fold experiments on the 110 features selected inside each training fold."""

import numpy as np
from lightgbm import LGBMRanker
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


def forecast(fit, labels, query, weights=None):
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=190, max_leaf_nodes=15, min_samples_leaf=45,
        l2_regularization=6, early_stopping=False, random_state=42)
    with threadpool_limits(limits=4):
        model.fit(fit, labels, sample_weight=weights)
        return model.predict_proba(query)[:, 1]


def regions(core, split_far=True):
    due = -core.rhythm_order_all_median_delta
    single = core.n_orders.eq(1)
    regular = core.n_orders.le(3) | core.clock_volume_top_n.eq(core.n_orders)
    region = np.select([single, regular, due.between(-14, 104)], [0, 1, 2], default=3)
    if split_far:
        region[(region == 3) & due.gt(104)] = 4
    return region


def local_forecast(fit, labels, query, fit_region, query_region, outside_weight):
    scores = np.empty(len(query))
    for group in np.unique(query_region):
        local = fit_region == group
        held = query_region == group
        if outside_weight:
            weights = np.where(local, 1., outside_weight)
            scores[held] = forecast(fit, labels, query.loc[held], weights)
        else:
            scores[held] = forecast(fit.loc[local], labels.loc[local], query.loc[held])
    return scores


def stratified_rank(fit, labels, dates, query, size, seed):
    """Shuffle positive/negative customers independently within each training date."""
    rng = np.random.default_rng(seed)
    groups = []
    for date in np.unique(dates):
        ix = np.flatnonzero(dates == date)
        n = max(1, round(len(ix) / size))
        pos = np.array_split(rng.permutation(ix[labels.iloc[ix].to_numpy() == 1]), n)
        neg = np.array_split(rng.permutation(ix[labels.iloc[ix].to_numpy() == 0]), n)
        groups.extend(rng.permutation(np.concatenate(pair)) for pair in zip(pos, neg))
    ix = np.concatenate(groups)
    ranker = LGBMRanker(n_estimators=210, learning_rate=.045, num_leaves=15,
        max_bin=127, min_child_samples=65, colsample_bytree=.65, reg_lambda=6,
        lambdarank_truncation_level=round(.15 * size), device_type='cpu',
        random_state=seed, n_jobs=4, verbosity=-1)
    ranker.fit(fit.to_numpy()[ix], labels.iloc[ix], group=[len(g) for g in groups])
    return ranker.booster_.predict(query.to_numpy(), num_threads=4)


def reference_scale(raw, reference):
    """Assign the reference score distribution to the ranker's ordering; ties share a score."""
    values, inv, frequency = np.unique(raw, return_inverse=True, return_counts=True)
    midpoint = frequency.cumsum() - (frequency + 1) / 2
    return np.interp(midpoint, np.arange(len(reference)), np.sort(reference))[inv]


def compare_blocks(core, extra, train, date, select_by_gain, ap20, reference, stage='features'):
    """Repeat one completed ablation from the group notebook, returning scores and metrics.

    core and extra are indexed by customer_id; reference is indexed by held customer_id.
    The selector sees only rows whose scoring date precedes the evaluation date.
    """
    import pandas as pd
    frame = train.set_index('customer_id')
    earlier = frame[frame.scoring_date < date]
    later = frame[frame.scoring_date == date]
    cols = select_by_gain(core.loc[earlier.index], earlier.target, 110)
    data = core[cols].join(extra)
    ref = reference.reindex(later.index).to_numpy()
    if not np.isfinite(ref).all():
        raise ValueError('The reference must cover every evaluation customer.')
    y = earlier.target
    scores = {}
    if stage == 'features':
        blocks = {'core': (), 'cdf': ('risk_',), 'sessions': ('closed_',),
                  'fine': ('fine_',), 'market_basket': ('basket_', 'observed_', 'demand_'),
                  'window_all': tuple(extra.columns)}
        for name, prefixes in blocks.items():
            fields = [*cols, *[c for c in extra if c.startswith(prefixes)]]
            scores[name] = forecast(data.loc[earlier.index, fields], y, data.loc[later.index, fields])
    elif stage == 'regional':
        for five, weight in [(False, 0), (True, 0), (True, .25)]:
            group = pd.Series(regions(core, five), index=core.index)
            scores[f'groups{4 + five}_outside{weight}'] = local_forecast(
                data.loc[earlier.index], y, data.loc[later.index],
                group.loc[earlier.index].to_numpy(), group.loc[later.index].to_numpy(), weight)
    elif stage == 'ranking':
        fit, query = data.loc[earlier.index], data.loc[later.index]
        scale = forecast(fit, y, query)
        for size in [250, 500, 1000]:
            members = []
            for seed in ([13, 37, 71, 107, 151] if size == 250 else [13]):
                raw = stratified_rank(fit, y, earlier.scoring_date.to_numpy(), query, size, seed)
                members.append(reference_scale(raw, scale))
                if len(members) in [1, 5]:
                    scores[f'stratrank_{size}_s{len(members)}'] = np.mean(members, axis=0)
    else:
        raise ValueError(f'Unknown experiment: {stage}')
    rows = []
    for name, p in scores.items():
        for weight in [1., .5]:
            combined = weight * p + (1 - weight) * ref
            top = np.argsort(-combined, kind='stable')[:int(np.ceil(.2 * len(later)))]
            selected = later.iloc[top]
            medium = core.n_orders.reindex(selected.index).between(7, 15)
            rows.append(dict(date=str(pd.Timestamp(date).date()), model=name, weight=weight,
                ap20=ap20(later.target, combined), buyers=int(selected.target.sum()),
                false_7_15=int((selected.target.eq(0) & medium).sum())))
    return pd.DataFrame(rows), pd.DataFrame(scores, index=later.index)
