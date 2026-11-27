"""Forecast the target quarter by replaying complete historical shopping trajectories."""

import numpy as np
import pandas as pd


def trajectory_features(b, horizon=90):
    # Age increases into the past. Stable order within a shopping day makes all
    # suffixes reproducible. Only rows of this customer precede/extend a suffix.
    rows = b.sort_values(['customer_id', 'age', 'order_id'], ascending=[True, True, False]).copy()
    uid = rows.customer_id.to_numpy()
    age = rows.age.to_numpy(dtype=float)
    starts = np.r_[0, np.flatnonzero(uid[1:] != uid[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(rows)])
    origin = np.repeat(starts, sizes)
    position = np.arange(len(rows)) - origin
    recency = age[origin]
    search = age - recency
    # Separate customer age scales by a stride, allowing a single sorted search.
    stride = int(age.max()) + 2 * horizon + 20
    offset = np.repeat(np.arange(len(starts)) * stride, sizes)
    sorted_age = age + offset
    ends = origin + np.repeat(sizes, sizes)
    def insertion(days):
        pos = np.searchsorted(sorted_age, days + offset, side='left')
        return np.clip(pos, origin, ends)
    def nearest(days):
        pos = insertion(days)
        newer = np.maximum(origin, pos - 1)
        older = np.minimum(ends - 1, pos)
        return np.where(abs(age[newer] - days) <= abs(age[older] - days), newer, older)
    start = insertion(search - horizon + 1)
    stop = insertion(search + 1)
    mid = nearest(search - (horizon - 1) / 2)
    due = search - age[mid]
    common = pd.DataFrame({'customer_id': uid, 'pos': position, 'age': age,
        'nearest_due': due, 'centre_distance': abs(due - (horizon - 1) / 2),
        'known': search >= horizon + 13, 'hits': stop - start,
        'early': insertion(search + 14) - stop,
        'late': start - insertion(search - horizon - 13),
        'period': search, 'next_due': search - age[np.maximum(origin, np.arange(len(age)) - 1)]})
    common['kernel'] = np.exp(-.5 * (common.centre_distance / 18) ** 2)
    common['hit'] = common.hits.gt(0)
    common['observable'] = common.hit | common.known
    out = []
    for key in ['main', 'mode', 'signature', 'category2', 'gender_product']:
        cat = pd.Categorical(rows[key]).codes
        same = np.ones(len(rows), dtype=bool)
        for lag in range(4):
            member = np.minimum(np.arange(len(rows)) + lag, ends - 1)
            last = np.minimum(origin + lag, ends - 1)
            same &= (position + lag < np.repeat(sizes, sizes)) & (cat[member] == cat[last])
            if key in ['mode', 'category2', 'gender_product'] and lag > 1:
                break
            if key == 'main' and lag == 0:
                out.append(aggregate_anchors(common.loc[position > 0], 'any'))
            sub = common.loc[same & (position > 0)]
            out.append(aggregate_anchors(sub, f'{key}_{lag + 1}'))
    return pd.concat(out, axis=1)


def aggregate_anchors(sub, name):
    g = sub.groupby('customer_id', sort=False)
    stats = g.agg(support=('age', 'size'), kernel=('kernel', 'mean'), hit_fraction=('hit', 'mean'),
        hits=('hits', 'mean'), early=('early', 'mean'), late=('late', 'mean'),
        completed=('known', 'sum'), usable=('observable', 'sum'),
        first_due=('next_due', 'median'), best_distance=('centre_distance', 'min'))
    stats['known_fraction'] = g.hit.sum() / stats.usable.clip(lower=1)
    latest = g.head(1).set_index('customer_id')
    for c in ['next_due', 'nearest_due', 'period', 'hits', 'early', 'late', 'known', 'pos']:
        stats['recent_' + c] = latest[c].astype(float)
    return stats.add_prefix('trajectory_' + name + '_')
