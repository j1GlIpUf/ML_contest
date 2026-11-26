"""Personal sequence forecasts over category baskets and their historical continuations."""

import numpy as np
import pandas as pd


def baskets(orders, cohort):
    h = orders.merge(cohort[['customer_id', 'scoring_date']], on='customer_id', validate='many_to_one')
    h['age'] = (h.scoring_date - h.order_date).dt.days
    h = h[h.age >= 14].copy()
    ids = ['customer_id', 'order_id', 'age']
    by = h.groupby(ids + ['category']).agg(lines=('product_id', 'size'), amount=('paid_price', 'sum')).reset_index()
    b = by.groupby(ids).agg(ncats=('category', 'size'), lines=('lines', 'sum')).reset_index()
    for name, sort in [('main', ['lines', 'amount', 'category']), ('mode', ['lines', 'category'])]:
        first = by.sort_values(sort, ascending=[False] * (len(sort) - 1) + [True]).drop_duplicates(ids)
        b = b.merge(first[ids + ['category']].rename(columns={'category': name}), on=ids, validate='one_to_one')
    cats = pd.Categorical(by.category, categories=sorted(h.category.unique()))
    by['bit'] = 1 << cats.codes.astype('int64')
    b = b.merge(by.groupby(ids).bit.sum().rename('signature'), on=ids, validate='one_to_one')
    for key in ['category2', 'brand', 'gender_product']:
        states = h.groupby(ids + [key]).agg(lines=('product_id', 'size'), value=('paid_price', 'sum')).reset_index()
        main = states.sort_values(['lines', 'value', key], ascending=[False, False, True]).drop_duplicates(ids)
        b = b.merge(main[ids + [key]], on=ids, validate='one_to_one')
    return b.sort_values(['customer_id', 'age', 'order_id'], ascending=[True, False, True])


def latest_categories(b):
    fields = ['main', 'category2', 'brand', 'gender_product']
    g = b.groupby('customer_id')
    last = g[fields].last()
    for key in ['main', 'category2', 'gender_product']:
        prev = g[key].shift().groupby(b.customer_id).last().fillna('<start>')
        last[key + '_pair'] = prev + ' > ' + last[key]
    return last.astype(str).add_prefix('token_')


def replay(b, key):
    """Retrieve the last suffix in older history and replay up to four successor dates."""
    b = b[['customer_id', 'age', key]].copy()
    g = b.groupby('customer_id')
    b['next'] = g[key].shift(-1)
    b['prev'] = g[key].shift()
    b['prev2'] = g[key].shift(2)
    recent = g.tail(1).set_index('customer_id')
    b['recency'] = b.customer_id.map(recent.age)
    # A historical continuation after each anchor is measured relative to that
    # anchor. Its end must be observed, including when the customer stops ordering.
    ar = np.column_stack([b.age - g.age.shift(-j) - b.recency for j in range(1, 5)])
    known = np.isfinite(ar)
    b['hits'] = ((ar >= 0) & (ar < 90)).sum(axis=1)
    b['near'] = np.where(known, abs(ar - 45), np.inf).min(axis=1)
    b['complete'] = (b.age - b.recency) >= b.recency + 90
    b['first'] = ar[:, 0]
    same = b[key].eq(b.customer_id.map(recent[key]))
    pair = same & b.prev.eq(b.customer_id.map(recent.prev))
    triple = pair & b.prev2.eq(b.customer_id.map(recent.prev2))
    # The pattern of repeats (e.g. A-B-A) can match even with different literal categories.
    repeats = b[key].eq(b.prev2) & b[key].ne(b.prev)
    same_shape = repeats.eq(b.customer_id.map(recent[key].eq(recent.prev2) & recent[key].ne(recent.prev)))
    same_shape &= b[key].eq(b.prev).eq(b.customer_id.map(recent[key].eq(recent.prev)))
    out = []
    for name, mask in [('triple', triple), ('pair', pair), ('shape', same_shape)]:
        z = b.loc[mask & b.next.notna()].copy()
        z['success'] = z.hits.gt(0).where(z.complete | z.hits.gt(0))
        z['kernel'] = np.exp(-.5 * (z.near / 20) ** 2)
        a = z.groupby('customer_id').agg(n=('first', 'count'), mean=('first', 'mean'), last=('first', 'last'),
            closest=('near', 'min'), density=('kernel', 'mean'), chance=('success', 'mean'), hits=('hits', 'mean'))
        out.append(a.add_prefix(f'replay_{key}_{name}_'))
    return pd.concat(out, axis=1)


def candidate_calendar(b, key):
    """Forecast each next category under personal transition counts with corpus backoff."""
    b = b[['customer_id', 'age', key]].copy()
    g = b.groupby('customer_id')
    b['src'] = g[key].shift()
    b['older'] = g[key].shift(2)
    b['interval'] = -g.age.diff()
    b['return'] = -b.groupby(['customer_id', key]).age.diff()
    cat = b.groupby(['customer_id', key]).agg(count=('age', 'size'), last_age=('age', 'min'),
        cycle=('return', 'median'), last_cycle=('return', 'last')).reset_index()
    latest = g.tail(1).set_index('customer_id')
    period = g.interval.median()
    cat['personal'] = cat.customer_id.map(period)
    cat['recency'] = cat.customer_id.map(latest.age)
    cat['src'] = cat.customer_id.map(latest[key])
    cat['older'] = cat.customer_id.map(latest.src)
    # Backoff uses only order histories in this scoring cohort, without labels.
    corpus = b.groupby(['src', key]).size().rename('population')
    cat = cat.join(corpus, on=['src', key])
    historic = b[b.src.eq(b.customer_id.map(latest[key]))]
    one = historic.groupby(['customer_id', key]).interval.agg(one='size', typical_step='median')
    two = historic[historic.older.eq(historic.customer_id.map(latest.src))].groupby(['customer_id', key]).size()
    cat = cat.join(one, on=['customer_id', key]).join(two.rename('two'), on=['customer_id', key])
    grp = cat.groupby('customer_id')
    pop = cat.population.fillna(0) + 1
    pop /= pop.groupby(cat.customer_id).transform('sum')
    cat['posterior'] = (pop + cat.one.fillna(0) + 2 * cat.two.fillna(0))
    cat['posterior'] /= grp.posterior.transform('sum')
    cat['interval_per_visit'] = cat.personal * grp['count'].transform('sum') / cat['count']
    # Give more support to the personal category estimate with more observed returns.
    weight = (cat['count'] - 1) / (cat['count'] + 1)
    cat['pooled'] = cat.cycle.fillna(0) * weight + cat.interval_per_visit * (1 - weight)
    cat['fallback'] = cat.cycle.where(cat['count'] >= 3, cat.personal)
    out = []
    for name, due in {
        'global_since_category': cat.personal - cat.last_age,
        'shrunk_since_category': cat.pooled - cat.last_age,
        'since_last_order': cat.pooled - cat.recency,
        'supported': cat.fallback - cat.last_age,
        'last_cycle': cat.last_cycle.fillna(cat.personal) - cat.last_age,
        'route': cat.typical_step.fillna(cat.personal) - cat.recency,
    }.items():
        cat['due'] = due
        cat['weighted'] = np.exp(-.5 * ((due - 45) / 22) ** 2) * cat.posterior
        cat['in_window'] = due.between(0, 90) * cat.posterior
        cat['edge'] = (due.between(-14, 0) | due.between(90, 104)) * cat.posterior
        mode = cat.sort_values(['posterior', 'count', 'last_age'], ascending=[False, False, True]).drop_duplicates('customer_id')
        a = grp.agg(intensity=('weighted', 'sum'), hit=('in_window', 'sum'), edge=('edge', 'sum'))
        a['mode_due'] = mode.set_index('customer_id').due
        out.append(a.add_prefix(f'candidate_{key}_{name}_'))
    return pd.concat(out, axis=1)


def weighted_gaps(b):
    ids = b.customer_id
    age = b.groupby('customer_id').age.min()
    out = []
    for key in ['lines', 'ncats']:
        # A receipt containing k distinct states (or k items) contributes k-1
        # additional zero intervals, unlike one observation for the entire basket.
        j = b.loc[b.index.repeat(b[key]), ['customer_id', 'age']].copy()
        j['gap'] = -j.groupby('customer_id').age.diff()
        for width in [180, 365, 745]:
            z = j[j.age <= width + 14]
            stats = z.groupby('customer_id').gap.agg(['mean', 'median'])
            stats = stats.sub(age, axis=0)
            out.append(stats.add_prefix(f'lineclock_{key}_{width}_'))
    return pd.concat(out, axis=1)


def sequence_features(orders, cohort):
    numeric, text = [], []
    for _, rows in cohort.groupby('scoring_date'):
        b = baskets(orders, rows)
        parts = [weighted_gaps(b)]
        for key in ['main', 'mode', 'signature']:
            parts += [replay(b, key), candidate_calendar(b, key)]
        numeric.append(pd.concat(parts, axis=1))
        text.append(latest_categories(b))
    return (pd.concat(numeric).reindex(cohort.customer_id).astype('float32'),
            pd.concat(text).reindex(cohort.customer_id))
