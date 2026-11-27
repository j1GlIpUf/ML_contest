"""Time-local support and time-aligned historical continuations of basket sequences."""
import numpy as np
import pandas as pd


def local_context(baskets, horizon):
    b = baskets.sort_values(['customer_id', 'age', 'order_id'], ascending=[True, False, True]).copy()
    g = b.groupby('customer_id', sort=False)
    pos = g.cumcount()
    backwards = g.cumcount(ascending=False)
    last = g.age.transform('min')
    period = (-g.age.diff()).groupby(b.customer_id).transform('median')
    b['due'] = period - last
    b['next_wait'] = b.age - g.age.shift(-1)
    b['next_age'] = g.age.shift(-1)
    b['is_recent'] = backwards
    parts = []
    for key in ['mode', 'category2', 'gender_product']:
        states = pd.Categorical(b[key]).codes.astype('int64') + 1
        width = int(states.max()) + 1
        b['sequence'] = states
        for length in range(1, 5 if key == 'mode' else 3):
            if length > 1:
                b['sequence'] = g.sequence.shift().fillna(0).astype('int64') * width + states
            enough = pos.ge(length - 1)
            matched = enough & b.sequence.eq(g.sequence.transform('last')) & backwards.gt(0)
            start = g.age.shift(length - 1)
            span = start - b.age
            latest_span = span.groupby(b.customer_id).transform('last')
            # Align a historical sequence on its beginning, its end, or by its observed speed.
            ratio = ((latest_span + 7) / (span + 7)).clip(.2, 5)
            z = b.loc[matched, ['customer_id', 'age', 'is_recent', 'due', 'next_wait']].copy()
            z['similarity'] = np.exp(-abs(np.log(ratio.loc[matched])))
            z['end_due'] = z.next_wait - last[matched]
            z['begin_due'] = z.end_due + span[matched] - latest_span[matched]
            z['speed_due'] = z.next_wait * ratio[matched] - last[matched]
            for days, suffix in [(180, 'half'), (365, 'year'), (None, 'last8'), (None, 'last16')]:
                observed = (start.le(13 + days) if days else (backwards + length - 1).lt(8 if suffix == 'last8' else 16))
                recent = z.loc[observed[matched]]
                a = recent.groupby('customer_id').agg(n=('age', 'size'), last_due=('end_due', 'last'),
                    usual_due=('end_due', 'median'), moved=('speed_due', 'median'))
                opportunities = (observed & enough & backwards.gt(0)).groupby(b.customer_id).sum()
                a = a.reindex(opportunities.index)
                a['n'] = a.n.fillna(0)
                a['rate'] = (a.n + .5) / (opportunities + 1)
                parts.append(a.add_prefix(f'local_{key}_{length}_{suffix}_'))
            if length == 1:
                continue
            z['w'] = z.similarity.pow(3)
            for rule in ['end_due', 'begin_due', 'speed_due']:
                z['in_window'] = z[rule].between(0, horizon - 1) * z.w
                z['central'] = np.exp(-.5 * ((z[rule] - horizon / 2) / 20) ** 2) * z.w
                groups = z.groupby('customer_id', sort=False)
                a = groups.agg(weight=('w', 'sum'), prob=('in_window', 'sum'), density=('central', 'sum'),
                               best=('central', 'max'), latest=(rule, 'last'), median=(rule, 'median'))
                a[['prob', 'density']] = a[['prob', 'density']].div(a.weight, axis=0)
                near = z.sort_values(['similarity', 'age'], ascending=[False, True]).drop_duplicates('customer_id')
                a['most_similar'] = near.set_index('customer_id')[rule]
                parts.append(a.add_prefix(f'align_{key}_{length}_{rule}_'))
    return pd.concat(parts, axis=1).astype('float32')
