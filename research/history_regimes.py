"""Detect a change in cadence, basket size or category mix using observed receipts."""
import numpy as np
import pandas as pd


def regime_features(receipts):
    events = receipts.sort_values(['customer_id', 'age', 'order_id'], ascending=[True, False, True])
    states = pd.get_dummies(events['mode'], dtype=float).to_numpy()
    grouped = events.groupby('customer_id', sort=False)
    ages = events.age.to_numpy()
    units = np.log1p(events.lines.to_numpy())
    sizes = grouped.size()
    offsets = sizes.cumsum().to_numpy()
    records = []
    for start, end in zip(offsets - sizes.to_numpy(), offsets):
        age = ages[start:end]
        row = {}
        records.append(row)
        if len(age) < 6:
            continue
        # The first purchase has no preceding gap. All rows below represent a
        # completed interval together with the basket at the end of that interval.
        gaps = -np.diff(age)
        observed = np.column_stack([np.log1p(gaps), units[start+1:end], states[start+1:end]])
        centered = observed - observed.mean(axis=0)
        centered[:, :2] /= np.maximum(observed[:, :2].std(axis=0), .4)
        n = len(gaps)
        splits = np.arange(2, n-1)
        sums = np.cumsum(centered, axis=0)[splits-1]
        # Between-segment sum of squares per dimension at each eligible split.
        gain = sums**2 * (n / (splits * (n-splits)))[:, None]
        for name, statistic in [('cadence', gain[:, 0]), ('basket', gain[:, 1:].sum(axis=1)),
                                ('joint', gain.sum(axis=1))]:
            best = np.argmax(statistic)
            split = splits[best]
            before, after = gaps[:split], gaps[split:]
            new, old = np.median(after), np.median(before)
            prefix = f'regime_{name}_'
            row.update({
                prefix+'gain': statistic[best],
                prefix+'fraction': (n-split)/n,
                prefix+'orders_after': n-split,
                prefix+'elapsed': age[split]-age[-1],
                prefix+'old_due': old-age[-1],
                prefix+'new_due': new-age[-1],
                prefix+'log_ratio': np.log1p(new)-np.log1p(old),
                prefix+'new_spread': np.median(np.abs(after-new)),
                prefix+'new_overdue': (age[-1]-13)/(new+1),
                prefix+'new_quarter': np.mean((after>=age[-1]) & (after<age[-1]+90)),
            })
    return pd.DataFrame.from_records(records, index=sizes.index).astype('float32')
