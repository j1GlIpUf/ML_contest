"""Forecast recurring category positions and customer-specific seasonal phases."""
import numpy as np
import pandas as pd


def cycle_positions(b):
    b = b.sort_values(['customer_id', 'age', 'order_id'], ascending=[True, False, True]).copy()
    grouped = b.groupby('customer_id', sort=False)
    part = []
    recency = grouped.age.last()
    newest = grouped.cumcount(ascending=False)
    for key in ['mode', 'gender_product', 'category2']:
        prior = b.groupby(['customer_id', key]).size().rename('n').reset_index()
        chance = ((prior.n / prior.groupby('customer_id').n.transform('sum')) ** 2).groupby(prior.customer_id).sum()
        choices = []
        for lag in range(1, 9):
            previous = grouped[key].shift(lag)
            sub = b[previous.notna()].copy()
            sub['agree'] = sub[key].eq(previous[sub.index])
            sub['period'] = (grouped.age.shift(lag) - b.age).loc[sub.index]
            # Matching the last few positions is distinct from agreement over
            # earlier cycles; a short suffix alone supplies weak evidence.
            sub['mismatch_at'] = newest.loc[sub.index].where(~sub.agree)
            sub['terminal'] = newest.loc[sub.index].lt(lag)
            g = sub.groupby('customer_id', sort=False)
            z = g.agg(n=('age', 'size'), hits=('agree', 'sum'), period=('period', 'median'),
                      last_period=('period', 'last'), break_at=('mismatch_at', 'min'), spread=('period', 'std'))
            z['tail_run'] = z.break_at.fillna(z.n)
            prefix = sub[~sub.terminal].groupby('customer_id').agree.agg(['sum','count'])
            z['early_rate'] = (prefix['sum'] + 2 * chance) / (prefix['count'] + 2)
            z['end_rate'] = sub[sub.terminal].groupby('customer_id').agree.mean()
            z['evidence'] = (z.hits - z.n * chance) / np.sqrt(2 + z.n * chance * (1 - chance))
            ref_age = b[newest.eq(lag - 1)].set_index('customer_id').age
            z['due'] = z.period - ref_age
            z['last_due'] = z.last_period - ref_age
            z['center'] = (z.due - 45).abs()
            z['mean_step_due'] = z.period / lag - recency
            if key == 'mode':
                part.append(z[['tail_run', 'evidence', 'early_rate', 'end_rate', 'due', 'last_due', 'center',
                               'mean_step_due']].add_prefix(f'lag_{key}_{lag}_'))
            choices.append(z.assign(lag=lag).reset_index())
        options = pd.concat(choices, ignore_index=True)
        for name, sorting in [('history', ['evidence', 'tail_run']), ('suffix', ['tail_run','evidence'])]:
            best = options.sort_values(sorting + ['lag'], ascending=[False,False,True]).drop_duplicates('customer_id')
            cols = ['lag','evidence','tail_run','due','last_due','center','early_rate','end_rate','mean_step_due','spread']
            part.append(best.set_index('customer_id')[cols].add_prefix(f'lag_{key}_{name}_'))
    return pd.concat(part, axis=1).astype('float32')


def spectral_days(b):
    # Fit Fourier components to historical purchase days only. Measure stability
    # using the older and newer halves of a history separately.
    day = b[['customer_id','age']].drop_duplicates().sort_values(['customer_id','age'])
    ids = day.customer_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    counts = np.diff(np.r_[starts,len(day)])
    ages = day.age.to_numpy()
    origin = np.repeat(starts, counts)
    newer = np.arange(len(day))-origin < np.repeat(np.ceil(counts/2),counts)
    out = {}
    for low, high, step in [(14, 89, 2), (90, 210, 3), (211, 390, 3)]:
        best = np.full(len(starts),-np.inf)
        winner = np.full((len(starts),6),np.nan)
        for period in range(low, high + 1, step):
            ph = np.exp(2j * np.pi * ((-ages) % period) / period)
            left = np.add.reduceat(ph * newer,starts)
            right = np.add.reduceat(ph * ~newer,starts)
            total = left + right
            coherence = np.real(left * right.conjugate()) / np.maximum(1, np.ceil(counts/2)*np.floor(counts/2))
            span = ages[starts+counts-1] - ages[starts]
            evidence = coherence * np.sqrt(counts) * np.minimum(1, span / (2 * period))
            select = (evidence > best) & (counts >= 4) & (span >= period)
            next_day = np.mod(np.angle(total)/(2*np.pi),1) * period
            previous_phase = np.mod(np.angle(right)/(2*np.pi),1) * period
            distance = np.abs((45-next_day+period/2) % period - period/2)
            vals = np.column_stack([np.full(len(starts),period),evidence, next_day, distance,
                                    previous_phase, np.abs(total)/counts])
            winner[select] = vals[select]
            best[select] = evidence[select]
        for i,name in enumerate(['period','strength','next','center','older_next','concentration']):
            out[f'spectral_{low}_{name}'] = winner[:,i]
    return pd.DataFrame(out,index=pd.Index(ids[starts],name='customer_id')).astype('float32')
