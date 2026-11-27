"""Weight earlier purchase sequences by similarity of price and promotions."""
import numpy as np
import pandas as pd


def price_context(receipts, items):
    sums = items.groupby(['customer_id','order_id'],sort=False).agg(
        total=('paid_price','sum'), discount=('discount_percent','mean'), loyalty=('loyalty_percent','mean'))
    b = receipts.join(sums, on=['customer_id','order_id'],validate='many_to_one')
    b = b.sort_values(['customer_id','age','order_id'],ascending=[True,False,True]).copy()
    b['price'] = np.log1p((b.total/b.lines).clip(lower=0))
    people = b.groupby('customer_id',sort=False)
    b['wait'] = b.age-people.age.shift(-1)
    b['next_age'] = people.age.shift(-1)
    b['state'] = pd.Categorical(b['mode']).codes.astype('int64')+1
    b['sequence'] = b.state
    radix = b.state.max()+1
    position = people.cumcount()
    recency = people.age.min()
    period = people.wait.median()
    fields = ['price','discount','loyalty']
    scales = [1.,20.,8.]
    distance = pd.Series(0.,index=b.index)
    frames = []
    for depth in range(1,4):
        if depth>1:
            b['sequence'] = people.sequence.shift().fillna(0).astype('int64')*radix+b.state
        for feature,scale in zip(fields,scales):
            lag = people[feature].shift(depth-1)
            # The current prefix and a historical prefix have the same depth.
            latest = lag.groupby(b.customer_id).transform('last')
            distance += ((lag-latest)/scale).abs().fillna(0)
        candidate = b.loc[position.ge(depth-1)&b.wait.notna()&
                          b.sequence.eq(people.sequence.transform('last'))].copy()
        candidate['weight'] = np.exp(-distance.loc[candidate.index]/depth)
        candidate['due'] = candidate.wait-candidate.customer_id.map(recency)
        candidate['cycle_due'] = candidate.customer_id.map(period)-candidate.next_age
        candidate['density'] = np.exp(-.5*((candidate.due-45)/22)**2)
        candidate['in_quarter'] = candidate.due.between(0,89).astype(float)
        by_person = candidate.groupby('customer_id',sort=False)
        closest = candidate.sort_values(['weight','age'],ascending=[False,True]).drop_duplicates('customer_id')
        out = closest.set_index('customer_id')[['weight','due','cycle_due','next_age']].copy()
        total = by_person.weight.sum()
        out['mean_weight'] = by_person.weight.mean()
        for field in ['due','density','in_quarter','cycle_due']:
            out['weighted_'+field] = (candidate.weight*candidate[field]).groupby(candidate.customer_id).sum()/total
        frames.append(out.add_prefix(f'price_prefix_{depth}_'))
    return pd.concat(frames, axis=1).astype('float32')
