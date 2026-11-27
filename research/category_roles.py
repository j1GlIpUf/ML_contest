"""Align customers' category histories by recency of use, then build transition clocks."""
import numpy as np
import pandas as pd


def role_clocks(receipts):
    b = receipts.sort_values(['customer_id', 'age', 'order_id'], ascending=[True, False, True]).copy()
    client = b.groupby('customer_id', sort=False)
    b['position'] = client.cumcount()
    # Role zero is the most recently used category, role one the preceding
    # distinct category. Different absolute categories can then share clocks.
    last_use = b.drop_duplicates(['customer_id', 'mode'], keep='last').copy()
    last_use['role'] = last_use.groupby('customer_id').cumcount(ascending=False).clip(upper=3)
    b = b.merge(last_use[['customer_id', 'mode', 'role']], on=['customer_id', 'mode'],
                how='left', validate='many_to_one', sort=False)
    client = b.groupby('customer_id', sort=False)
    b['wait'] = -client.age.diff()
    median = client.wait.median()
    year = b.loc[b.age.le(378)].copy()
    year['wait'] = -year.groupby('customer_id').age.diff()
    year_mean = year.groupby('customer_id').wait.mean()
    b['prior_role'] = client.role.shift()
    b['prior_age'] = client.age.shift()
    b['next_age'] = client.age.shift(-1)
    b['pair'] = 4*b.prior_role+b.role
    b['is_followed'] = b.next_age.notna().astype(int)
    results = []
    for name, key in [('category', 'role'), ('transition', 'pair')]:
        by_key = b.groupby(['customer_id', key], sort=False)
        b['return'] = -by_key.age.diff()
        b['steps'] = by_key.position.diff()
        table = by_key.agg(count=('age','size'), age=('age','min'), same_period=('return','median'),
                           position=('position','last'), lag=('steps','median'), wait=('wait','median'),
                           prior=('prior_age','last'), subsequent=('next_age','min'), followed=('is_followed','sum'))
        cid = table.index.get_level_values('customer_id')
        typical = median.reindex(cid).to_numpy()
        annual = year_mean.reindex(cid).to_numpy()
        table['from_end'] = client.position.max().reindex(cid).to_numpy() - table.position
        table['due'] = typical - table.age
        table['due_year'] = annual - table.age
        table['cycle_due'] = table.same_period-table.age
        table['projected_steps'] = typical*table.lag-table.age
        table['arrival_due'] = table.wait-client.age.min().reindex(cid).to_numpy()
        table['prior_due'] = typical-table.prior
        table['subsequent_due'] = typical-table.subsequent
        table['share'] = table['count']/client.age.size().reindex(cid).to_numpy()
        fields = ['count','from_end','due','due_year','cycle_due','projected_steps','arrival_due','share']
        if key=='pair':
            fields += ['prior_due','subsequent_due']
        wide = table[fields].unstack(key)
        expected = range(4 if key=='role' else 16)
        wide = wide.reindex(columns=pd.MultiIndex.from_product([fields,expected]))
        wide.columns = [f'{name}_{int(role)}_{field}' for field,role in wide.columns]
        results.append(wide)
    return pd.concat(results,axis=1).astype('float32')
