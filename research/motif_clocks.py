"""Recurrence and successor statistics for the customer's most recent category sequence."""

import numpy as np
import pandas as pd


def motif_clocks(receipts, category='mode', depth=4):
    b = receipts[['customer_id', 'order_id', 'age', category]].sort_values(
        ['customer_id', 'age', 'order_id'], ascending=[True, False, True]).copy()
    client = 'customer_id'
    g = b.groupby(client, sort=False)
    b['state'] = pd.Categorical(b[category]).codes.astype('int64') + 1
    width = int(b.state.max()) + 1
    b['next'] = g.state.shift(-1)
    b['wait'] = b.age - g.age.shift(-1)
    b['from_end'] = g.cumcount(ascending=False)
    b['seq'] = b.state
    steps = g.age.size()
    rec = g.age.min()
    personal = g.wait.median()
    b['unigram'] = b.state
    b['bigram'] = (g.state.shift().fillna(0).astype('int64') * width + b.state).where(g.cumcount() > 0)
    clocks = {}
    for prefix in ['unigram', 'bigram']:
        grp = b.groupby([client, prefix], sort=False)
        delta = -grp.age.diff()
        times = grp.age.agg(last='min', count='size')
        times['cycle'] = delta.groupby([b[client], b[prefix]]).median()
        clocks[prefix] = times
    outputs = []
    fallback = pd.DataFrame({'time': personal - rec, 'state_due': personal - rec, 'lag': 0., 'support': 0.}, index=rec.index)
    for size in range(1, depth + 1):
        if size > 1:
            b['seq'] = g.seq.shift().fillna(0).astype('int64') * width + b.state
        eligible = g.cumcount() >= size - 1
        recent = g.tail(1).set_index(client)
        hit = b[eligible & b.seq.eq(b[client].map(recent.seq))].copy()
        hg = hit.groupby(client)
        hit['interval'] = -hg.age.diff()
        hit['offset'] = hg.from_end.shift() - hit.from_end
        result = hg.agg(n=('age', 'size'), cycle=('interval', 'median'), mean_cycle=('interval', 'mean'),
            last_cycle=('interval', 'last'), order_period=('offset', 'median'), last_offset=('offset', 'last'))
        result['cycle_due'] = result.cycle - rec
        result['last_cycle_due'] = result.last_cycle - rec
        result['mean_due'] = result.mean_cycle - rec
        result['share'] = (result.n - 1) / (steps - size).clip(lower=1)
        past = hit[hit.next.notna()].copy()
        choices = past.groupby([client, 'next'], sort=False).agg(count=('next', 'size'), step=('wait', 'median'),
            last_age=('age', 'min'), next_min_wait=('wait', 'min'), next_max_wait=('wait', 'max')).reset_index()
        choices['total'] = choices.groupby(client)['count'].transform('sum')
        choices['p'] = choices['count'] / choices.total
        choices['h'] = -choices.p * np.log(choices.p)
        choice = choices.sort_values(['count', 'last_age', 'next'], ascending=[False, True, True]).drop_duplicates(client).set_index(client)
        result['support'] = past.groupby(client).size().reindex(result.index, fill_value=0)
        result['branches'] = choices.groupby(client).size().reindex(result.index, fill_value=0)
        result['confidence'] = choice.p
        result['entropy'] = choices.groupby(client).h.sum()
        result['same'] = choice.next.eq(recent.state.reindex(choice.index)).astype(float)
        result['step_due'] = choice.step - rec
        result['pooled_step_due'] = past.groupby(client).wait.median() - rec
        result['next_min_due'] = choice.next_min_wait - rec
        result['next_max_due'] = choice.next_max_wait - rec
        repeat = choices[choices.next.eq(choices[client].map(recent.state))].set_index(client)
        result['repeat_p'] = repeat.p.reindex(result.index).fillna(0)
        for label, state in [('unigram', choice.next), ('bigram', recent.state.reindex(choice.index) * width + choice.next)]:
            lookup = clocks[label].reindex(pd.MultiIndex.from_arrays([choice.index, state])).set_axis(choice.index)
            result[label + '_age'] = lookup['last']
            result[label + '_n'] = lookup['count']
            result[label + '_due'] = lookup.cycle - lookup['last']
            result[label + '_from_last'] = lookup.cycle - rec
            result[label + '_global_due'] = personal - lookup['last']
        # How predictable was the observed final state from its preceding context?
        prev_context = g.seq.shift().groupby(b[client]).last()
        prev_matches = b[eligible & b.seq.eq(b[client].map(prev_context)) & b.from_end.gt(1)].copy()
        prev_matches['correct'] = prev_matches.next.eq(prev_matches[client].map(recent.state))
        known = prev_matches.groupby(client).correct.agg(['sum', 'size'])
        result['past_likelihood'] = (known['sum'] + .5) / (known['size'] + 1)
        result['past_prefix_n'] = known['size']
        result['nonoverlap_n'] = past[past.from_end >= size].groupby(client).size().reindex(result.index, fill_value=0)
        result['gap_per_order'] = result.cycle / result.order_period
        result['gap_per_order_due'] = result.gap_per_order - rec
        update = result.index[result.support.gt(0)]
        fallback.loc[update, 'time'] = result.loc[update, 'step_due']
        fallback.loc[update, 'state_due'] = result.unigram_due.fillna(result.unigram_global_due).reindex(update)
        fallback.loc[update, 'lag'] = size
        fallback.loc[update, 'support'] = result.loc[update, 'support']
        outputs.append(result.add_prefix(f'motif_{category}_{size}_'))
    return pd.concat([*outputs, fallback.add_prefix('motif_' + category + '_backoff_')], axis=1)


def habitual_sequences(receipts, depth=4):
    """Select recurring category sequences from the whole history, including past ones."""
    b = receipts[['customer_id', 'order_id', 'age', 'mode']].copy()
    b = b.sort_values(['customer_id', 'age', 'order_id'], ascending=[True, False, True])
    client = 'customer_id'
    g = b.groupby(client, sort=False)
    b['state'] = pd.Categorical(b['mode']).codes.astype('int64') + 1
    width = int(b.state.max()) + 1
    b['seq'] = b.state
    b['gap'] = -g.age.diff()
    personal = g.gap.median()
    rec = g.age.min()
    parts = []
    for size in range(1, depth + 1):
        if size > 1:
            b['seq'] = g.seq.shift().fillna(0).astype('int64') * width + b.state
        eligible = g.cumcount() >= size - 1
        sub = b.loc[eligible].copy()
        h = sub.groupby([client, 'seq'], sort=False)
        sub['cycle'] = -h.age.diff()
        sub['begin'] = b.groupby(client).age.shift(size - 1)
        a = h.agg(n=('age', 'size'), age=('age', 'min'), first_age=('age', 'max'),
            begin=('begin', 'last'), period=('cycle', 'median'), latest_gap=('cycle', 'last')).reset_index()
        a['usual'] = a[client].map(personal)
        a['recency'] = a[client].map(rec)
        a['global_due'] = a.usual - a.age
        a['since_start'] = a.usual - a.begin
        a['specific_due'] = a.period - a.age
        a['anchored_now'] = a.period - a.recency
        a['specific_center'] = abs(a.specific_due - 45)
        a['global_center'] = abs(a.global_due - 45)
        a['latest'] = a.age.eq(a.recency)
        max_count = a.groupby(client).n.transform('max')
        totals = a.groupby(client).n.transform('sum')
        a['share'] = a.n / totals
        a['wins'] = a.n.eq(max_count)
        ag = a.groupby(client)
        stats = ag.agg(vocabulary=('seq', 'size'), top_share=('share', 'max'), tied=('wins', 'sum'))
        for name, ordering, ascending in [('recent', ['age', 'seq'],[True, True]),
            ('old', ['first_age', 'seq'], [False, True]), ('lex', ['seq'], [True])]:
            chosen = a[a.wins].sort_values(ordering, ascending=ascending).drop_duplicates(client).set_index(client)
            for col in ['n', 'age', 'global_due', 'specific_due', 'anchored_now', 'since_start', 'latest']:
                stats[name + '_' + col] = chosen[col].astype(float)
        for count in [1, 2, 3]:
            subset = a[a.n >= count]
            gg = subset.groupby(client)
            top = subset.sort_values(['age', 'seq']).drop_duplicates(client).set_index(client)
            for col in ['age', 'global_due', 'specific_due', 'n']:
                stats[f'last{count}_' + col] = top[col]
            stats[f'last{count}_best_global'] = gg.global_center.min()
            stats[f'last{count}_best_specific'] = gg.specific_center.min()
        parts.append(stats.add_prefix(f'habit_{size}_'))
    return pd.concat(parts, axis=1)


def alternate_histories(orders, customers):
    from research.sequence_rules import baskets
    b = baskets(orders, customers).sort_values(['customer_id', 'age', 'order_id'], ascending=[True, False, True])
    grouped = b.groupby('customer_id', sort=False)
    b['switch'] = b['mode'].ne(grouped['mode'].shift())
    b['run'] = b.switch.groupby(b.customer_id).cumsum()
    b['session'] = ((-grouped.age.diff()).gt(7) | grouped.age.diff().isna()).groupby(b.customer_id).cumsum()
    b['department'] = b.gender_product + '|' + b['mode']
    fields = ['support', 'share', 'past_likelihood', 'pooled_step_due', 'cycle_due', 'bigram_global_due',
              'unigram_global_due', 'branches', 'confidence', 'same']
    parts = []
    for name, events, key in [
        ('run_first', b.drop_duplicates(['customer_id', 'run'], keep='first'), 'mode'),
        ('run_last', b.drop_duplicates(['customer_id', 'run'], keep='last'), 'mode'),
        ('session_last', b.drop_duplicates(['customer_id', 'session'], keep='last'), 'mode'),
        ('day_last', b.drop_duplicates(['customer_id', 'age'], keep='last'), 'mode'),
        ('gender', b, 'gender_product'), ('subcat', b, 'category2'), ('department', b, 'department')
    ]:
        stats = motif_clocks(events, category=key, depth=3)
        cols = [f'motif_{key}_{k}_{c}' for k in range(1, 4) for c in fields]
        parts.append(stats[cols].add_prefix(name + '_'))
    return pd.concat(parts, axis=1).reindex(customers.customer_id).astype('float32')
