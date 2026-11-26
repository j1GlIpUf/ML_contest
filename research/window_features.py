"""Renewal probabilities and basket features from the observable order history."""

import numpy as np
import pandas as pd
from scipy.special import log_ndtr


def log_survival(days, mu, sigma):
    # Discretise a lognormal waiting time to whole days, with a minimum of one day.
    d = np.asarray(days)
    z = (mu - np.log(np.maximum(d, 1) + .5)) / sigma
    return np.where(d < 1, 0.0, log_ndtr(z))


def renewal_window(age, mu, sigma, horizon, lag=14):
    """Return probabilities given no further observed event through day -lag.

    Enumerate the possible first events in the blind days. After such an event,
    subsequent intervals are independent draws from the same discrete lognormal.
    """
    age, mu, sigma = np.broadcast_arrays(age, mu, sigma)
    observed = log_survival(age - lag, mu, sigma)
    start = log_survival(age - 1, mu, sigma)
    end = log_survival(age + horizon - 1, mu, sigma)
    first = np.exp(start - observed) * -np.expm1(end - start)
    blind = -np.expm1(start - observed)
    repeated = first.copy()
    # hit[i] is the chance of a purchase in [0, horizon) after an event on day -i.
    hit = []
    gap_mass = []
    for days in range(1, lag):
        prev = log_survival(days - 1, mu, sigma)
        curr = log_survival(days, mu, sigma)
        gap_mass.append(np.exp(prev) * -np.expm1(curr - prev))
        to_end = log_survival(days + horizon - 1, mu, sigma)
        chance = np.exp(prev) * -np.expm1(to_end - prev)
        for gap in range(1, days):
            chance += gap_mass[gap - 1] * hit[days - gap - 1]
        hit.append(chance)
        lower = log_survival(age - days - 1, mu, sigma)
        upper = log_survival(age - days, mu, sigma)
        repeated += np.exp(lower - observed) * -np.expm1(upper - lower) * chance
    return np.stack([first, blind, np.clip(repeated, 0, 1)], axis=1)


def renewal_by_key(h, key, horizon):
    group = ['customer_id', key]
    visits = h[group + ['age']].drop_duplicates().sort_values('age', ascending=False)
    g = visits.groupby(group, observed=True, sort=False)
    visits['log_gap'] = np.log(-g.age.diff())
    visits['log2'] = visits.log_gap.pow(2)
    personal = g.agg(age=('age', 'min'), n=('log_gap', 'count'),
                     s=('log_gap', 'sum'), s2=('log2', 'sum')).reset_index()
    # Only input rows of the current as-of cohort contribute, including customers
    # being scored. No labels or later scoring cohorts enter the population prior.
    pop = personal.groupby(key, observed=True)[['n', 's', 's2']].sum()
    for c in ['n', 's', 's2']:
        personal['pop_' + c] = personal[key].map(pop[c]).astype(float) - personal[c]
    n = personal.pop_n
    prior_mu = (personal.pop_s + 20 * visits.log_gap.mean()) / (n + 20)
    prior_m2 = (personal.pop_s2 + 20 * visits.log2.mean()) / (n + 20)
    count = personal.n + 2
    mu = (personal.s + 2 * prior_mu) / count
    sigma = np.sqrt(np.maximum((personal.s2 + 2 * prior_m2) / count - mu.pow(2), .10))
    personal[['first', 'blind', 'renewal']] = renewal_window(
        personal.age.to_numpy(), mu.to_numpy(), sigma.to_numpy(), horizon)
    personal['intensity'] = horizon * np.exp(-mu - .5 * sigma.pow(2))
    personal['period_due'] = np.exp(mu) - personal.age
    personal['population_due'] = np.exp(prior_mu) - personal.age
    personal['pop_repeat_rate'] = personal[key].map(
        personal.groupby(key, observed=True).n.apply(lambda a: (a.gt(0).sum() + 1) / (len(a) + 2))
    ).astype(float)
    personal['log_no_purchase'] = np.log1p(-personal.renewal.clip(upper=1 - 1e-12))
    by_customer = personal.groupby('customer_id', sort=False)
    result = by_customer.agg(strongest=('renewal', 'max'), blind=('blind', 'max'),
        first=('first', 'max'), intensity=('intensity', 'sum'), processes=(key, 'size'))
    result['parallel'] = -np.expm1(by_customer.log_no_purchase.sum())
    latest = personal.sort_values(['age', 'n', key], ascending=[True, False, True])
    latest = latest.drop_duplicates('customer_id').set_index('customer_id')
    result = result.join(latest[['period_due', 'population_due', 'pop_repeat_rate', 'renewal']])
    return result.add_prefix('risk_' + key + '_')


def completed_sessions(h):
    days = h.groupby(['customer_id', 'age'], sort=False).agg(
        orders=('order_id', 'nunique'), value=('paid_price', 'sum')).reset_index()
    days = days.sort_values(['customer_id', 'age'], ascending=[True, False])
    g = days.groupby('customer_id')
    days['gap'] = -g.age.diff()
    ordered = days.dropna(subset=['gap']).sort_values(['customer_id', 'gap'])
    ordered['log_gap'] = np.log(ordered.gap)
    ordered['separation'] = ordered.groupby('customer_id').log_gap.diff()
    jumps = ordered.dropna(subset=['separation']).sort_values('separation', kind='stable')
    jumps = jumps.drop_duplicates('customer_id', keep='last').set_index('customer_id')
    boundary = np.exp(jumps.log_gap - jumps.separation / 2).reindex(g.size().index)
    boundary = boundary.fillna(g.gap.median())
    days['new'] = days.gap.isna() | days.gap.gt(days.customer_id.map(boundary))
    days['session'] = g.new.cumsum()
    days['intra'] = days.gap.where(~days.new)
    session_keys = ['customer_id', 'session']
    labels = days[['customer_id', 'age', 'session']]
    categories = h[['customer_id', 'age', 'category']].drop_duplicates().merge(
        labels, on=['customer_id', 'age'], validate='many_to_one')
    sessions = days.groupby(session_keys).agg(begin=('age', 'max'), end=('age', 'min'),
        size=('age', 'size'), receipts=('orders', 'sum'), value=('value', 'sum'))
    sessions['variety'] = categories.groupby(session_keys).category.nunique()
    sg = sessions.groupby(level=0)
    sessions['idle'] = sg.end.shift() - sessions.begin
    sessions['period'] = -sg.begin.diff()
    closed = sg.cumcount(ascending=False) > 0
    last = sessions.loc[~closed].droplevel(1)
    old = sessions.loc[closed].copy()
    old['reached'] = old['size'].ge(last['size'].reindex(old.index.get_level_values(0)).values)
    old['grew'] = old['size'].gt(last['size'].reindex(old.index.get_level_values(0)).values)
    norm = old.groupby(level=0).agg(usual_days=('size', 'median'), usual_cats=('variety', 'median'),
        idle=('idle', 'median'), period=('period', 'median'), grew=('grew', 'sum'), reached=('reached', 'sum'))
    result = last[['size', 'receipts', 'end', 'begin', 'variety']].join(norm)
    closed_days = days[days.session.lt(g.session.transform('max'))]
    result['intra'] = closed_days.groupby('customer_id').intra.median()
    result['threshold'] = boundary
    result['fraction_open'] = (result.grew + .5) / (result.reached + 1)
    result['fraction_open'] *= (result.end - 14).le(boundary)
    result['intra_due'] = result.intra - result.end
    result['idle_due'] = result.idle - result.end
    result['start_due'] = result.period - result.begin
    result['mixture_due'] = result.fraction_open * result.intra_due + (1 - result.fraction_open) * result.idle_due
    return result.add_prefix('closed_')


def basket_and_market(h):
    ids = ['customer_id', 'order_id']
    counts = h.groupby(ids + ['category2'], sort=False).size().groupby(level=ids).max()
    receipts = h.groupby(ids, sort=False).agg(age=('age', 'first'), items=('product_id', 'size'),
        low=('paid_price', 'min'), high=('paid_price', 'max'), loyalty=('loyalty_percent', 'mean'))
    receipts['within_ratio'] = np.log1p(receipts.high) - np.log1p(receipts.low)
    receipts['same_subcat'] = counts
    receipts['same_share'] = counts / receipts['items']
    receipts = receipts.reset_index().sort_values(['age', 'order_id'], ascending=[False, True])
    g = receipts.groupby('customer_id')
    receipts['loyalty_change'] = g.loyalty.diff()
    out = [g[['within_ratio', 'same_subcat', 'same_share', 'loyalty_change']].agg(['last', 'mean'])]
    out[0].columns = ['basket_' + '_'.join(c) for c in out[0].columns]
    for window in [30, 90, 180, 365]:
        recent = h[h.age.between(14, 13 + window)]
        agg = recent.groupby('customer_id').agg(orders=('order_id', 'nunique'), items=('product_id', 'size'),
            spend=('paid_price', 'sum'), max_loyalty=('loyalty_percent', 'max')).reindex(g.size().index)
        agg['loyalty_from_peak'] = g.loyalty.last() - agg.pop('max_loyalty')
        agg[['orders', 'items', 'spend']] = agg[['orders', 'items', 'spend']].fillna(0)
        out.append(agg.add_prefix('observed_' + str(window) + '_'))
    # Use the same cohort throughout both windows for identical customer exposure.
    latest = h[h.age.eq(h.groupby('customer_id').age.transform('min'))]
    for key in ['brand', 'category2']:
        visits = h[['age', 'order_id', key]].drop_duplicates()
        recent = visits[visits.age.between(14, 43)].groupby(key).size()
        old = visits[visits.age.between(44, 133)].groupby(key).size()
        ratio = (recent.reindex(old.index.union(recent.index), fill_value=0) + 3) / (old.reindex(old.index.union(recent.index), fill_value=0) / 3 + 3)
        for suffix, frame in [('all', h), ('last', latest)]:
            trend = frame[key].map(ratio).groupby(frame.customer_id).agg(['mean', 'max'])
            out.append(trend.add_prefix('demand_' + key + '_' + suffix + '_'))
    return pd.concat(out, axis=1)


def fine_transitions(h, key):
    # One state per receipt. Ties are broken reproducibly using value and then name.
    ids = ['customer_id', 'order_id']
    choices = h.groupby(ids + [key], sort=False).agg(lines=('age', 'size'), amount=('paid_price', 'sum'), age=('age', 'first')).reset_index()
    b = choices.sort_values(['lines', 'amount', key], ascending=[False, False, True]).drop_duplicates(ids)
    b = b.sort_values(['customer_id', 'age', 'order_id'], ascending=[True, False, True])
    g = b.groupby('customer_id')
    b['from'] = g[key].shift()
    b['older'] = g[key].shift(2)
    b['gap'] = -g.age.diff()
    b['return'] = -b.groupby(['customer_id', key]).age.diff()
    a = b.groupby(['customer_id', key]).agg(n=('age', 'size'), age=('age', 'min'), period=('return', 'median'))
    last = g[key].last()
    previous = g['from'].last()
    mask = b['from'].eq(b.customer_id.map(last))
    out = []
    for name, selected in [('one', mask), ('two', mask & b.older.eq(b.customer_id.map(previous)))]:
        matches = b[selected].groupby(['customer_id', key]).agg(support=('age', 'size'), wait=('gap', 'median'))
        d = a.join(matches)
        d['support'] = d.support.fillna(0)
        d['weight'] = d.support + d.n / d.n.groupby(level=0).transform('sum')
        d['weight'] /= d.weight.groupby(level=0).transform('sum')
        clients = d.index.get_level_values(0)
        personal = g.gap.median().reindex(clients).values
        due = d.period.fillna(pd.Series(personal, index=d.index)) - d.age
        step = d.wait - g.age.min().reindex(clients).values
        d['expected'] = due * d.weight
        d['window'] = np.exp(-.5 * ((due - 45) / 32) ** 2) * d.weight
        d['step'] = np.exp(-.5 * ((step.fillna(due) - 45) / 32) ** 2) * d.weight
        stats = d.groupby(level=0).agg(expected=('expected', 'sum'), window=('window', 'sum'),
            strongest=('window', 'max'), step=('step', 'sum'), matches=('support', 'sum'))
        out.append(stats.add_prefix('fine_' + key + '_' + name + '_'))
    return pd.concat(out, axis=1)


def build_features(orders, cohort):
    """Produce one row per customer, in input order. Cohort dates are never pooled."""
    blocks = []
    for cutoff, customers in cohort.groupby('scoring_date', sort=True):
        h = orders.loc[orders.customer_id.isin(customers.customer_id) & (orders.order_date <= cutoff - pd.Timedelta(days=14))].copy()
        h['age'] = (cutoff - h.order_date).dt.days
        h['account'] = 'all'
        h['department'] = h.gender_product + '|' + h.category
        horizon = (cutoff + pd.DateOffset(months=3) - cutoff).days
        parts = [renewal_by_key(h, key, horizon) for key in
                 ['account', 'category', 'gender_product', 'department', 'category2', 'brand']]
        parts.extend(fine_transitions(h, key) for key in ['gender_product', 'department', 'category2'])
        parts.extend([completed_sessions(h), basket_and_market(h)])
        blocks.append(pd.concat(parts, axis=1).reindex(customers.customer_id))
    return pd.concat(blocks).reindex(cohort.customer_id).astype('float32')
