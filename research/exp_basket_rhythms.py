"""Test separate purchase rhythms for demographic groups and learned basket types."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import MiniBatchNMF
from sklearn.preprocessing import OneHotEncoder, normalize
from threadpoolctl import threadpool_limits

from research.exp_aux_intensity import published_scores, write_report
from research.rerank import compare
from research.tail_models import booster

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'research/results/basket_rhythms.csv'
COMPONENTS = 16
MIN_SHARE = .25


def read_baskets(path, cohorts):
    lines = pd.read_parquet(path / 'orders.parquet', columns=[
        'customer_id', 'order_id', 'order_date', 'product_id', 'paid_price'])
    products = pd.read_parquet(path / 'products.parquet', columns=['product_id', 'gender_product', 'category2'])
    lines['order_date'] = pd.to_datetime(lines.order_date)
    lines = lines.merge(cohorts[['scoring_date']], on='customer_id', validate='many_to_one')
    lines = lines[lines.order_date.le(lines.scoring_date - pd.Timedelta(days=14))]
    lines = lines.merge(products, on='product_id', how='left', validate='many_to_one')
    lines[['gender_product', 'category2']] = lines[['gender_product', 'category2']].fillna('unknown')
    keys = ['customer_id', 'order_date', 'order_id']
    baskets = lines[keys + ['scoring_date']].drop_duplicates().sort_values(keys, kind='stable')
    if baskets.duplicated(['customer_id', 'order_id']).any():
        raise ValueError('Order date is inconsistent within a basket.')
    baskets = baskets.reset_index(drop=True).rename_axis('basket')
    lines = lines.merge(baskets[keys].reset_index(), on=keys, validate='many_to_one')
    return lines, baskets


def price_bands(lines, training):
    fit = lines.loc[training, ['category2', 'paid_price']].dropna()
    bounds = fit.groupby('category2').paid_price.quantile([1 / 3, 2 / 3]).unstack()
    low = lines.category2.map(bounds.iloc[:, 0]).fillna(fit.paid_price.quantile(1 / 3))
    high = lines.category2.map(bounds.iloc[:, 1]).fillna(fit.paid_price.quantile(2 / 3))
    return np.select([lines.paid_price.le(low), lines.paid_price.le(high), lines.paid_price.gt(high)],
                     ['low', 'middle', 'high'], default='unknown')


def composition(lines, size, columns, training):
    encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float32).fit(lines.loc[training, columns])
    encoded = encoder.transform(lines[columns]).tocoo()
    baskets = sparse.csr_matrix((encoded.data, (lines.basket.to_numpy()[encoded.row], encoded.col)),
                                shape=(size, encoded.shape[1]), dtype=np.float32)
    return normalize(baskets, norm='l1', copy=False)


def membership(shares, max_types):
    ordered = np.argsort(-shares, axis=1, kind='stable')[:, :max_types]
    mass = np.take_along_axis(shares, ordered, axis=1)
    keep = mass >= MIN_SHARE
    keep[:, 0] = mass[:, 0] > 0
    row, rank = np.where(keep)
    return pd.DataFrame({'basket': row, 'kind': ordered[row, rank]})


def basket_memberships(lines, baskets, fit_ids):
    fit_baskets = baskets.customer_id.isin(fit_ids).to_numpy()
    train = lines.customer_id.isin(fit_ids)
    gender = composition(lines, len(baskets), ['gender_product'], train).toarray()
    lines = lines.assign(band=price_bands(lines, train))
    lines['token'] = lines.gender_product.astype(str) + '|' + lines.category2.astype(str) + '|' + lines.band
    joint = composition(lines, len(baskets), ['token'], train)
    # Separate marginal views keep related categories and price bands from
    # becoming orthogonal merely because their joint tokens differ.
    marginal = composition(lines, len(baskets), ['gender_product', 'category2', 'band'], train)
    design = sparse.hstack([.5 * joint, .5 * marginal], format='csr')
    learner = MiniBatchNMF(n_components=COMPONENTS, init='nndsvda', batch_size=2048,
                          max_iter=50, random_state=26)
    with threadpool_limits(limits=4):
        learner.fit(design[fit_baskets])
        latent = learner.transform(design)
    # Adjust for component scale before comparing memberships in a basket.
    latent *= learner.components_.sum(axis=1)
    topics = normalize(latent, norm='l1', copy=False)
    return {'gender': (membership(gender, COMPONENTS), gender.shape[1]),
            'basket_types': (membership(topics, 2), COMPONENTS)}


def separate_rhythms(eligible_days, summary, ids):
    enough = pd.Series(False, index=ids)
    distinct = enough.copy()
    for customer, visits in eligible_days.groupby('customer_id', sort=False):
        days = {kind: set(rows.order_date) for kind, rows in visits.groupby('kind', sort=False)}
        intervals = summary.loc[customer, 'gap']
        for first, first_days in days.items():
            for second, second_days in days.items():
                if second <= first or min(len(first_days - second_days), len(second_days - first_days)) < 2:
                    continue
                enough.loc[customer] = True
                low, high = sorted((intervals.loc[first], intervals.loc[second]))
                distinct.loc[customer] |= high - low >= max(14, .25 * low)
    return enough, distinct


def rhythm_features(baskets, members, n_kinds, ids, name):
    order_summary = members.join(baskets, on='basket')
    keys = ['customer_id', 'kind']
    by_kind = order_summary.groupby(keys, sort=False)
    stats = by_kind.agg(n_orders=('basket', 'size'), latest=('order_date', 'max'),
                       last_basket=('basket', 'max'), cutoff=('scoring_date', 'first'))
    stats['recency'] = (stats.pop('cutoff') - stats.pop('latest')).dt.days
    days = order_summary[keys + ['order_date']].drop_duplicates().sort_values(keys + ['order_date'])
    days['gap'] = days.groupby(keys, sort=False).order_date.diff().dt.days
    by_day = days.groupby(keys, sort=False)
    gap = by_day.gap.median().rename('gap')
    days['deviation'] = (days.gap - days.set_index(keys).index.map(gap)).abs()
    dispersion = days.groupby(keys, sort=False).deviation.median() / gap
    stats = stats.join(gap).join(dispersion.rename('dispersion'))
    stats['due'] = stats.gap - stats.recency
    stable = stats[by_day.gap.count().reindex(stats.index).ge(2) & stats.dispersion.le(.75)]
    stable_by = stable.groupby(level='customer_id', sort=False)
    count = stable_by.size().reindex(ids, fill_value=0)
    due = stable_by.due.agg(['min', 'max'])
    regular = stable.reset_index().sort_values(['dispersion', 'n_orders', 'kind'],
                    ascending=[True, False, True], kind='stable').drop_duplicates('customer_id').set_index('customer_id')
    last_basket = baskets.reset_index().groupby('customer_id').basket.max()
    last_match = regular.last_basket.eq(last_basket.reindex(regular.index)).astype(float)
    frame = pd.DataFrame({'stable_types': count,
                          'near_due': stable.due.abs().le(14).groupby(level='customer_id').sum().reindex(ids, fill_value=0),
                          'due_next_90d': stable.due.between(0, 90, inclusive='left').groupby(level='customer_id').sum().reindex(ids, fill_value=0),
                          'due_spread': (due['max'] - due['min']).where(count.ge(2)),
                          'last_is_most_regular': last_match}, index=ids)
    eligible = stable.index[stable.index.get_level_values('customer_id').map(count) >= 2]
    eligible_days = days[days.set_index(keys).index.isin(eligible)]
    enough, distinct = separate_rhythms(eligible_days, stable, ids)
    frame['two_supported'] = enough
    frame['two_distinct'] = distinct
    for field in ('n_orders', 'recency', 'gap', 'due', 'dispersion'):
        wide = stats[field].unstack('kind').reindex(index=ids, columns=range(n_kinds))
        if field == 'n_orders':
            wide = wide.fillna(0)
        frame = frame.join(wide.add_prefix(f'{field}_'))
    return frame.add_prefix(f'{name}_').astype('float32'), enough, distinct


def control_selection(features, labels, dates, quarters, fit):
    rank = quarters.loc[fit].groupby(dates.loc[fit]).rank(pct=True, method='first')
    pool = rank.index[rank.gt(.6)]
    model = booster('lightgbm').set_params(n_jobs=4).fit(features.loc[pool], labels.loc[pool])
    gain = pd.Series(model.booster_.feature_importance(importance_type='gain'), index=features.columns)
    return gain.sort_values(ascending=False, kind='stable').head(160).index


def report_metrics(y, p, ref, orders):
    medium = orders.between(7, 15)
    full = compare(y, p, ref, orders)
    group = compare(y[medium], p[medium], ref[medium], orders[medium])
    top = p.sort_values(ascending=False, kind='stable').index[:5000]
    medium_top = top[medium.loc[top]]
    return dict(ap20=full['ap20'], buyers_top5000=full['buyers'], medium_ap20=group['ap20'],
                medium_customers=int(medium.sum()), medium_buyers_top5000=int(y.loc[medium_top].sum()),
                medium_nonbuyers_top5000=full['false_7_15'])


def run(data, cache):
    cohorts = pd.read_csv(data / 'train.csv', parse_dates=['scoring_date']).set_index('customer_id', verify_integrity=True)
    dates, labels = cohorts.scoring_date, cohorts.target
    lines, baskets = read_baskets(data, cohorts)
    features = pd.concat([pd.read_parquet(cache / f'features_{m}.parquet') for m in (1, 2, 3)],
                         verify_integrity=True).loc[cohorts.index]
    quarters = published_scores(ROOT / 'research/oof').loc[cohorts.index, 'quarters']
    submission = pd.read_csv(ROOT / 'submission_tail.csv').set_index('customer_id').score
    april = pd.read_parquet(ROOT / 'research/oof/tail_predictions_4.parquet').lightgbm_plain_half
    if len(submission) != len(april) or not np.allclose(submission.loc[april.index], april, atol=1e-14, rtol=0):
        raise ValueError('The archived control does not reproduce submission_tail.csv.')
    report = []
    for month in (2, 3):
        fit = dates.index[dates.dt.month.lt(month)]
        query = dates.index[dates.dt.month.eq(month)]
        ids = fit.append(query)
        # Restrict the entire transformation to participating cohorts.
        active = baskets.loc[baskets.customer_id.isin(ids)]
        active_lines = lines.loc[lines.customer_id.isin(ids)].copy()
        active_lines['basket'] = active_lines.basket.map(pd.Series(np.arange(len(active)), index=active.index))
        active = active.reset_index(drop=True).rename_axis('basket')
        membership_by_name = basket_memberships(active_lines, active, fit)
        baseline = pd.read_parquet(ROOT / f'research/oof/tail_predictions_{month}.parquet').lightgbm_plain_half.loc[query]
        orders = features.loc[query, 'n_orders']
        y = labels.loc[query]
        control = report_metrics(y, baseline, baseline, orders)
        selected = control_selection(features, labels, dates, quarters, fit)
        core = features.loc[ids, selected]
        for name in ('core', 'gender', 'basket_types'):
            coverage = {}
            design = core
            if name != 'core':
                members, kinds = membership_by_name[name]
                member_counts = members.groupby('basket').size()
                held_baskets = active.index[active.customer_id.isin(query)]
                coverage['mixed_baskets_pct'] = 100 * member_counts.reindex(held_baskets, fill_value=0).gt(1).mean()
                block, supported, distinct = rhythm_features(active, members, kinds, ids, name)
                design = core.join(block)
                for key, subset in [('all', query), ('medium', orders.index[orders.between(7, 15)])]:
                    coverage[f'{key}_two_supported_pct'] = 100 * supported.loc[subset].mean()
                    coverage[f'{key}_two_distinct_pct'] = 100 * distinct.loc[subset].mean()
            model = booster('lightgbm').set_params(n_jobs=4).fit(design.loc[fit], labels.loc[fit])
            probability = model.predict_proba(design.loc[query])[:, 1]
            score = .5 * (quarters.loc[query] + probability)
            if name == 'core' and not np.allclose(score, baseline, atol=1e-14, rtol=0):
                raise AssertionError('Core did not reproduce the archived validation forecasts.')
            result = report_metrics(y, score, baseline, orders)
            report.append(dict(month=month, model=name, **result,
                               delta_vs_tail=result['ap20'] - control['ap20'],
                               medium_delta_vs_tail=result['medium_ap20'] - control['medium_ap20'], **coverage))
            print(f'{month} {name}: AP {result["ap20"]:.5f}, medium AP {result["medium_ap20"]:.5f}', flush=True)
        write_report(report, RESULT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' if (ROOT / 'data/train.csv').is_file() else ROOT)
    parser.add_argument('--cache', type=Path, default=ROOT / '.checkpoints/tail6')
    args = parser.parse_args()
    run(args.data, args.cache)
