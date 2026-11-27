"""Compare anti-FP features with the archived tail model on February and March.

Run from the repository root with python -m research.exp_anti_fp --data data.
An original competition ZIP can be passed in place of the data directory.
The addon estimates a log-odds correction from paired LightGBM fits with
and without the feature block. Both fits use earlier cohorts and their
out-of-sample stage-one forecasts. The archived tail forecast stays frozen.
"""
import argparse
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit, log_ndtr, logit

from research.rerank import compare

RESEARCH = Path(__file__).resolve().parent
RESULT = RESEARCH / 'results/exp_anti_fp.csv'
ARCHIVE = 'lamoda-hse-repeat-purchase-2026.zip'
FILES = ('train.csv', 'orders.parquet', 'products.parquet')
ORDER_COLUMNS = [
    'customer_id', 'order_id', 'order_date', 'product_id', 'paid_price',
    'loyalty_percent', 'discount_percent',
]
CONTROL_AP = {2: .77701, 3: .78434}
BASELINE = 'lightgbm_plain_half'
TOP_K = 5000
FIRST_OBSERVED_AGE = 14


def read_inputs(data):
    def read(reader, name, **kwargs):
        if data.is_dir():
            return reader(data / name, **kwargs)
        with ZipFile(data) as archive, archive.open(name) as stream:
            return reader(stream, **kwargs)

    train = read(pd.read_csv, 'train.csv', parse_dates=['scoring_date'])
    orders = read(pd.read_parquet, 'orders.parquet', columns=ORDER_COLUMNS)
    products = read(pd.read_parquet, 'products.parquet', columns=['product_id', 'category'])
    return train, orders, products


def window(history, width):
    # The supplied histories end 14 days before scoring. Include that day.
    return history.loc[history.age.between(FIRST_OBSERVED_AGE, FIRST_OBSERVED_AGE + width - 1)]


def features_for_cutoff(orders, products, ids, cutoff):
    """Compute only from records timestamped strictly before this cutoff."""
    history = orders.loc[orders.customer_id.isin(ids) & orders.order_date.lt(cutoff)]
    history = history.merge(products, on='product_id', how='left', validate='many_to_one')
    history['age'] = (cutoff.normalize() - history.order_date.dt.normalize()).dt.days
    receipts = (
        history.groupby(['customer_id', 'order_id'], sort=False)
        .agg(order_date=('order_date', 'min'), age=('age', 'max'),
             spend=('paid_price', 'sum'), units=('product_id', 'size'),
             loyalty=('loyalty_percent', 'mean'), discount=('discount_percent', 'mean'))
        .reset_index()
        .sort_values(['customer_id', 'order_date', 'order_id'], kind='stable')
    )
    customer = receipts.groupby('customer_id', sort=False)
    receipts['gap'] = -customer.age.diff()
    receipts['prev_gap'] = customer.gap.shift(1)
    receipts['older_gap'] = customer.gap.shift(2)
    receipts['prev_spend'] = customer.spend.shift(1)
    last = customer.tail(1).set_index('customer_id')
    median_gap = customer.gap.median()

    control = pd.DataFrame(index=ids)
    control['n_orders'] = customer.size().reindex(ids, fill_value=0)
    control['recency'] = last.age
    extra = pd.DataFrame(index=ids)

    # Use distinct days to estimate the cycle so same-day orders cannot
    # shorten its period. Require two intervals and a robust CV at most .35.
    days = receipts.drop_duplicates(['customer_id', 'age'])
    day_gaps = -days.groupby('customer_id').age.diff()
    gap_customer = days.customer_id
    period = day_gaps.groupby(gap_customer).median()
    mad = (day_gaps - gap_customer.map(period)).abs().groupby(gap_customer).median()
    variation = 1.4826 * mad / period
    regular = period.gt(70) & day_gaps.groupby(gap_customer).count().ge(2) & variation.le(.35)
    due = (period - last.age).where(regular)
    spread = (1.4826 * mad).clip(lower=period * .05)
    # The latest observed boundary is day -14. Condition the timing mass
    # on no intervening observed order before that boundary.
    left = (-FIRST_OBSERVED_AGE - due) / spread
    right = -due / spread
    risk = -np.expm1(log_ndtr(-right) - log_ndtr(-left))
    extra['blind_long_cycle_due'] = (due.ge(-FIRST_OBSERVED_AGE) & due.lt(0)).astype(float)
    extra['blind_long_cycle_risk'] = risk.fillna(0)

    yearly = window(receipts, 365).groupby('customer_id')
    year_categories = window(history, 365).groupby('customer_id').category.nunique()
    for width in (45, 60):
        recent = window(receipts, width).groupby('customer_id')
        for field in ('spend', 'units'):
            recent_rate = recent[field].sum().reindex(ids, fill_value=0) / width
            annual_rate = yearly[field].sum().replace(0, np.nan) / 365
            extra[f'{field}_{width}_pace'] = recent_rate / annual_rate
        recent_categories = window(history, width).groupby('customer_id').category.nunique()
        extra[f'category_coverage_{width}'] = (
            recent_categories.reindex(ids, fill_value=0) / year_categories.replace(0, np.nan)
        )

    # Split receipts at gaps larger than the customer's median. Estimate
    # normal_size from the prior, fully observed sessions.
    boundary = receipts.customer_id.map(median_gap)
    starts = receipts.gap.isna() | receipts.gap.gt(boundary)
    session_id = starts.groupby(receipts.customer_id).cumsum()
    sizes = receipts.groupby(['customer_id', session_id], sort=False).size()
    current_session = sizes.groupby(level=0, sort=False).tail(1)
    normal_size = sizes.drop(current_session.index).groupby(level=0).median()
    last_size = current_session.droplevel(1)
    extra['session_finished'] = last_size.ge(normal_size).astype(float).where(normal_size.notna())
    extra['last_spend_change'] = last.spend - last.prev_spend
    extra['last_spend_ratio'] = last.spend / last.prev_spend.replace(0, np.nan)

    for width in (180, 365):
        maximum = window(receipts, width).groupby('customer_id').loyalty.max()
        extra[f'loyalty_drop_{width}'] = last.loyalty - maximum
    extra['gap_ratio_previous'] = last.gap / last.prev_gap.replace(0, np.nan)
    extra['gap_ratio_median'] = last.gap / median_gap.replace(0, np.nan)
    growing = last.gap.gt(last.prev_gap)
    extra['gaps_growing_2'] = growing.astype(float).where(last.prev_gap.notna())
    extra['gaps_growing_3'] = (
        (growing & last.prev_gap.gt(last.older_gap)).astype(float).where(last.older_gap.notna())
    )

    bins = pd.cut(receipts.gap, [-1, 0, 3, 7, 14, 30, 60, 120, 365, np.inf])
    counts = receipts.groupby([receipts.customer_id, bins], observed=True).size()
    fractions = counts / counts.groupby(level=0).transform('sum')
    extra['gap_entropy'] = (-fractions * np.log(fractions)).groupby(level=0).sum()
    next_gap = customer.gap.shift(-1)
    for days in (1, 2, 3):
        clustered = receipts.gap.le(days) | next_gap.le(days)
        extra[f'burst_order_share_{days}'] = clustered.groupby(receipts.customer_id).mean()
    high_discount = receipts.discount.gt(receipts.discount.median()).where(receipts.discount.notna())
    extra['discount_above_cutoff_median'] = high_discount.groupby(receipts.customer_id).mean()
    return control, extra.astype('float32')


def score_correction(features, labels, dates, scores, fit, query):
    baseline_log_odds = logit(scores.loc[fit, 'quarters'].clip(1e-5, 1 - 1e-5))
    rank = scores.loc[fit, 'quarters'].groupby(dates.loc[fit]).rank(pct=True)
    weights = 1 + 2 * rank.ge(.6) + features.loc[fit, 'n_orders'].between(4, 15)
    model = LGBMClassifier(
        n_estimators=180, learning_rate=.035, num_leaves=9, min_child_samples=120,
        reg_lambda=12, n_jobs=4, verbosity=-1, random_state=42,
    )
    model.fit(features.loc[fit], labels.loc[fit], init_score=baseline_log_odds, sample_weight=weights)
    return model.booster_.predict(features.loc[query], raw_score=True, num_threads=4)


def top_false_positives(labels, scores, medium):
    ranked = scores.loc[labels.index].sort_values(ascending=False, kind='stable').index[:TOP_K]
    return ranked[labels.loc[ranked].eq(0) & medium.loc[ranked]]


def result_rows(month, labels, reference, candidate, order_count):
    medium = order_count.between(4, 15)
    before = top_false_positives(labels, reference, medium)
    after = top_false_positives(labels, candidate, medium)
    reference_ap = compare(labels, reference, reference, order_count)['ap20']
    rows = []
    for name, scores, fps in [(BASELINE, reference, before), ('base_plus_anti_fp', candidate, after)]:
        measured = compare(labels, scores, reference, order_count)
        rows.append(dict(
            month=month, model=name, ap20=measured['ap20'],
            delta_ap20_vs_current=measured['ap20'] - reference_ap,
            delta_ap20_vs_quarters_control=measured['ap20'] - CONTROL_AP[month],
            buyers_top5000=measured['buyers'], fp_top5000=TOP_K - measured['buyers'],
            fp_4_15_top5000=len(fps), fp_4_15_removed=len(before.difference(fps)),
            fp_4_15_added=len(fps.difference(before)), fp_4_15_net_removed=len(before) - len(fps),
            status='measured',
        ))
    return rows


def experiment(train, orders, products, oof_dir):
    labels = train.set_index('customer_id', verify_integrity=True)
    if not labels.target.isin([0, 1]).all():
        raise ValueError('Expected binary training labels')
    dates = labels.scoring_date
    cutoffs = sorted(dates.unique())
    if [cutoff.month for cutoff in cutoffs] != [1, 2, 3] or cutoffs[-1].year != cutoffs[0].year:
        raise ValueError('Expected January, February and March of the same year')
    if not dates.value_counts().eq(25000).all():
        raise ValueError('Expected 25,000 customers per cohort')
    january = [
        pd.read_parquet(oof_dir / f'january_fold_{fold}.parquet').drop(columns='fold')
        for fold in range(5)
    ]
    later = [pd.read_parquet(oof_dir / f'scores_{month}.parquet') for month in (2, 3)]
    scores = pd.concat(january + later, verify_integrity=True).reindex(labels.index)
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError('Incomplete stage-one forecasts')
    orders['order_date'] = pd.to_datetime(orders.order_date)
    blocks = [
        features_for_cutoff(orders, products, dates.index[dates.eq(cutoff)], cutoff)
        for cutoff in cutoffs
    ]
    common = scores.join(pd.concat([control for control, _ in blocks]))
    augmented = common.join(pd.concat([extra for _, extra in blocks]))
    rows = []
    for cutoff in cutoffs[1:]:
        month = cutoff.month
        fit = dates.index[dates.lt(cutoff)]
        query = dates.index[dates.eq(cutoff)]
        archived = pd.read_parquet(oof_dir / f'tail_predictions_{month}.parquet', columns=[BASELINE])
        reference = archived[BASELINE].reindex(query)
        if not np.isfinite(reference).all():
            raise ValueError(f'Incomplete tail baseline for month {month}')
        delta = score_correction(augmented, labels.target, dates, scores, fit, query)
        delta -= score_correction(common, labels.target, dates, scores, fit, query)
        # Fixed half-strength to shrink the learned correction without
        # looking at either validation cohort's outcomes.
        candidate = pd.Series(expit(logit(reference.clip(1e-5, 1 - 1e-5)) + .5 * delta), index=query)
        rows.extend(result_rows(month, labels.target.loc[query], reference, candidate, common.n_orders))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Competition data directory or original ZIP')
    parser.add_argument('--oof', type=Path, default=RESEARCH / 'oof')
    parser.add_argument('--output', type=Path, default=RESULT)
    args = parser.parse_args()
    configured = args.data or os.environ.get('LAMODA_DATA')
    if configured:
        candidates = [Path(configured)]
    else:
        candidates = [Path('data'), Path(ARCHIVE), Path('/kaggle/input/competitions') / ARCHIVE[:-4]]
    data = next((p for p in candidates if p.is_file() or all((p / f).is_file() for f in FILES)), None)
    if data is None:
        parser.error('provide train.csv, orders.parquet and products.parquet via --data or LAMODA_DATA')
    report = experiment(*read_inputs(data), args.oof)
    report.to_csv(args.output, index=False)
    print(report[['month', 'model', 'ap20', 'buyers_top5000', 'fp_4_15_top5000',
                  'fp_4_15_net_removed']].to_string(index=False))


if __name__ == '__main__':
    main()
