"""Replay complete 90-day windows and test auxiliary intensity features."""

import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold

from research.rerank import compare
from research.tail_models import booster

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'research/results/exp_aux_intensity.csv'
OFFSETS = (120, 210, 300, 365)
TARGETS = ('target_n_orders_90d', 'days_to_first_order', 'target_30d')
DATA_FILES = ('train.csv', 'test.csv', 'orders.parquet')
BLIND = pd.Timedelta(days=14)
HORIZON = pd.Timedelta(days=90)
BUFFER = pd.Timedelta(days=30)
PARAMS = dict(n_estimators=180, learning_rate=.05, num_leaves=15,
              min_child_samples=55, reg_lambda=8, n_jobs=4, verbosity=-1, random_state=26)


def read_data(path):
    if not path.exists():
        raise FileNotFoundError(f'No Lamoda data at {path}. Set --data or LAMODA_DATA.')

    def read(name, reader, **kwargs):
        if path.is_dir():
            return reader(path / name, **kwargs)
        with ZipFile(path) as archive, archive.open(name) as source:
            return reader(source, **kwargs)

    train, test = [read(name, pd.read_csv, parse_dates=['scoring_date']).set_index('customer_id')
                   for name in DATA_FILES[:2]]
    cohorts = pd.concat([train, test], verify_integrity=True)
    if (not train.target.isin([0, 1]).all() or cohorts.scoring_date.isna().any()
            or train.scoring_date.nunique() != 3 or test.scoring_date.nunique() != 1
            or train.scoring_date.max() >= test.scoring_date.min()):
        raise ValueError('Expected three chronological labelled cohorts and one subsequent test cohort.')
    orders = read(DATA_FILES[2], pd.read_parquet,
                  columns=['customer_id', 'order_id', 'order_date', 'paid_price'])
    orders['order_date'] = pd.to_datetime(orders.order_date)
    order_keys = ['customer_id', 'order_id']
    receipts = orders[orders.customer_id.isin(cohorts.index)].groupby(order_keys, as_index=False).agg(
        order_date=('order_date', 'min'), latest=('order_date', 'max'),
        paid=('paid_price', 'sum'), items=('order_id', 'size'))
    if not receipts.order_date.eq(receipts.pop('latest')).all():
        raise ValueError('Each order must have a single date before its lines can be combined.')
    receipts = receipts.merge(cohorts[['scoring_date']], on='customer_id', validate='many_to_one')
    receipts = receipts[receipts.order_date.le(receipts.scoring_date - BLIND)]
    return cohorts, receipts.sort_values(['customer_id', 'order_date', 'order_id'], kind='stable')


def core_features(receipts, ids, cutoff):
    history = receipts[receipts.order_date.le(cutoff - BLIND)]
    grouped = history.groupby('customer_id', sort=False)
    frame = grouped.agg(n_orders=('order_id', 'size'), n_days=('order_date', 'nunique'),
                        n_items=('items', 'sum'), total_paid=('paid', 'sum'),
                        first=('order_date', 'min'), last=('order_date', 'max'))
    frame['recency'] = (cutoff - frame.pop('last')).dt.days
    frame['age_days'] = (cutoff - frame.pop('first')).dt.days
    frame['mean_basket'] = frame.total_paid / frame.n_orders
    days = history[['customer_id', 'order_date']].drop_duplicates()
    gaps = days.groupby('customer_id', sort=False).order_date.diff().dt.days
    rhythm = gaps.groupby(days.customer_id, sort=False).agg(['mean', 'median', 'std', 'last'])
    frame = frame.join(rhythm.add_prefix('gap_'))
    frame['recency_over_gap'] = frame.recency / frame.gap_median
    frame['median_due_in_days'] = frame.gap_median - frame.recency
    for window in (30, 90, 180, 365):
        recent = history[history.order_date.ge(cutoff - pd.Timedelta(days=window))]
        totals = recent.groupby('customer_id', sort=False).agg(
            orders=('order_id', 'size'), spend=('paid', 'sum'), items=('items', 'sum'))
        frame = frame.join(totals.reindex(frame.index, fill_value=0).add_suffix(f'_{window}'))
    return frame.reindex(ids).astype('float32')


def recency_cap(core, dates):
    # A shared endpoint inside the observed history suggests a cohort activity cutoff.
    # An endpoint at the history limit is evidence of observation coverage only.
    endpoints = core[['recency', 'age_days']].groupby(dates).max().dropna()
    if (endpoints.empty or (endpoints.age_days - endpoints.recency).min() <= 1
            or endpoints.recency.max() - endpoints.recency.min() > 1):
        return None
    return int(endpoints.recency.max())


def distributions(core):
    result = {'customers': len(core), 'without_history': int(core.recency.isna().sum())}
    for name in ('recency', 'age_days'):
        for quantile, label in [(0, 'min'), (.5, 'p50'), (.9, 'p90'), (.99, 'p99'), (1, 'max')]:
            result[f'{name}_{label}'] = core[name].quantile(quantile)
    return result


def make_history(receipts, ids, scoring_date):
    """Anchor each replay to the scoring date of its source cohort."""
    windows = {}
    for offset in OFFSETS:
        cutoff = scoring_date - pd.Timedelta(days=offset)
        end = cutoff + HORIZON
        if end > scoring_date - BUFFER or cutoff < receipts.order_date.min():
            raise ValueError(f'Target window for offset {offset} extends past available history.')
        features = core_features(receipts, ids, cutoff)
        future = receipts[receipts.order_date.ge(cutoff) & receipts.order_date.lt(end)]
        grouped = future.groupby('customer_id', sort=False)
        first = (grouped.order_date.min() - cutoff).dt.days.reindex(ids, fill_value=91)
        targets = pd.DataFrame(dict(target_n_orders_90d=grouped.order_id.size().reindex(ids, fill_value=0),
                                    days_to_first_order=first, target_30d=first.lt(30).astype('int8')),
                               index=ids)
        windows[offset] = features.join(targets)
    return windows


def eligible(window, ids, cap):
    rows = window.loc[ids]
    rows = rows[rows.recency.notna()]
    return rows if cap is None else rows[rows.recency.le(cap)]


def auxiliary(windows, source_ids, query, cap):
    train = pd.concat([eligible(window, source_ids.intersection(window.index), cap)
                       for window in windows])
    if train.empty or not train.target_n_orders_90d.any() or train.target_30d.nunique() != 2:
        raise ValueError('The eligible historical windows do not contain enough target variation.')
    X = train.loc[:, query.columns]
    count = LGBMRegressor(objective='poisson', **PARAMS).fit(X, train[TARGETS[0]])
    wait = LGBMRegressor(objective='regression_l1', **PARAMS).fit(X, np.log1p(train[TARGETS[1]]))
    fast = LGBMClassifier(**PARAMS).fit(X, train[TARGETS[2]])
    pred = pd.DataFrame(dict(aux_orders_90d=count.predict(query),
                             aux_days_to_first=np.expm1(wait.predict(query)),
                             aux_purchase_30d=fast.predict_proba(query)[:, 1]), index=query.index)
    if not np.isfinite(pred.to_numpy()).all():
        raise ValueError('Nonfinite auxiliary predictions.')
    return pred


def published_scores(path):
    january = pd.concat([pd.read_parquet(path / f'january_fold_{i}.parquet').drop(columns='fold')
                         for i in range(5)], verify_integrity=True)
    later = [pd.read_parquet(path / f'scores_{m}.parquet') for m in (2, 3, 4)]
    return pd.concat([january, *later], verify_integrity=True)


def main_features(core, scores, cache):
    paths = [cache / f'features_{month}.parquet' for month in range(1, 5)]
    if all(path.is_file() for path in paths):
        data = pd.concat([pd.read_parquet(path) for path in paths], verify_integrity=True)
        return data.loc[core.index], 'tail_features'
    return core.join(scores.add_prefix('prior_')), 'core_and_oof_scores'


def main_predictions(features, aux, labels, scores, dates, fit, query):
    rank = scores.quarters.loc[fit].groupby(dates.loc[fit]).rank(pct=True, method='first')
    pool = rank.index[rank.gt(.6)]
    selector = booster('lightgbm').set_params(n_jobs=4).fit(features.loc[pool], labels.loc[pool])
    gains = pd.Series(selector.booster_.feature_importance(importance_type='gain'), index=features.columns)
    selected = list(gains.sort_values(ascending=False, kind='stable').head(160).index)
    prior = scores.quarters.loc[query]
    for name, data in [('matched_control', features[selected]), ('with_aux', features[selected].join(aux))]:
        model = booster('lightgbm').set_params(n_jobs=4)
        model.fit(data.loc[fit], labels.loc[fit])
        score = pd.Series(model.predict_proba(data.loc[query])[:, 1], index=query)
        yield name, (score + prior) / 2


def fixed_top_ap(labels, score, reference):
    limit = int(np.ceil(.2 * len(labels)))
    top = reference.reindex(labels.index).nlargest(limit, keep='first').index
    hits = labels.loc[score.loc[top].sort_values(ascending=False, kind='stable').index].to_numpy()
    return ((hits.cumsum() / np.arange(1, limit + 1)) @ hits) / max(1, min(limit, labels.sum()))


def write_report(rows, path):
    with NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            pd.DataFrame(rows).to_csv(output, index=False, lineterminator='\n')
            output.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def run(args):
    cohorts, receipts = read_data(args.data)
    dates, labels = cohorts.scoring_date, cohorts.target
    by_date = {date: group.index for date, group in cohorts.groupby('scoring_date')}
    receipt_groups = dict(tuple(receipts.groupby('scoring_date')))
    history_by_date = {date: receipt_groups.get(date, receipts.iloc[:0]) for date in by_date}
    core = pd.concat([core_features(history_by_date[date], ids, date) for date, ids in by_date.items()])
    core = core.reindex(cohorts.index)
    report = []
    for date, ids in by_date.items():
        report.append(dict(stage='cohort', status='measured', date=date.date(),
                           split='train' if labels.loc[ids].notna().all() else 'test',
                           actual_purchase_rate=labels.loc[ids].mean(),
                           recency_cap=recency_cap(core.loc[ids], dates.loc[ids]),
                           **distributions(core.loc[ids])))
    write_report(report, args.report)
    summary = ', '.join(f'{row["date"]} {row["recency_min"]:.0f}–{row["recency_max"]:.0f}' for row in report)
    print(f'Recency ranges by cohort: {summary}', flush=True)
    if args.diagnose_only:
        return

    scores = published_scores(args.scores).loc[core.index]
    dates_in_order = list(by_date)
    windows = {date: make_history(history_by_date[date], by_date[date], date)
               for date in dates_in_order[:-1]}
    features, feature_source = main_features(core, scores, args.cache)
    january = by_date[dates_in_order[0]]
    jan_features = core.loc[january]
    folds = KFold(n_splits=5, shuffle=True, random_state=26)
    predictions = []
    # January has no earlier cohort, so each row gets a prediction from other
    # January customers and only from windows already observable in January.
    for fit_pos, query_pos in folds.split(january):
        fit = january[fit_pos]
        cap = recency_cap(core.loc[fit], dates.loc[fit])
        predictions.append(auxiliary(windows[dates_in_order[0]].values(), fit, jan_features.iloc[query_pos], cap))
    raw_aux = pd.concat(predictions, verify_integrity=True)
    for i, date in enumerate(dates_in_order[1:], start=1):
        source_dates = dates_in_order[:i]
        fit = labels.index[dates.lt(date) & labels.notna()]
        query = by_date[date]
        cap = recency_cap(core.loc[fit], dates.loc[fit])
        training_windows = []
        for source in source_dates:
            for offset, window in windows[source].items():
                training_windows.append(window)
                kept = eligible(window, by_date[source], cap)
                report.append(dict(stage='history', status='measured', date=date.date(),
                                   source_date=source.date(), offset_days=offset, recency_cap=cap,
                                   historical_cutoff=(source - pd.Timedelta(days=offset)).date(),
                                   no_prior_orders=int(window.recency.isna().sum()),
                                   over_recency_cap=int(window.recency.notna().sum()) - len(kept),
                                   all_customers=len(window), historical_purchase_90d=kept[TARGETS[0]].gt(0).mean(),
                                   before_activity_filter_purchase_90d=window.loc[window.recency.notna(), TARGETS[0]].gt(0).mean(),
                                   historical_purchase_30d=kept[TARGETS[2]].mean(), **distributions(kept)))
        query_aux = auxiliary(training_windows, fit, core.loc[query], cap)
        raw_aux = pd.concat([raw_aux, query_aux], verify_integrity=True)
        ranks = raw_aux.groupby(dates.loc[raw_aux.index]).rank(pct=True).add_suffix('_rank')
        aux = raw_aux.join(ranks)
        candidates = dict(main_predictions(features, aux, labels, scores, dates, fit, query))
        if labels.loc[query].isna().all():
            report.append(dict(stage='test', status='predicted', date=date.date(), model='with_aux',
                               feature_source=feature_source, customers=len(query)))
            if args.predictions:
                query_aux.join(ranks.loc[query]).join(pd.DataFrame(candidates)).to_parquet(args.predictions)
        else:
            y = labels.loc[query].astype(int)
            prior = scores.quarters.loc[query]
            control = candidates['matched_control']
            order_count = core.n_orders.loc[query]
            quarters_ap = compare(y, prior, prior, order_count)['ap20']
            control_ap = compare(y, control, control, order_count)['ap20']
            for name, score in dict(quarters=prior, **candidates).items():
                metrics = compare(y, score, control, order_count)
                fixed_ap = fixed_top_ap(y, score, control)
                report.append(dict(stage='validation', status='measured', date=date.date(), model=name,
                                   feature_source=feature_source, ap20=metrics['ap20'],
                                   buyers_top5000=metrics['buyers'], changed_top5000=metrics['changed'],
                                   delta_vs_quarters=metrics['ap20'] - quarters_ap,
                                   delta_vs_matched_control=metrics['ap20'] - control_ap,
                                   fixed_top5000_ap20=fixed_ap, fixed_top5000_delta=fixed_ap - control_ap,
                                   added_buyers=metrics['added_buyers'], dropped_buyers=metrics['dropped_buyers']))
            row = report[-1]
            print(f'{date.date()} AP@20% {row["ap20"]:.5f}, Δ {row["delta_vs_matched_control"]:+.5f}, '
                  f'buyers {row["buyers_top5000"]}', flush=True)
        write_report(report, args.report)


def main():
    data_paths = (ROOT / 'data', ROOT, ROOT / 'lamoda-hse-repeat-purchase-2026.zip')
    default_data = next((path for path in data_paths if path.is_file()
                         or all((path / name).is_file() for name in DATA_FILES)), data_paths[0])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(os.getenv('LAMODA_DATA', str(default_data))))
    parser.add_argument('--scores', type=Path, default=ROOT / 'research/oof')
    parser.add_argument('--cache', type=Path, default=ROOT / '.checkpoints/tail6')
    parser.add_argument('--report', type=Path, default=RESULT)
    parser.add_argument('--predictions', type=Path, help='Optional output Parquet for unlabelled customers.')
    parser.add_argument('--diagnose-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
