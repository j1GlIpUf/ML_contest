"""Compare equal pair weights with the exact AP@20% cost of a rank swap."""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from research.rerank import compare
from research.tail_analysis import splice
from research.tail_models import booster

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / 'research/oof'
REPORT = ROOT / 'research/results/ap_swap.csv'
PREDICTIONS = OOF / 'ap_swap_predictions.parquet'
CONTROL = 'lightgbm_plain_half'
ARMS = ('equal', 'ap_swap')
STRENGTHS = (.025, .05, .10, .20, .35, .50, 1.)


def cutoff(size):
    return int(np.ceil(.2 * size))


def rank_order(scores):
    return np.argsort(-np.asarray(scores), kind='stable')


def ranked_ap(y, k, denominator):
    hits = y[:k]
    return (hits @ (np.cumsum(hits) / np.arange(1, k + 1))) / denominator


def swap_cost(y, buyer, nonbuyer, k):
    """Absolute AP change for opposite labels at zero-based positions in the full ranking."""
    earlier = np.minimum(buyer, nonbuyer) + 1
    later = np.maximum(buyer, nonbuyer) + 1
    counts = np.r_[0, np.cumsum(y)]
    reciprocal_hits = np.r_[0., np.cumsum(y / np.arange(1, len(y) + 1))]
    in_head = earlier <= k
    a, b = earlier[in_head], later[in_head]
    # Moving the buyer up inserts its precision at a, improves every
    # intervening buyer's precision by 1/rank, and removes its old term at b.
    improvement = (counts[a - 1] + 1) / a
    improvement += reciprocal_hits[np.minimum(b - 1, k)] - reciprocal_hits[a]
    improvement -= np.where(b <= k, counts[b] / b, 0.)
    weights = np.zeros(len(buyer))
    weights[in_head] = improvement / max(1, min(k, counts[-1]))
    return weights


def check_swaps(y, k, rng):
    positions = np.arange(len(y))
    denominator = max(1, min(k, y.sum()))
    baseline = ranked_ap(y, k, denominator)
    largest_error = 0.
    checked = 0
    # Include both directions across the boundary and the zero-weight tail.
    for buyer_inside, nonbuyer_inside in ((True, True), (True, False), (False, True), (False, False)):
        buyer = rng.choice(positions[(y == 1) & ((positions < k) == buyer_inside)], 128)
        nonbuyer = rng.choice(positions[(y == 0) & ((positions < k) == nonbuyer_inside)], 128)
        expected = swap_cost(y, buyer, nonbuyer, k)
        exchanged = y.copy()
        for i, j, estimate in zip(buyer, nonbuyer, expected):
            exchanged[i], exchanged[j] = 0, 1
            actual = abs(ranked_ap(exchanged, k, denominator) - baseline)
            largest_error = max(largest_error, abs(actual - estimate))
            exchanged[i], exchanged[j] = 1, 0
        checked += len(buyer)
    if largest_error > 1.e-12:
        raise AssertionError(f'AP swap audit failed with error {largest_error:g}.')
    return checked, largest_error


def assemble_pairs(labels, prior, dates):
    sample_rng = np.random.default_rng(73)
    audit_rng = np.random.default_rng(208)
    positives, negatives, costs, audits = [], [], [], []
    for date in sorted(dates.unique()):
        rows = np.flatnonzero(np.asarray(dates == date))
        ranked = rows[rank_order(prior.iloc[rows])]
        y = labels.iloc[ranked].to_numpy()
        k = cutoff(len(y))
        head, near_tail = np.arange(k), np.arange(k, 2 * k)
        head_buyers, head_nonbuyers = head[y[head] == 1], head[y[head] == 0]
        tail_buyers, tail_nonbuyers = near_tail[y[near_tail] == 1], near_tail[y[near_tail] == 0]
        for buyers, nonbuyers in ((head_buyers, head_nonbuyers), (head_buyers, tail_nonbuyers),
                                  (tail_buyers, head_nonbuyers)):
            sampled = np.vstack([sample_rng.choice(nonbuyers, 8, replace=False) for _ in buyers]).ravel()
            repeated = np.repeat(buyers, 8)
            positives.append(ranked[repeated])
            negatives.append(ranked[sampled])
            costs.append(swap_cost(y, repeated, sampled, k))
        count, error = check_swaps(y, k, audit_rng)
        audits.append(dict(split='swap_audit', month=date.month, verified_swaps=count,
                           max_swap_error=error, within_pairs=8 * len(head_buyers),
                           cross_pairs=8 * (len(head_buyers) + len(tail_buyers))))
    weight = np.concatenate(costs)
    if not (weight > 0).all():
        raise AssertionError('Expected a positive AP change for every sampled pair.')
    return np.concatenate(positives), np.concatenate(negatives), weight, audits


def upper_model():
    return booster('lightgbm').set_params(n_jobs=4)


def select_features(features, labels, ranking):
    pool = ranking.index[rank_order(ranking)[:2 * cutoff(len(ranking))]]
    selector = upper_model().fit(features.loc[pool], labels.loc[pool])
    gain = pd.Series(selector.booster_.feature_importance(importance_type='gain'), index=features.columns)
    return gain.sort_values(ascending=False, kind='stable').head(160).index


def january_oof(features, labels, upstream):
    ids = labels.index
    # The stored quarter scores for other outer folds have upstream models
    # that saw the current held fold. Use inner OOF on raw history instead to
    # form the selector's top 40%, and the saved honest quarter score for the blend.
    compact = features.loc[ids, features.columns.str.match(
        r'^(recency$|age_days$|n_orders$|orders_[0-9]+$|rhythm_order_all_|window_typical_band$)')]
    result = []
    for fold in sorted(upstream.fold.unique()):
        query = ids[upstream.loc[ids, 'fold'].eq(fold)]
        fit = ids.difference(query, sort=False)
        ranking = pd.Series(index=fit, dtype=float)
        inner = StratifiedKFold(3, shuffle=True, random_state=73)
        for a, b in inner.split(fit, labels.loc[fit]):
            proxy = upper_model().set_params(n_estimators=120)
            proxy.fit(compact.loc[fit[a]], labels.loc[fit[a]])
            ranking.loc[fit[b]] = proxy.predict_proba(compact.loc[fit[b]])[:, 1]
        selected = select_features(features.loc[fit], labels.loc[fit], ranking)
        model = upper_model().fit(features.loc[fit, selected], labels.loc[fit])
        probability = model.predict_proba(features.loc[query, selected])[:, 1]
        blend = .5 * (upstream.loc[query, 'quarters'] + probability)
        result.append(blend)
        print(f'January upper-model OOF fold {fold + 1}/5', flush=True)
    oof = pd.concat(result, verify_integrity=True).reindex(ids)
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete January upper-model OOF.')
    return oof


def read_inputs(cache):
    cohorts = pd.read_csv(ROOT / 'train.csv', parse_dates=['scoring_date']).set_index('customer_id', verify_integrity=True)
    features = pd.concat([pd.read_parquet(cache / f'features_{i}.parquet') for i in (1, 2, 3)],
                         verify_integrity=True).loc[cohorts.index]
    january = pd.concat([pd.read_parquet(OOF / f'january_fold_{i}.parquet') for i in range(5)], verify_integrity=True)
    january_ids = cohorts.index[cohorts.scoring_date.dt.month.eq(1)]
    if set(january.index) != set(january_ids) or set(january.fold) != set(range(5)):
        raise ValueError('Saved January folds do not match training customers.')
    reference = pd.read_csv(ROOT / 'submission_tail.csv').set_index('customer_id', verify_integrity=True).score
    april = pd.read_parquet(OOF / 'tail_predictions_4.parquet')[CONTROL]
    if not np.allclose(reference.loc[april.index], april, atol=1e-14, rtol=0) or len(reference) != len(april):
        raise ValueError('The archived control does not reproduce submission_tail.csv.')
    validation = [pd.read_parquet(OOF / f'tail_predictions_{m}.parquet')[CONTROL] for m in (2, 3)]
    stage_one = pd.concat([january.drop(columns='fold'), *[pd.read_parquet(OOF / f'scores_{m}.parquet')
                           for m in (2, 3)]], verify_integrity=True).loc[cohorts.index]
    oof = january_oof(features, cohorts.target.loc[january_ids], january)
    prior = pd.concat([oof, *validation], verify_integrity=True).loc[cohorts.index]
    if not np.isfinite(prior).all():
        raise ValueError('Incomplete control forecasts.')
    return cohorts, features, stage_one, prior


def fit_correction(fit, query, labels, base, pairs, weights):
    pos, neg = pairs
    weights = weights * (len(labels) / weights.sum())
    y = labels.to_numpy()

    def objective(raw, _):
        score = raw + base
        chance = expit(score[neg] - score[pos])
        error = weights * chance
        curve = error * (1. - chance)
        grad = np.bincount(neg, error, minlength=len(y)) - np.bincount(pos, error, minlength=len(y))
        hess = np.bincount(pos, curve, minlength=len(y)) + np.bincount(neg, curve, minlength=len(y))
        prob = expit(score)
        return grad + .5 * (prob - y), hess + .5 * prob * (1. - prob)

    params = dict(objective=objective, learning_rate=.035, num_leaves=8, min_data_in_leaf=125,
                  lambda_l2=20, feature_fraction=.65, num_threads=4, verbosity=-1, seed=73)
    model = lgb.train(params, lgb.Dataset(fit, label=y), num_boost_round=170)
    return pd.Series(model.predict(query, num_threads=4), index=query.index)


def correction_scores(prior, delta, strength):
    pool = delta.index
    z = logit(prior.loc[pool].clip(1e-6, 1 - 1e-6)) + strength * delta
    return splice(prior, z, pool)


def evaluate(labels, score, prior, orders):
    k = cutoff(len(labels))
    ref = prior.index[rank_order(prior)[:k]]
    denominator = max(1, min(k, labels.sum()))
    kept = labels.loc[ref].to_numpy()
    fixed = ranked_ap(kept[rank_order(score.loc[ref])], k, denominator)
    baseline = ranked_ap(kept, k, denominator)
    metrics = compare(labels, score, prior, orders)
    return dict(ap20=metrics['ap20'], delta_ap20=metrics['ap20'] - baseline,
                fixed_top5000_ap20=fixed, fixed_top5000_delta=fixed - baseline,
                buyers_top5000=metrics['buyers'], added_buyers=metrics['added_buyers'],
                dropped_buyers=metrics['dropped_buyers'], changed_top5000=metrics['changed'])


def run(cache):
    cohorts, features, upstream, prior = read_inputs(cache)
    dates, labels = cohorts.scoring_date, cohorts.target
    rows, forecasts = [], []
    strength = None
    audited_months = set()
    for month in (2, 3):
        fit = dates.index[dates.dt.month.lt(month)]
        held = dates.index[dates.dt.month.eq(month)]
        k = cutoff(len(held))
        ref = prior.loc[held]
        query = held[rank_order(ref)[:2 * k]]
        selected = select_features(features.loc[fit], labels.loc[fit], prior.loc[fit])
        design = features.loc[fit.append(query), selected].join(upstream.add_prefix('upstream_'))
        design['tail_percentile'] = prior.groupby(dates).rank(pct=True)
        pos, neg, costs, audits = assemble_pairs(labels.loc[fit], prior.loc[fit], dates.loc[fit])
        for audit in audits:
            if audit['month'] not in audited_months:
                rows.append(audit)
                audited_months.add(audit['month'])
        deltas = {}
        for arm, weight in zip(ARMS, (np.ones_like(costs), costs)):
            deltas[arm] = fit_correction(design.loc[fit], design.loc[query], labels.loc[fit],
                                         logit(prior.loc[fit].clip(1e-6, 1 - 1e-6)).to_numpy(), (pos, neg), weight)
        y, orders = labels.loc[held], features.loc[held, 'n_orders']
        rows.append(dict(split='validation', month=month, model='submission_tail', strength=0.,
                         selected=True, **evaluate(y, ref, ref, orders)))
        candidates = {}
        for scale in STRENGTHS if month == 2 else (strength,):
            scores = {arm: correction_scores(ref, deltas[arm], scale) for arm in ARMS}
            metrics = {arm: evaluate(y, score, ref, orders) for arm, score in scores.items()}
            candidates[scale] = (scores, metrics)
        if month == 2:
            # One shared coefficient, selected symmetrically between the two arms.
            strength = max(candidates, key=lambda s: (sum(m['ap20'] for m in candidates[s][1].values()), -s))
        for scale, (_, metrics) in candidates.items():
            for arm, values in metrics.items():
                rows.append(dict(split='validation', month=month, model=arm, strength=scale,
                                 selected=scale == strength, **values))
        pd.DataFrame(rows).to_csv(REPORT, index=False)
        scores, metrics = candidates[strength]
        forecasts.append(pd.DataFrame({'scoring_date': dates.loc[held], 'submission_tail': ref, **scores}))
        pd.concat(forecasts, verify_integrity=True).to_parquet(PREDICTIONS)
        print(f'{month}: strength {strength:g}, equal {metrics["equal"]["ap20"]:.5f}, '
              f'AP swap {metrics["ap_swap"]["ap20"]:.5f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / '.checkpoints/tail6',
                        help='Directory with the existing tail feature matrices.')
    run(parser.parse_args().cache)
