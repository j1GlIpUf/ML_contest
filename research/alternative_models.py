"""Native categorical boosting, honest pattern encoding and time-fold calibration."""

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from research.window_models import forecast, reference_scale


def categorical_boost(fit, labels, query, tokens, query_tokens, rank=False):
    from catboost import CatBoostClassifier, CatBoostRanker, Pool
    numeric = fit.join(tokens)
    hold = query.join(query_tokens)
    params = dict(iterations=500, depth=5, learning_rate=.06, l2_leaf_reg=8,
                  thread_count=4, allow_writing_files=False, random_seed=42, verbose=False)
    cats = list(tokens.columns)
    if rank:
        rng = np.random.default_rng(42)
        order = rng.permutation(len(fit))
        groups = np.arange(len(order)) // 120
        data = Pool(numeric.iloc[order], labels.iloc[order], cat_features=cats, group_id=groups)
        model = CatBoostRanker(loss_function='YetiRank', **params).fit(data)
        raw = model.predict(Pool(hold, cat_features=cats))
        return reference_scale(raw, forecast(fit, labels, query))
    model = CatBoostClassifier(**params).fit(numeric, labels, cat_features=cats)
    return model.predict_proba(hold)[:, 1]


def xgboost_forecast(fit, labels, query):
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=350, max_depth=4, max_leaves=18,
        grow_policy='lossguide', learning_rate=.04, min_child_weight=14, reg_alpha=.5,
        reg_lambda=8, subsample=.8, colsample_bytree=.45, tree_method='hist', n_jobs=4, random_state=42)
    return model.fit(fit.to_numpy(), labels).predict_proba(query.to_numpy())[:, 1]


def encode_patterns(tokens, labels, query):
    columns = {}
    prior = labels.mean()
    for name in tokens:
        d = pd.DataFrame({'key': tokens[name], 'target': labels}, index=tokens.index)
        totals = d.groupby('key').target.agg(['sum', 'size'])
        rate = (totals['sum'] + 25 * prior) / (totals['size'] + 25)
        columns['rate_' + name] = query[name].map(rate).fillna(prior)
    return pd.DataFrame(columns, index=query.index)


def out_of_fold_encodings(tokens, labels, query_tokens):
    pieces = []
    splits = StratifiedKFold(4, shuffle=True, random_state=23)
    for train_pos, valid_pos in splits.split(tokens, labels):
        pieces.append(encode_patterns(tokens.iloc[train_pos], labels.iloc[train_pos], tokens.iloc[valid_pos]))
    return pd.concat(pieces).reindex(tokens.index), encode_patterns(tokens, labels, query_tokens)


def calibrated_forecast(fit, labels, query, fit_group, query_group, fit_context, query_context):
    """Learn group-specific Platt slopes and a context-aware stack from inner OOF predictions."""
    splits = StratifiedKFold(3, shuffle=True, random_state=130)
    oof = np.empty(len(fit))
    for train_pos, valid_pos in splits.split(fit, labels):
        oof[valid_pos] = forecast(fit.iloc[train_pos], labels.iloc[train_pos], fit.iloc[valid_pos])
    scores = forecast(fit, labels, query)
    def design(p, g, ctx, context):
        raw = logit(np.clip(p, .0001, .9999))
        group = np.eye(5)[g]
        col = [raw[:, None], group, group * raw[:, None]]
        if context:
            col.append(np.log1p(ctx[['n_orders', 'recency']].to_numpy()))
        return np.column_stack(col)
    predictions = {}
    for name, context in [('group_platt', False), ('group_platt_context', True)]:
        model = LogisticRegression(C=.25, max_iter=600)
        with threadpool_limits(limits=4):
            model.fit(design(oof, fit_group, fit_context, context), labels)
            predictions[name] = model.predict_proba(design(scores, query_group, query_context, context))[:, 1]
    return predictions


def stack_components(train_parts, y, query_parts, train_core, query_core, use_context=True):
    """Logistic stack fitted on upstream predictions made without each customer's label."""
    from research.window_models import regions
    def design(parts, core):
        raw = logit(np.clip(parts[['transitions', 'full', 'regional']].to_numpy(), .0001, .9999))
        if not use_context:
            return raw
        group = np.eye(5)[regions(core)]
        level = raw.mean(axis=1, keepdims=True)
        ctx = np.log1p(core[['n_orders', 'recency']].to_numpy())
        return np.column_stack([raw, group, group * level, ctx])
    model = LogisticRegression(C=.3, max_iter=1000)
    with threadpool_limits(limits=4):
        model.fit(design(train_parts, train_core), y)
        return model.predict_proba(design(query_parts, query_core))[:, 1]
