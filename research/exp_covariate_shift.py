"""Reweight the original tail LightGBM by cross-fitted month density ratios."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from research.category_phase import category_phase
from research.motif_model import build_motifs
from research.personal_backtest import personal_backtest
from research.rerank import compare
from research.sequence_rules import baskets
from research.tail_features import warning_features
from research.tail_models import booster
from research.trajectory import trajectory_features


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
RESULTS = RESEARCH / "results"
THRESHOLD = .65
WEIGHT_MIN, WEIGHT_MAX = .25, 4.
DOMAIN_FEATURES = (
    "n_orders", "n_days", "n_items", "n_brands", "n_categories", "unique_sku", "subcats",
    "recency", "age_days", "gap_mean", "gap_std", "gap_last", "interval_median",
    "orders_30", "orders_90", "orders_180", "orders_365", "items_90",
    "spend_90", "spend_365", "mean_basket", "mean_price", "mean_discount", "mean_loyalty",
    "basket_last_amount", "basket_last_units", "trend_45", "trend_90",
)
GROUP_FUNCTIONS = {
    "features", "rich_features", "rhythm_features", "cycle_features", "sequence_features",
    "category_clocks", "transitions", "basket_patterns", "forecast_windows", "assemble",
    "receipt_table", "transition_clocks", "basket_graph", "personal_sessions", "extend_features",
}


def group_feature_functions(orders):
    """Reuse the original feature definitions without running notebook fits or writes."""
    path = ROOT / "lamoda_groups.ipynb"
    definitions = {}
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        syntax = ast.parse("".join(cell["source"]))
        definitions.update((node.name, node) for node in syntax.body
                           if isinstance(node, ast.FunctionDef) and node.name in GROUP_FUNCTIONS)
    if definitions.keys() != GROUP_FUNCTIONS:
        raise ValueError("The original notebook feature definitions changed.")
    scope = {"orders": orders, "pd": pd, "np": np}
    module = ast.Module(body=list(definitions.values()), type_ignores=[])
    exec(compile(module, str(path), "exec"), scope)
    return scope


def load_cohorts(data):
    return pd.read_csv(data / "train.csv", usecols=["customer_id", "scoring_date"],
                       parse_dates=["scoring_date"])


def selected_features():
    return {month: pd.read_csv(RESULTS / f"tail_selected_{month}.csv").feature.tolist()
            for month in (2, 3)}


def build_features(data, cohorts, chosen):
    orders = pd.read_parquet(data / "orders.parquet")
    orders["order_date"] = pd.to_datetime(orders.order_date)
    products = pd.read_parquet(data / "products.parquet",
                               columns=["product_id", "brand", "category", "category2", "gender_product"])
    orders = orders.merge(products, on="product_id", how="left", validate="many_to_one")
    functions = group_feature_functions(orders)
    base = functions["features"](cohorts)
    history = functions["rich_features"](cohorts)
    for name in ("rhythm_features", "cycle_features", "sequence_features"):
        history = history.join(functions[name](cohorts))
    clocks = functions["category_clocks"](cohorts)
    enriched = functions["assemble"](base, history, clocks, cohorts)
    _, core = functions["extend_features"](enriched, cohorts)
    core = core.set_axis(cohorts.customer_id)
    del base, history, clocks, enriched

    # Keep the original float32 conversion and use the selection saved by each prior-only fit.
    required = list(dict.fromkeys([*chosen[2], *chosen[3], *DOMAIN_FEATURES]))
    blocks = []
    for cutoff, customers in cohorts.groupby("scoring_date"):
        frame = core.loc[customers.customer_id].astype("float32")
        receipts = baskets(orders, customers)
        horizon = (cutoff + pd.DateOffset(months=3) - cutoff).days
        motifs = build_motifs(orders, customers).filter(regex="^motif_mode_")
        frame = frame.join(motifs).join(trajectory_features(receipts, horizon))
        frame = frame.join(personal_backtest(receipts)).join(category_phase(receipts))
        frame = frame.join(warning_features(orders, customers, frame))
        blocks.append(frame.loc[:, required].astype("float32"))
    return pd.concat(blocks).loc[cohorts.customer_id]


def load_labels(data, row_by_id, ids):
    # Cohort row locations were read without target; only parse labels for requested rows.
    lines = set(row_by_id.loc[ids].to_numpy() + 1)
    labels = pd.read_csv(data / "train.csv", usecols=["customer_id", "target"],
                         skiprows=lambda row: row != 0 and row not in lines)
    labels = labels.set_index("customer_id", verify_integrity=True).target.reindex(ids)
    if not labels.isin([0, 1]).all():
        raise ValueError("A requested customer lacks a binary target.")
    return labels.astype(int)


def density_weights(features, fit, query):
    """Cross-fit domain probabilities and correct odds for each fold's class prior."""
    x = features.loc[fit.append(query), list(DOMAIN_FEATURES)]
    domain = np.r_[np.zeros(len(fit), dtype=int), np.ones(len(query), dtype=int)]
    probabilities = np.empty(len(domain))
    ratios = np.empty(len(domain))
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=26)
    for train, held in splits.split(x, domain):
        model = LGBMClassifier(
            n_estimators=120, learning_rate=.04, num_leaves=12, min_child_samples=60,
            reg_lambda=10, random_state=26, n_jobs=4, verbosity=-1,
        )
        model.fit(x.iloc[train], domain[train])
        p = model.predict_proba(x.iloc[held])[:, 1].clip(1e-6, 1 - 1e-6)
        probabilities[held] = p
        n_query = domain[train].sum()
        ratios[held] = p / (1 - p) * (len(train) - n_query) / n_query
    raw = ratios[:len(fit)]
    clipped = raw.clip(WEIGHT_MIN, WEIGHT_MAX)
    weights = pd.Series(clipped / clipped.mean(), index=fit)
    diagnostics = {
        "domain_auc": roc_auc_score(domain, probabilities),
        "fraction_clipped_low": (raw < WEIGHT_MIN).mean(),
        "fraction_clipped_high": (raw > WEIGHT_MAX).mean(),
        "weight_min": weights.min(), "weight_max": weights.max(),
    }
    return weights, diagnostics


def effective_size(weights):
    return weights.sum() ** 2 / weights.pow(2).sum()


def validation_predictions(features, chosen, y_fit, quarter, weights):
    x = features.loc[y_fit.index, chosen]
    held = features.loc[quarter.index, chosen]
    predictions = {}
    for name, sample_weight in (("tail_unweighted", None), ("tail_reweighted", weights)):
        model = booster("lightgbm").fit(x, y_fit, sample_weight=sample_weight)
        predictions[name] = (quarter + model.predict_proba(held)[:, 1]) / 2
    return predictions


def tail_group_metrics(y, prediction, ids):
    y, p = y.loc[ids], prediction.loc[ids]
    return {
        "n": len(ids), "buyers": int(y.sum()), "mean_score": p.mean(), "buyer_rate": y.mean(),
        "score_minus_rate": p.mean() - y.mean(), "brier": (p - y).pow(2).mean(),
    }


def run(data, cohorts, features, chosen):
    row_by_id = pd.Series(cohorts.index, index=cohorts.customer_id)
    dates = cohorts.set_index("customer_id", verify_integrity=True).scoring_date
    n_orders = features.n_orders
    small_history = n_orders.between(4, 6)
    quarterly = {
        month: pd.read_parquet(RESEARCH / "oof" / f"scores_{month}.parquet", columns=["quarters"]).quarters
        for month in (2, 3)
    }
    fixed_tail = {
        month: scores.index[scores.gt(THRESHOLD) & small_history.reindex(scores.index)]
        for month, scores in quarterly.items()
    }
    # This second domain comparison isolates February versus March in the same reference segment.
    transported_weights, transport_diagnostics = density_weights(features, fixed_tail[2], fixed_tail[3])
    rows, evaluated = [], {}
    for month in (2, 3):
        cutoffs = dates.loc[quarterly[month].index].unique()
        if len(cutoffs) != 1 or cutoffs[0].month != month:
            raise ValueError(f"Saved forecasts do not match the scoring month {month}.")
        cutoff = cutoffs[0]
        fit, query = dates.index[dates.lt(cutoff)], dates.index[dates.eq(cutoff)]
        quarter = quarterly[month].reindex(query)
        if quarter.isna().any():
            raise ValueError(f"Saved forecasts have missing customers in month {month}.")
        weights, diagnostics = density_weights(features, fit, query)
        y_fit = load_labels(data, row_by_id, fit)
        predictions = validation_predictions(features, chosen[month], y_fit, quarter, weights)

        archived = pd.read_parquet(RESEARCH / "oof" / f"tail_predictions_{month}.parquet",
                                   columns=["lightgbm_plain_half"]).lightgbm_plain_half.loc[query]
        max_error = (predictions["tail_unweighted"] - archived).abs().max()
        if not max_error <= 1e-12:
            raise ValueError(f"Original LightGBM predictions differ from the saved control in month {month}: {max_error:.3g}")

        y = load_labels(data, row_by_id, query)
        evaluated[month] = y
        control = compare(y, quarter, quarter, n_orders)["ap20"]
        tail_ap = compare(y, archived, quarter, n_orders)["ap20"]
        for name, prediction in {"quarters": quarter, **predictions}.items():
            metrics = compare(y, prediction, quarter, n_orders)
            top = prediction.nlargest(5000, keep="first").index
            group_top = top[small_history.loc[top]]
            rows.append(dict(
                section="ranking", month=month, model=name, ap20=metrics["ap20"],
                delta_vs_quarters=metrics["ap20"] - control, delta_vs_tail=metrics["ap20"] - tail_ap,
                buyers_top5000=metrics["buyers"], n_4_6_top5000=len(group_top),
                buyers_4_6_top5000=int(y.loc[group_top].sum()),
                reconstruction_max_error=max_error if name == "tail_unweighted" else None,
            ))
            groups = {"fixed_quarters_gt_065": fixed_tail[month],
                      "own_score_gt_065": query[small_history.loc[query] & prediction.gt(THRESHOLD)]}
            for scope, ids in groups.items():
                rows.append(dict(section="4_6_orders", month=month, model=name, scope=scope,
                                 **tail_group_metrics(y, prediction, ids)))
        weights_4_6 = weights.loc[small_history.loc[fit]]
        rows.append(dict(section="weights", month=month, model="tail_reweighted", n=len(fit),
                         ess=effective_size(weights), ess_fraction=effective_size(weights) / len(fit),
                         n_4_6=len(weights_4_6), ess_4_6=effective_size(weights_4_6), **diagnostics))

    feb, mar = fixed_tail[2], fixed_tail[3]
    for month, name, ids, weights in (
        (2, "feb_observed", feb, np.ones(len(feb))),
        (2, "feb_transported_to_mar", feb, transported_weights.to_numpy()),
        (3, "mar_observed", mar, np.ones(len(mar))),
    ):
        rows.append(dict(section="4_6_composition", month=month, model=name, scope="fixed_quarters_gt_065",
                         n=len(ids), ess=weights.sum() ** 2 / (weights @ weights),
                         mean_score=np.average(quarterly[month].loc[ids], weights=weights),
                         buyer_rate=np.average(evaluated[month].loc[ids], weights=weights),
                         **(transport_diagnostics if name == "feb_transported_to_mar" else {})))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT,
                        help="directory containing train.csv, orders.parquet, and products.parquet")
    args = parser.parse_args()
    cohorts = load_cohorts(args.data)
    chosen = selected_features()
    with threadpool_limits(limits=4):
        features = build_features(args.data, cohorts, chosen)
        results = run(args.data, cohorts, features, chosen)
    results.to_csv(RESULTS / "covariate_shift.csv", index=False)
    summary = results.loc[results.section.eq("ranking"), ["month", "model", "ap20", "delta_vs_tail", "buyers_top5000"]]
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))


if __name__ == "__main__":
    main()
