"""Compare three ways to improve the top of the saved quarters ranking."""

import argparse
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from research.rerank import compare
from research.tail_analysis import MODELS, calibrate, splice


RESEARCH = Path(__file__).resolve().parent
RESULT = RESEARCH / "results" / "exp_tophead.csv"
TRAIN_POOLS = (35, 40, 45)
QUERY_POOL = 40
ALPHA, GAMMA, NEGATIVE_MULTIPLIER = 3, 2, 2
TAIL_THRESHOLD = 0.65
ORDER_BANDS = ("2-3", "4-6", "7-15", "16+")
CATEGORY_BANDS = ("mono", "multi")


def read_table(data, name, reader, **kwargs):
    if data.is_dir():
        return reader(data / name, **kwargs)
    with ZipFile(data) as archive, archive.open(name) as source:
        return reader(source, **kwargs)


def saved_scores(cohorts):
    files = RESEARCH / "oof"
    january = []
    for fold_id in range(5):
        fold = pd.read_parquet(files / f"january_fold_{fold_id}.parquet")
        if len(fold) != 5000 or not fold["fold"].eq(fold_id).all():
            raise ValueError(f"Invalid January OOF fold {fold_id}.")
        january.append(fold[MODELS])

    frames = [pd.concat(january)]
    frames.extend(pd.read_parquet(files / f"scores_{month}.parquet")[MODELS] for month in (2, 3))
    for month, frame in enumerate(frames, start=1):
        expected = cohorts.index[cohorts.scoring_date.dt.month.eq(month)]
        if not frame.index.is_unique or len(frame) != len(expected) or not expected.isin(frame.index).all():
            raise ValueError(f"Saved predictions do not match the customer cohort for month {month}.")
        if not ((frame >= 0) & (frame <= 1)).all().all():
            raise ValueError(f"Invalid probabilities for month {month}.")
    return pd.concat(frames).loc[cohorts.index]


def history_features(data, dates):
    orders = read_table(data, "orders.parquet", pd.read_parquet,
                        columns=["customer_id", "order_id", "product_id", "order_date"])
    orders["order_date"] = pd.to_datetime(orders.order_date)
    orders = orders.join(dates, on="customer_id", how="inner", validate="many_to_one")
    before_cutoff = orders.order_date.lt(orders.scoring_date)
    history = orders.loc[before_cutoff, ["customer_id", "order_id", "product_id", "order_date"]].copy()
    del orders
    products = read_table(data, "products.parquet", pd.read_parquet, columns=["product_id", "category"])
    history = history.merge(products, on="product_id", how="left", validate="many_to_one")
    features = history.groupby("customer_id").agg(
        n_orders=("order_id", "nunique"), n_days=("order_date", "nunique"),
        n_categories=("category", "nunique"), last_order=("order_date", "max"),
    ).reindex(dates.index)
    features["recency_days"] = (dates - features.pop("last_order")).dt.days
    for column in ("n_orders", "n_days", "n_categories"):
        features[column] = features[column].fillna(0).astype(int)
    return features


def pool_mask(scores, dates, percent):
    groups = scores.groupby(dates)
    positions = groups.rank(method="first", ascending=False)
    return positions.le(np.ceil(groups.transform("size") * percent / 100))


def classifier(fit, labels, query, weights=None):
    model = LGBMClassifier(
        objective="binary", n_estimators=160, learning_rate=0.04, num_leaves=9,
        min_child_samples=100, reg_lambda=10, deterministic=True, force_col_wise=True,
        random_state=42, n_jobs=4, verbosity=-1,
    )
    model.fit(fit, labels, sample_weight=weights)
    return pd.Series(model.predict_proba(query)[:, 1], index=query.index)


def ranking_predictions(features, labels, dates, prior, fit, query):
    past, reference = prior.loc[fit], prior.loc[query]
    candidates = reference.index[pool_mask(reference, dates.loc[query], QUERY_POOL)]
    candidate_features = features.loc[candidates]
    predictions = {"control": reference}
    for percent in TRAIN_POOLS:
        selected = past.index[pool_mask(past, dates.loc[fit], percent)]
        raw = classifier(features.loc[selected], labels.loc[selected], candidate_features)
        predictions[f"stage2_top{percent}"] = splice(reference, raw, candidates)

    fit_labels = labels.loc[fit]
    weights = 1 + ALPHA * past.pow(GAMMA)
    weights.loc[past.gt(TAIL_THRESHOLD) & fit_labels.eq(0)] *= NEGATIVE_MULTIPLIER
    for name, sample_weight in (("logloss_unweighted", None), ("logloss_weighted", weights)):
        raw = classifier(features.loc[fit], fit_labels, candidate_features, sample_weight)
        predictions[name] = splice(reference, raw, candidates)
    return predictions


def logloss(y, p):
    p = p.clip(1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()


def tail_calibration(labels, prior, history, fit, query):
    length = pd.cut(history.n_orders, [1, 3, 6, 15, np.inf], labels=ORDER_BANDS)
    categories = history.n_categories.map({1: "mono"}).mask(history.n_categories.ge(2), "multi")
    group = length.astype("string") + "/" + categories.astype("string")
    past, reference = prior.loc[fit], prior.loc[query]
    corrected = reference.copy()
    eligible = reference.gt(TAIL_THRESHOLD) & group.loc[query].notna()
    held = reference.index[eligible]
    if len(held):
        # The persisted February forecasts are also out of sample and are past for March.
        corrected.loc[held] = calibrate(past, labels.loc[fit], group.loc[fit].fillna(""),
                                         reference.loc[held], group.loc[held], method="shift")

    rows = []
    for band in ORDER_BANDS:
        for category in CATEGORY_BANDS:
            key = f"{band}/{category}"
            in_group = group.eq(key).fillna(False)
            ids = held[in_group.loc[held]]
            y, before, after = labels.loc[ids], reference.loc[ids], corrected.loc[ids]
            rows.append(dict(
                section="calibration", model="tail_calibration", order_band=band, category_band=category,
                fit_n=int((in_group.loc[fit] & past.gt(TAIL_THRESHOLD)).sum()),
                n=len(ids), buyers=int(y.sum()), rate=y.mean(), mean_before=before.mean(), mean_after=after.mean(),
                gap_before=y.mean() - before.mean(), gap_after=y.mean() - after.mean(),
                logloss_before=logloss(y, before), logloss_after=logloss(y, after),
            ))
    return corrected, rows


def run(data):
    cohorts = read_table(data, "train.csv", pd.read_csv,
                         usecols=["customer_id", "scoring_date", "target"], parse_dates=["scoring_date"])
    cohorts = cohorts.set_index("customer_id", verify_integrity=True)
    dates, labels = cohorts.scoring_date, cohorts.target
    if dates.isna().any() or sorted(dates.drop_duplicates().dt.month) != [1, 2, 3] or dates.dt.year.nunique() != 1:
        raise ValueError("Expected one cutoff per month from January through March of the same year.")
    if not labels.isin([0, 1]).all():
        raise ValueError("Expected binary targets for all cohorts.")

    scores = saved_scores(cohorts)
    prior = scores.quarters
    history = history_features(data, dates)
    # Use ranks within the entire scoring slice, including for the restricted training pools.
    ranks = scores.groupby(dates).rank(pct=True).rename(columns={
        name: "rank_pct" if name == "quarters" else f"rank_pct_{name}" for name in MODELS
    })
    features = history.join(ranks)
    rows = []
    for cutoff in sorted(dates.unique())[1:]:
        fit, query = dates.index[dates.lt(cutoff)], dates.index[dates.eq(cutoff)]
        reference, y = prior.loc[query], labels.loc[query]
        predictions = ranking_predictions(features, labels, dates, prior, fit, query)
        predictions["tail_calibration"], diagnostics = tail_calibration(labels, prior, history, fit, query)
        control = compare(y, reference, reference, history.n_orders)["ap20"]
        for name, prediction in predictions.items():
            metrics = compare(y, prediction, reference, history.n_orders)
            rows.append(dict(section="ranking", month=cutoff.month, model=name, ap20=metrics["ap20"],
                             delta_vs_control=metrics["ap20"] - control, buyers_top5000=metrics["buyers"]))
        rows.extend(dict(row, month=cutoff.month) for row in diagnostics)
    results = pd.DataFrame(rows)
    results.insert(3, "status", "measured")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("LAMODA_DATA", "data")),
                        help="directory or Lamoda ZIP containing train.csv, orders.parquet and products.parquet")
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    if not args.data.exists():
        parser.error(f"Lamoda inputs missing at {args.data}. Supply their directory or original ZIP via --data.")
    results = run(args.data)
    results.to_csv(args.output, index=False)
    summary = results.loc[results.section.eq("ranking")].pivot(
        index="model", columns="month", values=["ap20", "delta_vs_control", "buyers_top5000"],
    )
    print(summary.to_string(float_format=lambda value: f"{value:.5f}"))


if __name__ == "__main__":
    main()
