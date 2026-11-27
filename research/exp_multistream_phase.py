"""Evaluate multi-label purchase clocks and trajectory replay on February and March.

From the repository root run:
    python -m research.exp_multistream_phase --data /path/to/data_or_zip
"""
import argparse
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, ndtr
from threadpoolctl import threadpool_limits

from research.rerank import compare

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("category", "gender_product", "category2")
BLIND_START = -14
HORIZON = 90
CATEGORY2_LIMIT = 40
MODELS = ("quarters", "current_base", "base_plus_phase", "base_plus_trajectory",
          "base_plus_multistream")
METRIC_COLUMNS = [
    "month", "model", "status", "ap20", "delta_vs_quarters", "delta_vs_current", "buyers_top5000",
    "false_top5000_4_6", "false_top5000_7_15", "false_top5000_4_15",
    "missed_buyers_4_15", "added_buyers_4_15", "dropped_buyers_4_15", "source",
]


def read_table(source, name, **kwargs):
    reader = pd.read_csv if name.endswith(".csv") else pd.read_parquet
    if source.is_dir():
        return reader(source / name, **kwargs)
    with ZipFile(source) as archive:
        members = [member for member in archive.namelist() if Path(member).name == name]
        if len(members) != 1:
            raise ValueError(f"Expected one {name} in {source}")
        with archive.open(members[0]) as member:
            return reader(member, **kwargs)


def load_data(source):
    train = read_table(source, "train.csv", parse_dates=["scoring_date"]).set_index("customer_id")
    if not train.index.is_unique or not train.target.isin([0, 1]).all():
        raise ValueError("Expected unique training customers with binary labels")
    cutoffs = train.scoring_date.drop_duplicates().sort_values()
    if cutoffs.dt.month.tolist() != [1, 2, 3] or cutoffs.dt.year.nunique() != 1:
        raise ValueError("Expected one January, one February, and one March cutoff from the same year")
    orders = read_table(source, "orders.parquet",
                        columns=["customer_id", "order_id", "order_date", "product_id"])
    orders["order_date"] = pd.to_datetime(orders.order_date)
    orders = orders.join(train.scoring_date, on="customer_id", how="inner", validate="many_to_one")
    orders = orders.loc[orders.order_date.lt(orders.scoring_date)].copy()
    orders["age"] = (orders.scoring_date.dt.normalize() - orders.order_date.dt.normalize()).dt.days
    products = read_table(source, "products.parquet", columns=["product_id", *FAMILIES])
    orders = orders.drop(columns="order_date").merge(
        products, on="product_id", how="left", validate="many_to_one"
    )
    return train, orders


def fit_period(ages):
    """Fit a circular mean on a personal grid of gaps and missed-cycle divisors."""
    span = ages[-1] - ages[0]
    intervals = [span]
    for lag in range(1, min(len(ages), 5)):
        gaps = np.unique(ages[lag:] - ages[:-lag])
        if len(gaps) > 12:
            gaps = gaps[np.linspace(0, len(gaps) - 1, 12, dtype=int)]
        intervals.extend(gaps)
    periods = (np.asarray(intervals)[:, None] / np.arange(1, 5)).ravel()
    periods = np.unique(np.round(np.multiply.outer(periods, [.95, 1, 1.05]).ravel() * 2) / 2)
    # Integer shop dates make a one-day harmonic fit every history perfectly.
    # Allow subweekly periods for streams observed over less than a week.
    periods = periods[(periods >= min(7, span)) & (periods <= span)]

    positions = (ages - ages[0]) / periods[:, None]
    circular_mean = np.exp(2j * np.pi * positions).mean(axis=1)
    coherence = np.abs(circular_mean).clip(0, 1)
    # Prefer the fundamental over harmonics unless shorter cycles fit better.
    adjacent_cycles = np.diff(ages) / periods[:, None]
    skipped = np.maximum(0, np.rint(adjacent_cycles) - 1).mean(axis=1)
    choice = np.argmax(coherence - .08 * np.log1p(skipped))
    period = periods[choice]
    strength = coherence[choice]

    # The circular-mean phase nearest the last purchase represents that purchase.
    # Advance from it to the first unobserved phase in or after the blind window.
    last_phase = -ages[0] - np.angle(circular_mean[choice]) * period / (2 * np.pi)
    cycles = max(1, int(np.ceil((BLIND_START - last_phase) / period - 1e-10)))
    due = last_phase + cycles * period
    if abs(due - round(due)) < 1e-9:
        due = float(round(due))
    sigma = np.clip(period * np.sqrt(-2 * np.log(max(strength, 1e-12))) / (2 * np.pi), 2, 30)
    reliability = strength * (len(ages) - 1) / (len(ages) + 1) / np.sqrt(cycles)
    cdf = ndtr((np.array([BLIND_START, 0, HORIZON]) - due) / sigma)
    blind, target = reliability * np.diff(cdf)
    return period, strength, due, target, blind


def build_streams(lines):
    records = []
    for family in FAMILIES:
        keys = ["customer_id", family]
        dates = lines[[*keys, "age"]].dropna().drop_duplicates()
        dates = dates.loc[dates.duplicated(keys, keep=False)].sort_values("age")
        for (customer, label), group in dates.groupby(keys, sort=False, observed=True):
            records.append((customer, family, label, *fit_period(group.age.to_numpy())))
    return pd.DataFrame.from_records(
        records, columns=["customer_id", "family", "label", "period", "coherence", "due",
                          "p_target", "p_blind"]
    )


def aggregate_streams(streams, customers, frequent_category2):
    kept = streams.loc[streams.family.ne("category2") | streams.label.isin(frequent_category2)].copy()
    kept["target_due"] = kept.due.between(0, HORIZON)
    kept["blind_due"] = kept.due.ge(BLIND_START) & kept.due.lt(0)
    kept["target_coherence"] = kept.coherence.where(kept.target_due)
    features = pd.DataFrame(index=customers)
    for family in FAMILIES:
        group = kept.loc[kept.family.eq(family)].set_index("customer_id")
        complement = 1 - group[["p_target", "p_blind"]].astype(float)
        risk = (1 - complement.groupby(level=0).prod()).reindex(customers, fill_value=0)
        timing = group.groupby(level=0)[["target_due", "blind_due"]].sum()
        features = features.join(risk.add_prefix(f"phase_{family}_"))
        features = features.join(timing.reindex(customers, fill_value=0).add_prefix(f"phase_{family}_"))
    grouped = kept.groupby("customer_id")
    features["phase_max_target_coherence"] = grouped.target_coherence.max()
    features["phase_target_due_count"] = grouped.target_due.sum().reindex(customers, fill_value=0)
    features["phase_blind_due_count"] = grouped.blind_due.sum().reindex(customers, fill_value=0)
    return features.astype("float32")


def replay_trajectories(lines, customers):
    receipt_key = ["customer_id", "order_id", "age"]
    receipts = lines[receipt_key].drop_duplicates().copy()
    if receipts.duplicated(["customer_id", "order_id"]).any():
        raise ValueError("An order has multiple dates")
    pairs = lines[[*receipt_key, "category", "gender_product"]].dropna().drop_duplicates()
    # Each token is a pair from a single product line in the receipt.
    pairs["token"] = list(zip(pairs.category, pairs.gender_product))
    tokens = pairs.groupby(receipt_key, sort=False).token.agg(frozenset)
    receipts = receipts.join(tokens, on=receipt_key).sort_values(
        ["customer_id", "age", "order_id"], ascending=[True, False, True]
    )

    result = {}
    for customer, history in receipts.groupby("customer_id", sort=False):
        ages = history.age.to_numpy()
        labels = [token if isinstance(token, frozenset) else frozenset() for token in history.token]
        anchor = labels[-1]
        weights = np.zeros(len(ages) - 1)
        for i, past in enumerate(labels[:-1]):
            if ages[i] > ages[-1] and anchor and past:
                weights[i] = len(anchor & past) / len(anchor | past)
        matches = np.flatnonzero(weights > 0)
        stats = {"replay_matches": len(matches), "replay_best_jaccard": weights.max(initial=0)}
        similarity = weights[matches] ** 2
        for step in range(1, 5):
            has_follower = matches + step < len(ages)
            observed = matches[has_follower]
            weight = similarity[has_follower]
            due = ages[observed] - ages[-1] - ages[observed + step]
            prefix = f"replay_step{step}_"
            stats[prefix + "support"] = weight.sum()
            if len(observed):
                stats[prefix + "target"] = np.average((due >= 0) & (due <= HORIZON), weights=weight)
                stats[prefix + "blind"] = np.average((due >= BLIND_START) & (due < 0), weights=weight)
                stats[prefix + "due"] = np.average(due, weights=weight)
        result[customer] = stats
    columns = ["replay_matches", "replay_best_jaccard"]
    columns += [f"replay_step{k}_{stat}" for k in range(1, 5)
                for stat in ("support", "target", "blind", "due")]
    frame = pd.DataFrame.from_dict(result, orient="index", columns=columns).reindex(customers).astype("float32")
    zeros = [name for name in columns if name.endswith("_support") or name in columns[:2]]
    frame[zeros] = frame[zeros].fillna(0)
    return frame


def archive_scores(train, archive):
    january = pd.concat([pd.read_parquet(archive / f"january_fold_{i}.parquet").drop(columns="fold")
                         for i in range(5)])
    dated = {1: january, **{month: pd.read_parquet(archive / f"scores_{month}.parquet")
                            for month in (2, 3)}}
    for month, scores in dated.items():
        ids = train.index[train.scoring_date.dt.month.eq(month)]
        if len(scores) != len(ids) or not scores.index.is_unique or not ids.isin(scores.index).all():
            raise ValueError(f"Month {month} does not match the archived baseline customers")
    base = pd.concat(dated.values()).reindex(train.index)
    if not np.isfinite(base.to_numpy()).all():
        raise ValueError("The archived baseline has missing or nonfinite scores")
    return base


def residual(fit, query, labels, quarters):
    model = LGBMClassifier(n_estimators=180, learning_rate=.035, num_leaves=9,
                           min_child_samples=120, reg_lambda=12, deterministic=True,
                           force_col_wise=True, verbosity=-1, random_state=42, n_jobs=4)
    offset = logit(quarters.clip(1e-5, 1 - 1e-5))
    model.fit(fit, labels, init_score=offset)
    return model.booster_.predict(query, raw_score=True, num_threads=4)


def evaluate(month, labels, scores, reference, counts):
    ids = labels.index
    ranked = scores.sort_values(ascending=False, kind="stable").index
    k = int(np.ceil(.2 * len(ids)))
    top = ranked[:k]
    baseline_top = reference.sort_values(ascending=False, kind="stable").index[:k]
    medium = counts.between(4, 15)
    positives = labels.eq(1)
    false = counts.loc[top[~positives.loc[top]]]
    added = top.difference(baseline_top)
    dropped = baseline_top.difference(top)
    metrics = compare(labels, scores, reference, counts)
    return dict(
        month=month, status="measured", source="time_split", ap20=metrics["ap20"],
        buyers_top5000=metrics["buyers"],
        false_top5000_4_6=int(false.between(4, 6).sum()),
        false_top5000_7_15=int(false.between(7, 15).sum()),
        false_top5000_4_15=int(false.between(4, 15).sum()),
        missed_buyers_4_15=int((positives & medium & ~ids.isin(top)).sum()),
        added_buyers_4_15=int((positives & medium).loc[added].sum()),
        dropped_buyers_4_15=int((positives & medium).loc[dropped].sum()),
    )


def run(source, archive, output):
    train, orders = load_data(source)
    base = archive_scores(train, archive)
    counts = orders.groupby("customer_id").order_id.nunique().reindex(train.index, fill_value=0)
    category2_receipts = orders[["scoring_date", "customer_id", "order_id", "category2"]].dropna().drop_duplicates()
    frequency = category2_receipts.groupby(["scoring_date", "category2"], observed=True).size()
    # Period fitting is independent for each customer. Only the fold's frequent
    # category2 vocabulary and the supervised correction depend on training rows.
    streams = {}
    replays = []
    for cutoff, customers in train.groupby("scoring_date"):
        lines = orders.loc[orders.scoring_date.eq(cutoff)]
        streams[cutoff] = build_streams(lines)
        replays.append(replay_trajectories(lines, customers.index))
    trajectory = pd.concat(replays).reindex(train.index)
    rows = []
    for month in (2, 3):
        query = train.loc[train.scoring_date.dt.month.eq(month)]
        cutoff = query.scoring_date.min()
        fit = train.loc[train.scoring_date.lt(cutoff)]
        past = frequency.loc[frequency.index.get_level_values("scoring_date") < cutoff]
        frequent = past.groupby(level="category2", observed=True).sum().sort_index().nlargest(CATEGORY2_LIMIT).index
        phase = pd.concat([aggregate_streams(streams[date], cohort.index, frequent)
                           for date, cohort in train.loc[train.scoring_date.le(cutoff)].groupby("scoring_date")])
        current = pd.read_parquet(archive / f"tail_predictions_{month}.parquet",
                                  columns=["lightgbm_plain_half"]).lightgbm_plain_half.reindex(query.index)
        if not np.isfinite(current.to_numpy()).all():
            raise ValueError(f"Month {month} has missing current baseline forecasts")
        quarters = base.quarters.loc[query.index]
        predictions = dict(quarters=quarters, current_base=current)
        common = residual(base.loc[fit.index], base.loc[query.index], fit.target, base.quarters.loc[fit.index])
        baseline_logit = logit(current.clip(1e-5, 1 - 1e-5))
        for name, block in (("phase", phase), ("trajectory", trajectory),
                            ("multistream", phase.join(trajectory))):
            features = base.join(block)
            correction = residual(features.loc[fit.index], features.loc[query.index],
                                  fit.target, base.quarters.loc[fit.index])
            # The latest baseline has no January OOF. Transfer the incremental
            # logit correction learned relative to the matched scores-only model.
            predictions[f"base_plus_{name}"] = baseline_logit + (correction - common)
        metrics = {name: evaluate(month, query.target, predictions[name], current, counts.loc[query.index])
                   for name in MODELS}
        for name, record in metrics.items():
            rows.append(dict(model=name, delta_vs_quarters=record["ap20"] - metrics["quarters"]["ap20"],
                             delta_vs_current=record["ap20"] - metrics["current_base"]["ap20"], **record))
        result = pd.DataFrame(rows, columns=METRIC_COLUMNS)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        result.to_csv(temporary, index=False)
        temporary.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = ROOT / "data"
    if not default.is_dir():
        default = ROOT / "lamoda-hse-repeat-purchase-2026.zip"
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("LAMODA_DATA", default)))
    parser.add_argument("--archive", type=Path, default=ROOT / "research/oof")
    parser.add_argument("--output", type=Path, default=ROOT / "research/results/exp_multistream_phase.csv")
    args = parser.parse_args()
    if not args.data.exists():
        parser.error(f"Competition data missing at {args.data}. Set --data or LAMODA_DATA.")
    with threadpool_limits(limits=4):
        results = run(args.data, args.archive, args.output)
    print(results.pivot(index="model", columns="month", values="ap20").to_string(float_format="%.5f"))


if __name__ == "__main__":
    main()
