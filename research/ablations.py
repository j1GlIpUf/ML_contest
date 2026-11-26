"""Additional feature groups evaluated during the second round of Lamoda experiments.

Use with orders, train, receipt_table, transition_train, fit_model and ap20 from
lamoda_groups.ipynb. Each feature function returns one row per customer ID.
"""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def portfolio_drift(orders, cohort):
    h = orders.merge(
        cohort[["customer_id", "scoring_date"]],
        on="customer_id",
        validate="many_to_one",
    )
    h = h[h.order_date < h.scoring_date].copy()
    h["age"] = (h.scoring_date - h.order_date).dt.days
    features = []
    for field in ["category", "category2", "gender_product"]:
        for width in [90, 180]:
            # Compare two consecutive windows of equal length ending at the blind period.
            pair = h[h.age.between(14, 13 + 2 * width)].copy()
            pair["recent"] = pair.age < 14 + width
            a = (
                pair.groupby(["customer_id", field, "recent"])
                .agg(spend=("paid_price", "sum"), orders=("order_id", "nunique"))
                .unstack(fill_value=0)
            )
            for measure in ["spend", "orders"]:
                value = a[measure]
                prior = value[False]
                recent = value[True]
                old_share = prior / prior.groupby(level=0).transform("sum").replace(
                    0, np.nan
                )
                new_share = recent / recent.groupby(level=0).transform("sum").replace(
                    0, np.nan
                )
                diff = new_share.fillna(0) - old_share.fillna(0)
                stats = pd.DataFrame(
                    {
                        "gain": diff,
                        "loss": -diff,
                        "turnover": diff.abs() / 2,
                        "new_mass": new_share.where(prior.eq(0), 0),
                        "dropped_mass": old_share.where(recent.eq(0), 0),
                        "overlap": np.sqrt((old_share * new_share).fillna(0)),
                    }
                )
                stats = stats.groupby(level=0).agg(
                    {
                        "gain": "max",
                        "loss": "max",
                        "turnover": "sum",
                        "new_mass": "sum",
                        "dropped_mass": "sum",
                        "overlap": "sum",
                    }
                )
                features.append(stats.add_prefix(f"drift_{field}_{width}_{measure}_"))
    return pd.concat(features, axis=1)


def waiting_time_features(receipts):
    b = receipts.copy()
    g = b.groupby("customer_id")
    # A historical receipt contributes a waiting time only when its successor is observed.
    b["wait"] = b.age - g.age.shift(-1)
    b["before"] = g.category.shift(1)
    end = b.groupby("customer_id").tail(1).set_index("customer_id")
    b["elapsed"] = b.customer_id.map(end.age)
    b["due"] = b.wait - b.elapsed
    same = b.category.eq(b.customer_id.map(end.category))
    previous = b.before.eq(b.customer_id.map(end.before))
    size_diff = (np.log1p(b.amount) - np.log1p(b.customer_id.map(end.amount))).abs()
    known = b.wait.notna()
    weights = {
        "all": np.ones(len(b)),
        "last_category": same,
        "last_pair": same & previous,
        "similar_value": np.exp(-size_diff),
        "recent": np.exp(-(b.age - b.elapsed) / 120),
        "similar_basket": np.exp(-size_diff) * (0.2 + 0.8 * same),
    }
    parts = []
    for name, weight in weights.items():
        w = np.asarray(weight) * known
        values = pd.DataFrame(
            {
                "mass": w,
                "survived": w * (b.due >= 0),
                "quarter": w * b.due.between(0, 90),
                "central": w * b.due.between(15, 75),
                "kernel": w * np.exp(-0.5 * ((b.due - 45) / 25) ** 2).fillna(0),
                "overdue": w * (b.due < -14),
            }
        )
        a = values.groupby(b.customer_id).sum()
        a["hazard"] = (a.quarter + 0.5) / (a.survived + 1)
        a["window_share"] = (a.central + 0.25) / (a.mass + 1)
        a["window_density"] = a.kernel / (a.mass + 1)
        parts.append(a.add_prefix("waiting_" + name + "_"))
    quant = g.wait.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    projected = quant.sub(end.age, axis=0)
    parts.append(projected.add_prefix("waiting_due_q"))
    parts.append((projected - 45).abs().add_prefix("waiting_center_q"))
    return pd.concat(parts, axis=1)


def multiscale_clocks(b):
    b = b.copy()
    g = b.groupby("customer_id")
    b["wait"] = -g.age.diff()
    b["pos"] = g.cumcount(ascending=False)
    rec = g.age.last()
    parts = []
    for name, mask in [
        (f"d{d}", b.age <= d + 14) for d in [30, 60, 90, 120, 150, 180, 270, 365]
    ] + [(f"n{n}", b.pos < n) for n in [2, 3, 4, 6, 9, 12]]:
        z = b[mask]
        a = z.groupby("customer_id").agg(
            span=("age", "max"),
            n=("age", "size"),
            median=("wait", "median"),
            mean=("wait", "mean"),
            min_gap=("wait", "min"),
            max_gap=("wait", "max"),
        )
        a["due"] = a["median"] - rec
        a["centrality"] = (a.due - 45).abs()
        a["frequency_change"] = a["median"] / g.wait.median().replace(0, np.nan)
        a["elapsed_ratio"] = rec / a["median"].replace(0, np.nan)
        parts.append(a.add_prefix(f"scale_{name}_"))
    return pd.concat(parts, axis=1)


def pooled_intervals(b):
    b = b.copy()
    g = b.groupby("customer_id")
    b["prev"] = g.category.shift()
    b["plain"] = -g.age.diff()
    b["i"] = g.cumcount()
    b["run"] = (b.category.ne(b.prev)).groupby(b.customer_id).cumsum()
    by_cat = b.groupby(["customer_id", "category"])
    b["return"] = -by_cat.age.diff()
    b["steps"] = by_cat.i.diff()
    b["return_per_order"] = b["return"] / b.steps
    b["change"] = b.plain.where(b.category.ne(b.prev))
    b["stay"] = b.plain.where(b.category.eq(b.prev))
    b["transition_return"] = -b.groupby(["customer_id", "prev", "category"]).age.diff()
    b["first_in_run"] = b.age.eq(
        b.groupby(["customer_id", "run"]).age.transform("first")
    )
    run_start = b[b.first_in_run].copy()
    b["return_after_switch"] = -run_start.groupby(
        ["customer_id", "category"]
    ).age.diff()
    rec = g.age.last()
    last_cat = g.category.last()
    results = []
    for sample_name, sample in [
        ("all", b),
        ("current", b[b.category.eq(b.customer_id.map(last_cat))]),
        ("others", b[~b.category.eq(b.customer_id.map(last_cat))]),
    ]:
        grouped = sample.groupby("customer_id")
        for interval in [
            "return",
            "return_after_switch",
            "return_per_order",
            "change",
            "stay",
            "transition_return",
        ]:
            a = grouped[interval].agg(["median", "mean", "last", "count"])
            for col in ["median", "mean", "last"]:
                a[col] = a[col] - rec
            a["center"] = (a["median"] - 45).abs()
            results.append(a.add_prefix(f"pool_{sample_name}_{interval}_"))
    return pd.concat(results, axis=1)


def compare_group(
    group, orders, train, transition_train, receipt_table, fit_model, ap20
):
    """Evaluate one additional group with the same classifier on both cutoffs."""
    if group == "portfolio":
        features = portfolio_drift(orders, train)
    else:
        builders = {
            "waiting_times": waiting_time_features,
            "intervals": pooled_intervals,
            "scales": multiscale_clocks,
        }
        features = builders[group](receipt_table(orders, train))
    aligned = features.reindex(train.customer_id).set_axis(train.index)
    frame = transition_train.join(aligned)
    rows = []
    for cutoff in sorted(train.scoring_date.unique())[1:]:
        past = train.scoring_date < cutoff
        held = train.scoring_date == cutoff
        model = fit_model(frame.loc[past], train.loc[past, "target"])
        with threadpool_limits(limits=4):
            scores = model.predict_proba(frame.loc[held])[:, 1]
        rows.append(
            {
                "date": cutoff,
                "group": group,
                "ap20": ap20(train.loc[held, "target"], scores),
            }
        )
    return pd.DataFrame(rows)
