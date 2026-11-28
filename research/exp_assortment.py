"""Compare brand sales trends and personal assortment changes on temporal holdouts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.exp_covariate_shift import (
    RESEARCH, RESULTS, ROOT, build_features, load_cohorts, load_labels, selected_features,
)
from research.paired_validation import compare_predictions
from research.rerank import compare
from research.tail_models import booster


CUSTOMER = "customer_id"
SKU = ["product_id"]
PAIR = ["brand", "category2"]
PRICE_GROUP = [*PAIR, "price_band"]
BLIND_DAYS, RECENT_DAYS, PRIOR_DAYS = 13, 60, 180
RATE_SCALE = RECENT_DAYS / PRIOR_DAYS
PERSONAL_COLUMNS = [
    "sku_spend_at_risk", "sku_repeats_at_risk", "sku_supported_spend_fraction",
    "pair_spend_at_risk", "pair_repeats_at_risk", "pair_trend",
    "replacement_spend_share", "replacement_mean_log_sku_count",
]
COVERAGE_COLUMNS = [
    "prior_history", "sku_spend", "sku_repeat", "sku_prior_3_orders", "pair_spend",
    "pair_repeat", "replacement_group", "replacement_found", "personal_measurable",
]


def sales_by_customer(lines, keys):
    return (lines.drop_duplicates([CUSTOMER, "order_id", *keys])
            .groupby([CUSTOMER, *keys], observed=True).size())


def peer_portfolio(prior, recent, keys):
    """Attach counts of other customers' distinct orders to the prior portfolio."""
    spend = prior.groupby([CUSTOMER, *keys], observed=True).paid_price.sum().rename("spend")
    old_own, new_own = sales_by_customer(prior, keys), sales_by_customer(recent, keys)
    portfolio = pd.concat([spend, old_own.rename("old_own")], axis=1).join(new_own.rename("new_own"))
    portfolio = portfolio.reset_index()
    for window, counts in (("old", old_own), ("new", new_own)):
        total = counts.groupby(level=keys).sum().rename("market_total")
        portfolio = portfolio.join(total, on=keys, validate="many_to_one")
        portfolio[window] = portfolio.pop("market_total").fillna(0) - portfolio[f"{window}_own"].fillna(0)
    portfolio["repeats"] = portfolio.old_own - 1
    return portfolio.drop(columns=["old_own", "new_own"])


def weighted_mean(portfolio, values, weight, eligible):
    mass = portfolio[weight].where(eligible, 0.)
    numerator = (values * mass).groupby(portfolio[CUSTOMER]).sum()
    denominator = mass.groupby(portfolio[CUSTOMER]).sum()
    return numerator / denominator.where(denominator.gt(0))


def continuation(portfolio, name):
    supported = portfolio.old.gt(0)
    active = portfolio.new.gt(0)
    # Poisson probability of seeing a sale at the old rate. A silent rare SKU has little evidence of loss.
    loss = (~active) * -np.expm1(-RATE_SCALE * portfolio.old)
    values = {}
    for weight in ("spend", "repeats"):
        values[f"{name}_{weight}_continued"] = weighted_mean(portfolio, active, weight, supported)
        values[f"{name}_{weight}_at_risk"] = weighted_mean(portfolio, loss, weight, supported)
    values[f"{name}_supported_spend_fraction"] = weighted_mean(
        portfolio, supported, "spend", pd.Series(True, index=portfolio.index),
    )
    return pd.DataFrame(values)


def brand_rate(portfolio):
    return np.log((1 + portfolio.new) / (1 + RATE_SCALE * portfolio.old))


def replacements(prior, recent):
    price = peer_portfolio(prior, recent, PRICE_GROUP)
    # A replacement must be absent from both the client's old portfolio and old peer sales.
    # Unique recent buyer IDs let us remove products whose only recent buyer was the focal client.
    candidates = recent.loc[~recent.product_id.isin(prior.product_id), [*PRICE_GROUP, *SKU, CUSTOMER]]
    recent_buyers = candidates.drop_duplicates().groupby([*PRICE_GROUP, *SKU], observed=True)[CUSTOMER]
    products = recent_buyers.agg(buyers="size", buyer="first").reset_index()
    new_count = products.groupby(PRICE_GROUP, observed=True).size().rename("new_count")
    exclusive = (products.loc[products.buyers.eq(1)].rename(columns={"buyer": CUSTOMER})
                 .groupby([CUSTOMER, *PRICE_GROUP], observed=True).size().rename("exclusive"))
    price = price.join(new_count, on=PRICE_GROUP, validate="many_to_one")
    price = price.join(exclusive, on=[CUSTOMER, *PRICE_GROUP], validate="many_to_one")
    new_peers = price.new_count.fillna(0) - price.exclusive.fillna(0)
    supported = price.old.add(price.new).gt(0)
    return pd.DataFrame({
        "replacement_spend_share": weighted_mean(price, new_peers.gt(0), "spend", supported),
        "replacement_mean_log_sku_count": weighted_mean(price, np.log1p(new_peers), "spend", supported),
    })


def assortment_at_cutoff(orders, customers, cutoff):
    history = orders.loc[orders.customer_id.isin(customers) & orders.order_date.lt(cutoff)].copy()
    age = (cutoff - history.order_date).dt.days
    recent = history.loc[age.between(BLIND_DAYS + 1, BLIND_DAYS + RECENT_DAYS)]
    prior = history.loc[age.between(BLIND_DAYS + RECENT_DAYS + 1, BLIND_DAYS + RECENT_DAYS + PRIOR_DAYS)]
    sku = peer_portfolio(prior, recent, SKU)
    pair = peer_portfolio(prior, recent, PAIR)
    brand = peer_portfolio(prior, recent, ["brand"])
    result = continuation(sku, "sku").join(continuation(pair, "pair"), how="outer")
    result["brand_trend"] = weighted_mean(brand, brand_rate(brand), "spend", brand.old.add(brand.new).gt(0))
    result["pair_trend"] = weighted_mean(pair, brand_rate(pair), "spend", pair.old.add(pair.new).gt(0))
    result = result.join(replacements(prior, recent)).reindex(customers)

    coverage = pd.DataFrame(index=customers)
    coverage["prior_history"] = customers.isin(prior.customer_id)
    coverage["sku_spend"] = result.sku_spend_continued.notna()
    coverage["sku_repeat"] = result.sku_repeats_continued.notna()
    strong = sku.loc[sku.old.ge(3) & sku.spend.gt(0), CUSTOMER]
    coverage["sku_prior_3_orders"] = customers.isin(strong)
    coverage["pair_spend"] = result.pair_spend_continued.notna()
    coverage["pair_repeat"] = result.pair_repeats_continued.notna()
    coverage["replacement_group"] = result.replacement_spend_share.notna()
    coverage["replacement_found"] = result.replacement_spend_share.gt(0)
    coverage["personal_measurable"] = coverage[["sku_spend", "sku_repeat", "pair_spend",
                                                "pair_repeat", "replacement_group"]].any(axis=1)
    return result, coverage


def build_assortment(data, cohorts):
    orders = pd.read_parquet(data / "orders.parquet",
                             columns=[CUSTOMER, "order_id", "product_id", "order_date", "paid_price"])
    products = pd.read_parquet(data / "products.parquet", columns=[*SKU, *PAIR])
    orders["order_date"] = pd.to_datetime(orders.order_date)
    orders["price_band"] = np.floor(np.log2(orders.paid_price.clip(lower=1))).astype("int16")
    orders = orders.merge(products, on=SKU, how="inner", validate="many_to_one")
    inputs = []
    coverage = []
    # Each scoring cohort has the same complete observation windows. Other cohorts end at other dates.
    for cutoff, subset in cohorts.groupby("scoring_date"):
        values, measured = assortment_at_cutoff(orders, pd.Index(subset.customer_id), cutoff)
        inputs.append(values)
        coverage.append(measured)
    return pd.concat(inputs).astype("float32"), pd.concat(coverage)


def evaluate(data, cohorts, original, assortment, coverage, chosen):
    row_by_id = pd.Series(cohorts.index, index=cohorts.customer_id)
    dates = cohorts.set_index(CUSTOMER, verify_integrity=True).scoring_date
    features = original.join(assortment, validate="one_to_one")
    rows = []
    for month in (2, 3):
        cutoff, = dates.loc[dates.dt.month.eq(month)].unique()
        fit, query = dates.index[dates.lt(cutoff)], dates.index[dates.eq(cutoff)]
        quarter = pd.read_parquet(RESEARCH / "oof" / f"scores_{month}.parquet", columns=["quarters"]).quarters.loc[query]
        y_fit = load_labels(data, row_by_id, fit)
        predictions = {"quarters": quarter}
        extra = {"tail": [], "brand_trend": ["brand_trend"],
                 "personal_assortment": ["brand_trend", *PERSONAL_COLUMNS]}
        for name, columns in extra.items():
            x = features.loc[:, [*chosen[month], *columns]]
            model = booster("lightgbm").fit(x.loc[fit], y_fit)
            predictions[name] = (quarter + model.predict_proba(x.loc[query])[:, 1]) / 2

        archived = pd.read_parquet(RESEARCH / "oof" / f"tail_predictions_{month}.parquet",
                                   columns=["lightgbm_plain_half"]).lightgbm_plain_half.loc[query]
        error = (predictions["tail"] - archived).abs().max()
        if not error <= 1e-12:
            raise ValueError(f"Saved control differs from reconstructed control for month {month}: {error:.3g}")
        y = load_labels(data, row_by_id, query)
        metrics = {name: compare(y, score, quarter, original.n_orders) for name, score in predictions.items()}
        previous = {"brand_trend": "tail", "personal_assortment": "brand_trend"}
        for name, scores in predictions.items():
            ap = metrics[name]["ap20"]
            row = dict(section="ranking", month=month, model=name, ap20=ap,
                       delta_vs_quarters=ap - metrics["quarters"]["ap20"],
                       delta_vs_tail=ap - metrics["tail"]["ap20"],
                       delta_vs_brand=ap - metrics["brand_trend"]["ap20"],
                       buyers_top5000=metrics[name]["buyers"])
            if name in previous:
                lower, upper = compare_predictions(y, predictions[previous[name]], scores)
                row.update(paired_reference=previous[name], delta_ci_low=lower, delta_ci_high=upper)
            rows.append(row)

        for kind in COVERAGE_COLUMNS:
            n = int(coverage.loc[query, kind].sum())
            rows.append(dict(section="coverage", month=month, measure=kind,
                             n=n, cohort_size=len(query), fraction=n / len(query)))
        for field in assortment:
            values = assortment.loc[query, field].dropna()
            rows.append(dict(section="feature", month=month, measure=field, n=len(values),
                             cohort_size=len(query), fraction=len(values) / len(query), mean=values.mean()))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT)
    args = parser.parse_args()
    cohorts = load_cohorts(args.data)
    chosen = selected_features()
    with threadpool_limits(limits=4):
        assortment, coverage = build_assortment(args.data, cohorts)
        original = build_features(args.data, cohorts, chosen)
        results = evaluate(args.data, cohorts, original, assortment, coverage, chosen)
    results.to_csv(RESULTS / "assortment.csv", index=False)
    summary = results.loc[results.section.eq("ranking"),
                          ["month", "model", "ap20", "delta_vs_brand", "buyers_top5000"]]
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))


if __name__ == "__main__":
    main()
