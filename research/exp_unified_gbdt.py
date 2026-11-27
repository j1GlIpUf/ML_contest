"""Run from the repository root with python -m research.exp_unified_gbdt.

Supply the original competition directory or ZIP via --data or LAMODA_DATA.
The report is replaced atomically after each completed validation month.
"""

import argparse
import ast
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits

from research.category_phase import category_phase
from research.motif_model import build_motifs
from research.personal_backtest import personal_backtest
from research.rerank import compare
from research.sequence_rules import baskets
from research.trajectory import trajectory_features

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "research/oof"
REPORT = ROOT / "research/results/exp_unified_gbdt.csv"
DATA_FILES = ("train.csv", "orders.parquet", "products.parquet")
DATES = pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"])
BASE_FUNCTIONS = (
    "features", "rich_features", "rhythm_features", "cycle_features",
    "sequence_features", "category_clocks", "transitions", "basket_patterns",
    "forecast_windows", "assemble", "receipt_table", "transition_clocks",
    "basket_graph", "personal_sessions", "extend_features",
)
SEED = 26
N_JOBS = 4


def find_data(path):
    candidates = [path] if path is not None else [
        ROOT / "data", ROOT / "lamoda-hse-repeat-purchase-2026.zip",
        *sorted(Path("/workspace/attachments").glob(
            "*/lamoda-hse-repeat-purchase-2026.zip"
        )),
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            return candidate
        if all((candidate / name).is_file() for name in DATA_FILES):
            return candidate
    raise FileNotFoundError("Pass the original competition directory or ZIP via --data or LAMODA_DATA.")


def read_table(data, name, **options):
    reader = pd.read_csv if name.endswith(".csv") else pd.read_parquet
    if data.is_dir():
        return reader(data / name, **options)
    with ZipFile(data) as archive, archive.open(name) as source:
        return reader(source, **options)


def load_inputs(data):
    cohorts = read_table(data, "train.csv", parse_dates=["scoring_date"])
    cohorts = cohorts[cohorts.scoring_date.isin(DATES)].set_index(
        "customer_id", verify_integrity=True
    )
    if not cohorts.target.isin([0, 1]).all() or set(cohorts.scoring_date) != set(DATES):
        raise ValueError("Expected binary labels and all three January–March cohorts.")
    orders = read_table(
        data, "orders.parquet",
        columns=["customer_id", "order_id", "order_date", "product_id",
                 "paid_price", "discount_percent", "loyalty_percent"],
    )
    cutoff = orders.customer_id.map(cohorts.scoring_date)
    orders["order_date"] = pd.to_datetime(orders.order_date)
    orders = orders.loc[orders.order_date.lt(cutoff)]
    products = read_table(
        data, "products.parquet",
        columns=["product_id", "brand", "category", "category2", "gender_product"],
    )
    orders = orders.merge(products, on="product_id", how="left", validate="many_to_one")
    return cohorts, orders


def base_features(cohort, orders):
    # Reuse only the feature definitions. Executing notebook cells would also
    # train the previous models and write their submission and validation files.
    path = ROOT / "lamoda_groups.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    definitions = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"]), filename=str(path)).body:
            if isinstance(node, ast.FunctionDef) and node.name in BASE_FUNCTIONS:
                definitions[node.name] = node
    module = ast.Module(body=[definitions[name] for name in BASE_FUNCTIONS], type_ignores=[])
    builders = {"np": np, "pd": pd, "orders": orders}
    exec(compile(module, str(path), "exec"), builders)
    history = pd.concat([
        builders[name](cohort) for name in
        ("rich_features", "rhythm_features", "cycle_features", "sequence_features")
    ], axis=1)
    core = builders["assemble"](
        builders["features"](cohort), history, builders["category_clocks"](cohort), cohort
    )
    _, full = builders["extend_features"](core, cohort)
    return full.set_axis(pd.Index(cohort.customer_id, name="customer_id"))


def build_matrix(cohorts, orders):
    frames = []
    for cutoff, rows in cohorts.groupby("scoring_date", sort=True):
        cohort = rows[["scoring_date"]].reset_index()
        history = orders.loc[orders.customer_id.isin(rows.index)]
        receipts = baskets(history, cohort)
        horizon = (cutoff + pd.DateOffset(months=3) - cutoff).days
        matrix = pd.concat([
            base_features(cohort, history),
            build_motifs(history, cohort),
            trajectory_features(receipts, horizon),
            personal_backtest(receipts),
            category_phase(receipts),
        ], axis=1, verify_integrity=True).reindex(rows.index)
        frames.append(matrix.astype("float32"))
    return pd.concat(frames).replace([np.inf, -np.inf], np.nan)


def control_scores(cohorts):
    january = pd.concat([
        pd.read_parquet(OOF / f"january_fold_{fold}.parquet", columns=["quarters"])
        for fold in range(5)
    ], verify_integrity=True)
    scores = {DATES[0]: january.quarters}
    for date in DATES[1:]:
        scores[date] = pd.read_parquet(OOF / f"scores_{date.month}.parquet", columns=["quarters"]).quarters
    for date, values in scores.items():
        expected = cohorts.index[cohorts.scoring_date.eq(date)]
        if not values.index.is_unique or len(values) != len(expected) or not expected.isin(values.index).all():
            raise ValueError(f"Archived control scores do not match the {date:%Y-%m} customer IDs.")
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Non-finite control score for {date:%Y-%m}.")
    return pd.concat(scores.values(), verify_integrity=True).reindex(cohorts.index)


def lgbm(colsample):
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.035, num_leaves=15, min_child_samples=55,
        colsample_bytree=colsample, subsample=0.8, subsample_freq=1,
        reg_alpha=0.5, reg_lambda=8, verbosity=-1, n_jobs=N_JOBS, random_state=SEED,
    )


def select_features(features, training, control):
    # January scores are OOF. February scores were fitted on January only.
    # Stable sorting retains exactly the upper 40% (rounded up) of each month.
    head = [
        control.loc[rows.index].nlargest(int(np.ceil(0.4 * len(rows)))).index
        for _, rows in training.groupby("scoring_date", sort=True)
    ]
    pool = head[0].append(head[1:])
    selector = lgbm(0.5)
    selector.fit(features.loc[pool].to_numpy(), training.target.loc[pool])
    gains = selector.booster_.feature_importance(importance_type="gain")
    order = np.argsort(-gains, kind="stable")
    cumulative = gains[order].cumsum()
    if cumulative[-1] <= 0:
        raise ValueError("The training candidate pool produced no positive feature gains.")
    # Retain columns accounting for 95% of the training gain.
    count = np.searchsorted(cumulative, 0.95 * cumulative[-1]) + 1
    return features.columns[order[:count]]


def model_families():
    families = {"lightgbm": {"lightgbm_035": lgbm(0.35), "lightgbm_050": lgbm(0.50)}}
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as error:
        if error.name != "catboost":
            raise
    else:
        families["catboost"] = {"catboost": CatBoostClassifier(
            iterations=360, depth=5, learning_rate=0.045, l2_leaf_reg=8,
            rsm=0.5, bootstrap_type="Bernoulli", subsample=0.8,
            verbose=False, allow_writing_files=False, thread_count=N_JOBS, random_seed=SEED,
        )}
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as error:
        if error.name != "xgboost":
            raise
    else:
        families["xgboost"] = {"xgboost": XGBClassifier(
            n_estimators=320, max_depth=4, learning_rate=0.045, min_child_weight=12,
            reg_alpha=0.5, reg_lambda=9, colsample_bytree=0.45, subsample=0.8,
            tree_method="hist", verbosity=0, n_jobs=N_JOBS, random_state=SEED,
        )}
    return families


def predict(features, training, query, control):
    # Exclude columns with no observed values in the training slice. The query
    # may contain categories or sequence depths that did not occur in training.
    fit = features.loc[training.index].dropna(axis=1, how="all")
    chosen = select_features(fit, training, control)
    x_train = fit.loc[:, chosen].to_numpy()
    x_query = features.loc[query, chosen].to_numpy()
    scores = pd.DataFrame({"control": control.loc[query]})
    family_scores = []
    for models in model_families().values():
        for name, model in models.items():
            model.fit(x_train, training.target)
            if isinstance(model, LGBMClassifier):
                scores[name] = model.booster_.predict(x_query, num_threads=N_JOBS)
            else:
                scores[name] = model.predict_proba(x_query)[:, 1]
        family_scores.append(scores[list(models)].mean(axis=1))
    scores["gbdt_mean"] = pd.concat(family_scores, axis=1).mean(axis=1)
    scores["gbdt_control_half"] = (scores.gbdt_mean + scores.control) / 2
    return scores, len(chosen), len(fit.columns)


def run(data):
    cohorts, orders = load_inputs(data)
    control = control_scores(cohorts)
    with threadpool_limits(limits=N_JOBS):
        features = build_matrix(cohorts, orders)
        del orders
        results = []
        for date in DATES[1:]:
            training = cohorts.loc[cohorts.scoring_date.lt(date)]
            query = cohorts.loc[cohorts.scoring_date.eq(date), "target"].astype(int)
            scores, selected, candidates = predict(features, training, query.index, control)
            for name, prediction in scores.items():
                stats = compare(query, prediction, scores.control, features.n_orders)
                results.append({
                    "month": date.month, "model": name, "ap20": stats["ap20"],
                    "buyers_top5000": stats["buyers"], "features_selected": selected if name != "control" else None,
                    "feature_candidates": candidates if name != "control" else None, "status": "measured",
                })
            report = pd.DataFrame(results).convert_dtypes()
            temporary = REPORT.with_suffix(".tmp")
            report.to_csv(temporary, index=False)
            temporary.replace(REPORT)
    summary = report.pivot(index="model", columns="month", values=["ap20", "buyers_top5000"])
    print(summary.round(5).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=os.environ.get("LAMODA_DATA"))
    args = parser.parse_args()
    try:
        data = find_data(args.data)
    except FileNotFoundError as error:
        parser.error(str(error))
    run(data)


if __name__ == "__main__":
    main()
