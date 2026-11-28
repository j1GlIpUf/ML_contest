"""Audit the published tail ensemble after removing individual purchases.

Run from the repository root with:
    python -m research.exp_basket_fragility --data /path/to/competition/files
"""
import argparse
import ast
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from research.category_phase import category_phase
from research.motif_clocks import habitual_sequences, motif_clocks
from research.rerank import compare
from research.sequence_rules import baskets, candidate_calendar, replay, weighted_gaps
from research.tail_features import warning_features
from research.tail_models import importance_columns, booster
from research.trajectory import trajectory_features
from research.personal_backtest import personal_backtest

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "research/oof"
SCENARIOS = ("recent1", "recent2", "recent3", "recent4", "older")
PAIRED = (*SCENARIOS, "unchanged")
SCORE_BINS = [0, .60, .70, .75, .80, .85, .90, .95, 1.00001]
NOTICEABLE = .03
SEED = 827


def notebook_recipes(stable=False):
    """Load reusable definitions only. Notebook cells and outputs stay untouched."""
    path = ROOT / "lamoda_groups.ipynb"
    notebook = json.loads(path.read_text())
    scope = dict(np=np, pd=pd, HistGradientBoostingClassifier=HistGradientBoostingClassifier,
                 threadpool_limits=threadpool_limits)
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            definitions.extend(node for node in tree.body if isinstance(node, ast.FunctionDef))
    module = ast.Module(body=definitions, type_ignores=[])
    if stable:
        # The published code defaults to quicksort in some intermediate
        # tables. With tied dates or counts, other customers can affect which
        # row wins. Apply stable sorting to both sides of each paired replay.
        for node in ast.walk(module):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "sort_values" and not any(k.arg == "kind" for k in node.keywords)):
                node.keywords.append(ast.keyword(arg="kind", value=ast.Constant("stable")))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope


def load_data(data):
    train = pd.read_csv(data / "train.csv", parse_dates=["scoring_date"])
    test = pd.read_csv(data / "test.csv", parse_dates=["scoring_date"])
    cohort = pd.concat([train, test], ignore_index=True)
    orders = pd.read_parquet(data / "orders.parquet")
    products = pd.read_parquet(data / "products.parquet",
                               columns=["product_id", "brand", "category", "category2", "gender_product"])
    orders["order_date"] = pd.to_datetime(orders.order_date)
    orders = orders.merge(products, on="product_id", how="left", validate="many_to_one")
    orders = orders.merge(cohort[["customer_id", "scoring_date"]], on="customer_id", validate="many_to_one")
    return cohort, orders.loc[orders.order_date.lt(orders.scoring_date)].drop(columns="scoring_date")


def scenario_orders(orders, cohort):
    candidates = (orders.loc[orders.customer_id.isin(cohort.customer_id),
                             ["customer_id", "order_id", "order_date"]]
                  .drop_duplicates(["customer_id", "order_id"])
                  .sort_values(["customer_id", "order_date", "order_id"],
                               ascending=[True, False, False]))
    groups = candidates.groupby("customer_id", sort=False)
    eligible = groups.order_id.transform("size").between(4, 15)
    candidates["position"] = groups.cumcount()
    candidates = candidates.loc[eligible].copy()
    old = candidates.loc[candidates.position.ge(4)].copy()
    generator = np.random.default_rng(SEED)
    old["draw"] = generator.random(len(old))
    chosen = {f"recent{i + 1}": candidates.loc[candidates.position.eq(i)] for i in range(4)}
    chosen["older"] = old.sort_values("draw").drop_duplicates("customer_id")
    return {name: selected.set_index("customer_id").order_id for name, selected in chosen.items()}


def build_core(scope, cohort):
    get = lambda name: scope[name](cohort)
    base = get("features")
    history = get("rich_features")
    for name in ("rhythm_features", "cycle_features", "sequence_features"):
        history = history.join(get(name))
    combined = scope["assemble"](base, history, get("category_clocks"), cohort)
    transition, full = scope["extend_features"](combined, cohort)
    return transition.set_axis(cohort.customer_id), full.set_axis(cohort.customer_id)


def transition_corpus(receipts, key):
    previous = receipts.groupby("customer_id", sort=False)[key].shift()
    return receipts.groupby([previous.rename("src"), key]).size().rename("population")


def fixed_candidate_calendar():
    """Reuse the repository recipe with an external, frozen cohort transition table."""
    tree = ast.parse(inspect.getsource(candidate_calendar))
    fn = tree.body[0]
    fn.args.args.append(ast.arg(arg="corpus"))
    fn.body = [node for node in fn.body if not (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "corpus" for t in node.targets)
    )]
    scope = dict(candidate_calendar.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(candidate_calendar), "exec"), scope)
    return scope["candidate_calendar"]


def share_context(orders, cohort):
    cutoff = cohort.scoring_date.iloc[0]
    history = orders.loc[orders.customer_id.isin(cohort.customer_id) &
                         orders.order_date.le(cutoff - pd.Timedelta(days=14))]
    receipts = baskets(history, cohort)
    discounts = history.groupby(["customer_id", "order_id"]).discount_percent.mean()
    return {
        **{key: transition_corpus(receipts, key) for key in ("main", "mode", "signature")},
        "categories": sorted(history.category.unique()),
        "mix_columns": [f"mix_{key}_{value}" for key in ("category", "gender_product")
                        for value in history[key].unique()],
        "discount_levels": {"warning_discount_above_median": discounts.median(),
                            "warning_discount_above_p90": discounts.quantile(.9)},
    }


def original_signatures(receipts, visible, categories):
    local = sorted(visible.category.unique())
    if local != categories:
        signature = receipts.signature.to_numpy()
        corrected = np.zeros(len(receipts), dtype="int64")
        for i, value in enumerate(local):
            corrected |= ((signature >> i) & 1) << categories.index(value)
        receipts = receipts.assign(signature=corrected)
    return receipts


def build_blocks(scope, orders, cohort, original_context=None, core=None):
    scope["orders"] = orders
    if core is None:
        _, core = build_core(scope, cohort)
    receipts = baskets(orders, cohort)
    cutoff = cohort.scoring_date.iloc[0]
    if original_context is not None:
        history = orders.loc[orders.customer_id.isin(cohort.customer_id) &
                             orders.order_date.le(cutoff - pd.Timedelta(days=14))]
        receipts = original_signatures(receipts, history, original_context["categories"])
        for column in original_context["mix_columns"]:
            if column not in core:
                core[column] = 0.
    pieces = [motif_clocks(receipts), habitual_sequences(receipts), weighted_gaps(receipts)]
    for key in ("main", "mode", "signature"):
        calendar = (candidate_calendar(receipts, key) if original_context is None else
                    fixed_candidate_calendar()(receipts, key, original_context[key]))
        pieces += [replay(receipts, key), calendar]
    extra = pd.concat(pieces, axis=1).reindex(core.index).astype("float32")
    personal = personal_backtest(receipts)
    phase = category_phase(receipts)
    horizon = (cutoff + pd.DateOffset(months=3) - cutoff).days
    warning = warning_features(orders, cohort, core.astype("float32"))
    if original_context is not None:
        # The threshold is a cohort-level statistic. Keep it fixed while the
        # fraction of this customer's remaining receipts is recomputed.
        discounts = history.groupby(["customer_id", "order_id"]).discount_percent.mean()
        for name, level in original_context["discount_levels"].items():
            warning[name] = discounts.gt(level).groupby(level=0).mean()
    tail = core.astype("float32")
    for block in (extra.filter(regex="^motif_mode_"), trajectory_features(receipts, horizon),
                  personal, phase, warning):
        tail = tail.join(block)
    return core, extra, personal, phase, tail


def save(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)


def prepare(data, cache, months):
    cache.mkdir(parents=True, exist_ok=True)
    cohort, orders = load_data(data)
    scope = notebook_recipes()
    paired_scope = notebook_recipes(stable=True)
    scope["orders"] = orders
    labelled = cohort.loc[cohort.target.notna()]
    core_paths = [cache / f"original_{month}.parquet" for month in (1, 2, 3)]
    if not all(path.exists() for path in core_paths):
        trans, full = build_core(scope, labelled)
        pd.Series(trans.columns).to_json(cache / "transition_columns.json", orient="values")
        pd.Series(full.columns).to_json(cache / "core_columns.json", orient="values")
        for month, rows in labelled.groupby(labelled.scoring_date.dt.month):
            save(full.loc[rows.customer_id], cache / f"original_{month}.parquet")
    columns = json.loads((cache / "core_columns.json").read_text())
    save(cohort.set_index("customer_id"), cache / "cohort.parquet")
    for month in months:
        rows = cohort.loc[cohort.scoring_date.dt.month.eq(month)]
        order_choices = scenario_orders(orders, rows)
        original_path = cache / f"original_{month}.parquet"
        frame = pd.read_parquet(original_path) if original_path.exists() else None
        # The feature definitions use the full catalog for categorical codes.
        # Orders from other dates are only present for that vocabulary.
        context = share_context(orders, rows)
        for scenario in ("original", *PAIRED):
            path = cache / f"tail_{month}_{scenario}.parquet"
            if path.exists():
                continue
            if scenario == "original":
                query, history = rows, orders
                core = frame
            elif scenario == "unchanged":
                query, history = rows.loc[rows.customer_id.isin(order_choices["recent1"].index)], orders
                core = None
            else:
                dropped = order_choices[scenario]
                query = rows.loc[rows.customer_id.isin(dropped.index)]
                history = orders.loc[~orders.order_id.eq(orders.customer_id.map(dropped))]
                core = None
            baseline = scenario == "original"
            result = build_blocks(scope if baseline else paired_scope, history, query,
                                  None if baseline else context, core=core)
            for name, block in zip(("core", "extra", "personal", "phase", "tail"), result):
                if name == "core":
                    block = block.reindex(columns=columns)
                if name == "tail":
                    # Take tail columns in the exact order of the unmodified recipe.
                    desired = columns + [c for c in block if c not in result[0].columns]
                    block = block.reindex(columns=desired)
                save(block, cache / f"{name}_{month}_{scenario}.parquet")


def feature_table(cache, block, dates, ids, columns=None):
    parts = []
    for month, rows in dates.loc[ids].groupby(dates.dt.month, sort=True):
        reference = pd.read_parquet(cache / f"{block}_{month}_original.parquet", columns=columns)
        parts.append(reference.loc[rows.index])
    return pd.concat(parts).loc[ids]


def model_inputs(cache, month, block, dates, fit, held, layouts):
    original = feature_table(cache, block, dates, fit.append(held))
    holdouts = []
    for scenario, aliases in layouts.items():
        if scenario == "train":
            replacement = original.loc[fit].copy()
        else:
            replacement = pd.read_parquet(cache / f"{block}_{month}_{scenario}.parquet").loc[aliases.index]
        holdouts.append(replacement.set_axis(aliases))
    return pd.concat([original, *holdouts])


def make_layout(cache, month, fit, held, inner_rank):
    layouts = {}
    next_id = -1
    for scenario in PAIRED + (("train",) if inner_rank else ()):
        index = fit if scenario == "train" else held.intersection(
            pd.read_parquet(cache / f"core_{month}_{scenario}.parquet", columns=[]).index
        )
        layouts[scenario] = pd.Series(np.arange(next_id, next_id - len(index), -1), index=index)
        next_id -= len(index)
    return layouts


def mapped_rank(raw, mixture, held, layouts):
    """Keep the rank-to-probability reference fixed across purchase deletions."""
    mapped = pd.Series(index=raw.index, dtype=float)

    def reference(ids):
        observed = raw.loc[ids]
        return (np.sort(observed), np.sort(mixture.loc[ids]),
                np.linspace(0, 1, len(observed)))

    known, values, grid = reference(held)
    mapped.loc[held] = np.interp(raw.loc[held].rank(pct=True), grid, values)
    for name, aliases in layouts.items():
        changed = raw.loc[aliases]
        if name == "train":
            _, train_values, train_grid = reference(aliases)
            mapped.loc[aliases] = np.interp(changed.rank(pct=True), train_grid, train_values)
        else:
            original = raw.loc[aliases.index].to_numpy()
            change = changed.to_numpy()
            left = np.searchsorted(known, change, side="left") - (original < change)
            right = np.searchsorted(known, change, side="right") - (original <= change)
            rank = (left + right + 2) / (2 * len(known))
            mapped.loc[aliases] = np.interp(rank, grid, values)
    return mapped.to_numpy()


def fit_quarters(scope, core, transition_columns, extra, personal, fit, query, y, dates,
                 held, layouts):
    """Fit the same components as stage_one and predict every history together."""
    from research.behavior_experts import gate_features, learned_experts
    from research.context_backoff import backoff_features
    from research.motif_model import motif_experts
    from research.window_models import forecast

    fit_model = scope["fit_model"]
    full_model = None

    def reuse_full_model(frame, labels):
        # The general estimate and both feature selectors fit the same model
        # on the full training matrix. Share it within this split only.
        nonlocal full_model
        if not frame.columns.equals(core.columns) or not frame.index.equals(fit):
            return fit_model(frame, labels)
        if full_model is None:
            full_model = fit_model(frame, labels)
        return full_model

    scope["fit_model"] = reuse_full_model
    chosen = list(scope["select_by_gain"](core.loc[fit], y, 110))
    estimates = []
    for columns in (transition_columns, core.columns):
        x, q = core.loc[fit, columns], core.loc[query, columns]
        general = scope["fit_model"](x, y).predict_proba(q)[:, 1]
        specialist = scope["segmented_predict"](x, y, q)
        mixture = pd.Series((general + specialist) / 2, index=query)
        ranker = scope["fit_ranker"](x, y, dates.loc[fit])
        raw = pd.Series(ranker.predict(q.to_numpy()), index=query)
        estimates.append(.40 * mixture.to_numpy() + .60 * mapped_rank(raw, mixture, held, layouts))
    del x, q, ranker
    regional = scope["regional_forecast"](core.loc[fit], y, core.loc[query])
    scope["fit_model"] = fit_model
    del full_model
    first = .25 * (estimates[0] + estimates[1]) + .50 * regional
    general_features = core[chosen].join(extra)
    general = forecast(general_features.loc[fit], y, general_features.loc[query])
    del general_features
    history = extra.filter(regex="^motif_mode_")
    local_columns = list(dict.fromkeys([*chosen, "n_orders", "recency", "rhythm_order_all_median_delta",
                                        "clock_volume_top_n"]))
    local_features = core[local_columns].join(history)
    local = motif_experts(local_features.loc[fit], y, local_features.loc[query], chosen)
    del local_features
    motifs = .25 * first + .25 * general + .50 * local
    gates = gate_features(core, history)
    features = core[chosen].join(history)
    group = learned_experts(features.loc[fit], y, features.loc[query], gates.loc[fit], gates.loc[query])
    features = features.join(backoff_features(core, history)).join(personal)
    pred = forecast(features.loc[fit], y, features.loc[query])
    return pd.DataFrame({"regional": regional, "motifs": motifs,
                         "quarters": .50 * motifs + .25 * (group + pred)}, index=query)


def head_scores(cache, month, dates, fit, held, layouts, labels, training_quarters, published):
    if published:
        # These archived columns were selected inside the corresponding
        # training split. The published tail notebook also cast the complete
        # feature table to float32 before fitting its final model.
        features = ROOT / f"research/results/tail_selected_{month}.csv"
        chosen = pd.read_csv(features).feature.to_list()
    else:
        percentile = training_quarters.groupby(dates.loc[fit]).rank(pct=True, method="first")
        top = fit[percentile.gt(.60)]
        chosen = list(importance_columns(feature_table(cache, "tail", dates, top), labels.loc[top]).index)
    deletions = {name: aliases for name, aliases in layouts.items() if name != "train"}
    query = held.append([pd.Index(aliases) for aliases in deletions.values()])
    train = feature_table(cache, "tail", dates, fit, columns=chosen)
    tail = [feature_table(cache, "tail", dates, held, columns=chosen)]
    for scenario, aliases in deletions.items():
        features = pd.read_parquet(cache / f"tail_{month}_{scenario}.parquet", columns=chosen)
        tail.append(features.loc[aliases.index].set_axis(aliases))
    query_data = pd.concat(tail)
    if published:
        train = train.astype("float32")
        query_data = query_data.astype("float32")
    model = booster("lightgbm").fit(train, labels)
    return pd.Series(model.predict_proba(query_data)[:, 1], index=query)


def score_split(cache, month, name, fit, held, cohort, archive):
    dest = cache / f"scored_{name}.parquet"
    if dest.exists():
        return pd.read_parquet(dest)
    inner_rank = month == 1
    layouts = make_layout(cache, month, fit, held, inner_rank)
    query = held.append([pd.Index(aliases) for aliases in layouts.values()])
    dates, y = cohort.scoring_date, cohort.target.loc[fit].astype(int)
    core = model_inputs(cache, month, "core", dates, fit, held, layouts)
    extra = model_inputs(cache, month, "extra", dates, fit, held, layouts)
    stage_path = cache / f"quarter_stage_{name}.parquet"
    if stage_path.exists():
        stage = pd.read_parquet(stage_path)
    else:
        personal = model_inputs(cache, month, "personal", dates, fit, held, layouts)
        transition_columns = json.loads((cache / "transition_columns.json").read_text())
        stage = fit_quarters(notebook_recipes(), core, transition_columns, extra, personal,
                             fit, query, y, dates, held, layouts)
        save(stage, stage_path)
        del personal
    # In January the archived training-row folds were built with labels from
    # this outer fold. Rank only in-split training predictions for selection
    # so the held-out labels cannot influence the January head model.
    training_quarters = (stage.quarters.loc[layouts["train"]].set_axis(fit)
                         if inner_rank else archive.quarters.loc[fit])
    plain = head_scores(cache, month, dates, fit, held, layouts, y, training_quarters,
                        published=not inner_rank)
    simulated = (stage.quarters.loc[plain.index] + plain) / 2
    if month == 1:
        official = simulated.loc[held]
    elif month == 4:
        official = pd.read_csv(ROOT / "submission_tail.csv", float_precision="round_trip").set_index(
            "customer_id"
        ).score.reindex(held)
    else:
        official = pd.read_parquet(OOF / f"tail_predictions_{month}.parquet").lightgbm_plain_half.loc[held]
    if month != 1 and not np.allclose(simulated.loc[held], official, rtol=0, atol=1e-10):
        raise ValueError(f"The restored ensemble differs from the published month {month} scores")
    baseline = core.loc[held]
    result = pd.DataFrame({
        "score": official, "n_orders": baseline.n_orders,
        "underlying_score": simulated.loc[held],
        "quarters_error": stage.quarters.loc[held] - archive.quarters.loc[held],
        "consensus_sd": archive.loc[held].std(axis=1, ddof=0),
    }, index=held)
    unchanged = layouts["unchanged"]
    paired = simulated.loc[unchanged].set_axis(unchanged.index)
    reference_core = core.loc[unchanged].set_axis(unchanged.index)
    result["paired_baseline_error"] = simulated.loc[paired.index] - paired
    for scenario in SCENARIOS:
        alias = layouts[scenario]
        result[f"drop_{scenario}"] = paired.loc[alias.index] - simulated.loc[alias].set_axis(alias.index)
        changed_core = core.loc[alias].set_axis(alias.index)
        due_before = -reference_core.loc[alias.index, "rhythm_order_all_median_delta"]
        due_after = -changed_core.rhythm_order_all_median_delta
        old_motif = extra.loc[alias.index, "motif_mode_2_support"]
        new_motif = extra.loc[alias, "motif_mode_2_support"].set_axis(alias.index)
        result[f"lost_cadence_{scenario}"] = (due_before.between(0, 90) & ~due_after.between(0, 90))
        result[f"lost_repeat_{scenario}"] = old_motif.gt(0) & ~new_motif.gt(0)
    medium = result.n_orders.between(4, 15)
    if result.loc[medium, [f"drop_{s}" for s in SCENARIOS[:4]]].isna().any().any():
        raise ValueError(f"Missing purchase deletions for {name}")
    if not result.loc[medium & result.n_orders.gt(4), "drop_older"].notna().all():
        raise ValueError(f"Missing older purchase deletions for {name}")
    save(result, dest)
    return result


def load_archive():
    january = [pd.read_parquet(OOF / f"january_fold_{i}.parquet") for i in range(5)]
    scores = [pd.read_parquet(OOF / f"scores_{month}.parquet") for month in (2, 3, 4)]
    return january, pd.concat([*[fold.drop(columns="fold") for fold in january], *scores])


def score_month(cache, month):
    cohort = pd.read_parquet(cache / "cohort.parquet")
    dates = cohort.scoring_date.dt.month
    january, archive = load_archive()
    ids = cohort.index[dates.eq(month)]
    if month == 1:
        parts = []
        for fold, archive_fold in enumerate(january):
            held = ids[ids.isin(archive_fold.index)]
            fit = ids[~ids.isin(held)]
            if not held.equals(archive_fold.index):
                raise ValueError("January row order differs from archived folds")
            parts.append(score_split(cache, month, f"jan_{fold}", fit, held, cohort, archive))
        result = pd.concat(parts).loc[ids]
    else:
        fit = cohort.index[dates.lt(month)]
        result = score_split(cache, month, str(month), fit, ids, cohort, archive)
    save(result, cache / f"score_month_{month}.parquet")
    return result


def characterize(score):
    sample = score.loc[score.n_orders.between(4, 15)].copy()
    recent = sample[[f"drop_{s}" for s in SCENARIOS[:4]]]
    sample["max_drop"] = recent.max(axis=1).clip(lower=0)
    sample["mean_recent_drop"] = recent.mean(axis=1)
    sample["n_noticed"] = recent.ge(NOTICEABLE).sum(axis=1)
    sample["has_older"] = sample.drop_older.notna()
    sample["excess"] = sample.max_drop - sample.drop_older
    sample["mean_excess"] = sample.mean_recent_drop - sample.drop_older
    biggest = recent.to_numpy().argmax(axis=1)
    for signal in ("cadence", "repeat"):
        lost = sample[[f"lost_{signal}_{scenario}" for scenario in SCENARIOS[:4]]]
        sample[f"lost_{signal}"] = lost.to_numpy(dtype=bool)[np.arange(len(sample)), biggest]
    sample["stratum"] = (sample.n_orders.astype(int).astype(str) + "/" +
                         pd.cut(sample.score, SCORE_BINS, right=False).cat.codes.astype(str))
    return sample


def conditional_rate(sample, labels, exposed, control):
    rows = sample.loc[exposed | control]
    outcome = (1 - labels.loc[rows.index]).astype(float)
    exposure = exposed.loc[rows.index].astype(float)
    mean_exposure = exposure.groupby(rows.stratum).transform("mean")
    overlap = mean_exposure.between(0, 1, inclusive="neither")
    x = (exposure - mean_exposure).loc[overlap]
    y = (outcome - outcome.groupby(rows.stratum).transform("mean")).loc[overlap]
    if len(x) == 0:
        return dict(n_customers=0, n_exposed=0)
    beta = (x @ y) / (x @ x)
    se = np.sqrt(np.square(x * (y - beta * x)).sum()) / (x @ x)
    return dict(n_customers=len(x), n_exposed=int(exposure.loc[overlap].sum()),
                conditional_nonbuy_delta=beta, ci_low=beta - 1.96 * se, ci_high=beta + 1.96 * se)


def calibration_features(sample):
    baseline = logit(sample.score.clip(1e-6, 1 - 1e-6)).to_numpy()[:, None]
    added = np.column_stack([
        sample.n_noticed.eq(1), sample.n_noticed.gt(1),
        sample.max_drop.clip(0, .35) / .10, sample.has_older,
        sample.excess.fillna(0).clip(-.35, .35) / .10,
        sample.drop_older.fillna(0).clip(-.35, .35) / .10,
    ]).astype(float)
    return baseline, added


def freeze(cache):
    scores = pd.read_parquet(cache / "score_month_1.parquet")
    cohort = pd.read_parquet(cache / "cohort.parquet", columns=["target"])
    sample = characterize(scores)
    target = cohort.target.loc[sample.index].astype(int)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    strata = encoder.fit_transform(sample[["stratum"]])
    baseline, added = calibration_features(sample)
    base = np.column_stack([strata, baseline])
    common = LogisticRegression(C=.2, max_iter=500).fit(base, target)
    full = LogisticRegression(C=.2, max_iter=500).fit(np.column_stack([base, added]), target)
    # Remove the base-only calibration. Rank changes then come from the
    # incremental relationship between deletion sensitivity and outcomes.
    recipe = {
        "score_bins": SCORE_BINS, "drop_threshold": NOTICEABLE, "max_penalty_logit": .20,
        "strata": encoder.categories_[0].tolist(),
        "intercept": float(full.intercept_[0] - common.intercept_[0]),
        "base_weights": (full.coef_[0, :base.shape[1]] - common.coef_[0]).tolist(),
        "added_weights": full.coef_[0, base.shape[1]:].tolist(),
        "consensus_sd": float(sample.consensus_sd.quantile(.25)),
        "training_rows": len(sample),
    }
    path = cache / "january_rule.json"
    contents = json.dumps(recipe, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text() != contents:
        raise ValueError("The January rule already exists and differs. Use a separate cache for a new recipe.")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(contents)
    temporary.replace(path)
    return recipe


def apply_rule(scores, sample, recipe):
    baseline, added = calibration_features(sample)
    strata = sample.stratum.to_numpy()[:, None] == np.asarray(recipe["strata"])[None, :]
    common = np.column_stack([strata, baseline])
    change = recipe["intercept"] + common @ recipe["base_weights"] + added @ recipe["added_weights"]
    # Only penalize histories where removal of exactly one recent receipt
    # causes a large drop. Histories with other drop patterns stay untouched.
    change = np.clip(change, -recipe["max_penalty_logit"], 0)
    eligible = sample.n_noticed.eq(1) & sample.stratum.isin(recipe["strata"]) & (change < 0)
    adjusted = scores.score.copy()
    adjusted.loc[sample.index[eligible]] = expit(baseline[eligible, 0] + change[eligible])
    return adjusted


def describe(month, scores, sample, labels, recipe):
    rows = []
    order = scores.score.sort_values(ascending=False, kind="stable").index
    band = order[1500:3500]
    in_band = sample.index.isin(band)
    agrees = sample.consensus_sd.le(recipe["consensus_sd"])
    for name, selected in (
        ("all_4_15", sample), ("positions_1501_3500", sample.loc[in_band]),
        ("model_agreement", sample.loc[agrees]),
        ("positions_1501_3500_model_agreement", sample.loc[in_band & agrees]),
    ):
        single = selected.n_noticed.eq(1)
        following = selected.n_noticed.gt(1)
        full_errors = (scores.score - scores.underlying_score).abs()
        row = dict(section="fragility", month=month, variant=name, n_customers=len(selected),
                   n_exposed=int(single.sum()), n_multiple=int(following.sum()),
                   mean_max_drop=selected.max_drop.mean(), mean_n_noticed=selected.n_noticed.mean(),
                   mean_older_drop=selected.drop_older.mean(), mean_excess_max=selected.excess.mean(),
                   mean_excess_average=selected.mean_excess.mean(), n_older=int(selected.has_older.sum()),
                   mean_loss_cadence=selected.loc[single, "lost_cadence"].mean(),
                   mean_loss_repeat=selected.loc[single, "lost_repeat"].mean(),
                   baseline_replica_mae=full_errors.mean(), baseline_replica_max=full_errors.max(),
                   quarter_replica_mae=scores.quarters_error.abs().mean(),
                   paired_baseline_mae=selected.paired_baseline_error.abs().mean())
        for scenario in SCENARIOS[:4]:
            row[f"mean_drop_{scenario}"] = selected[f"drop_{scenario}"].mean()
        rows.append(row)
        if month == 1:
            comparisons = (
                ("one_vs_zero", single, selected.n_noticed.eq(0)),
                ("one_vs_multiple", single, following),
                ("older_only_vs_older_stable", selected.drop_older.ge(NOTICEABLE),
                 selected.has_older & selected.drop_older.lt(NOTICEABLE)),
            )
            for contrast, exposed, control in comparisons:
                rows.append(dict(section="january_association", month=month,
                                 variant=f"{name}:{contrast}",
                                 **conditional_rate(selected, labels, exposed, control)))
    return rows


def ranking_metrics(month, scores, adjusted, labels, quarter_ap):
    reference = scores.score
    base_order = reference.sort_values(ascending=False, kind="stable").index
    changed_order = adjusted.sort_values(ascending=False, kind="stable").index
    k = int(np.ceil(.2 * len(labels)))
    fixed_top = adjusted.loc[base_order[:k]].sort_values(ascending=False, kind="stable").index
    changed_members = changed_order[:k]
    member_only = base_order[base_order.isin(changed_members)]
    band_ids = base_order[1500:3500]
    fixed_band = adjusted.loc[band_ids].sort_values(ascending=False, kind="stable").index
    candidates = {
        "tail_baseline": base_order[:k], "fixed_1501_3500": base_order[:1500].append([fixed_band, base_order[3500:k]]),
        "fixed_top5000": fixed_top, "membership_only": member_only, "unrestricted": changed_members,
    }
    denom = min(k, labels.sum())
    base_top = base_order[:k]
    rows = []
    for name, top in candidates.items():
        hits = labels.loc[top].to_numpy(dtype=int)
        precision = hits.cumsum() / np.arange(1, k + 1)
        ap = precision @ hits / denom
        baseline_ap = ap if not rows else rows[0]["ap20"]
        # Loss due to nonbuyers in these positions, including their effect on
        # lower-ranked buyers. A fixed-band replay changes only this ordering.
        later = np.cumsum((hits / np.arange(1, k + 1))[::-1])[::-1]
        loss = (1 - hits) * later / denom
        entered, exited = top.difference(base_top), base_top.difference(top)
        medium = scores.n_orders.between(4, 15)
        rows.append(dict(section="ranking", month=month, variant=name, ap20=ap,
                         delta_vs_tail=ap - baseline_ap, delta_vs_quarters=ap - quarter_ap,
                         buyers_top5000=int(hits.sum()), buyers_1501_3500=int(hits[1500:3500].sum()),
                         nonbuyer_loss_1501_3500=loss[1500:3500].sum(), audience_swaps=len(entered),
                         added_buyers=int(labels.loc[entered].sum()),
                         dropped_buyers=int(labels.loc[exited].sum()),
                         false_4_15=int(((hits == 0) & medium.loc[top]).sum())))
    return rows


def report(cache, output, months):
    recipe = json.loads((cache / "january_rule.json").read_text())
    cohort = pd.read_parquet(cache / "cohort.parquet")
    archive = load_archive()[1]
    records = [dict(section="frozen_rule", month=1, variant="january_only",
                    recipe=json.dumps(recipe, sort_keys=True))]
    for month in months:
        scores = pd.read_parquet(cache / f"score_month_{month}.parquet")
        sample = characterize(scores)
        labels = cohort.target.loc[scores.index]
        records += describe(month, scores, sample, labels, recipe)
        if month in (2, 3):
            labels = labels.astype(int)
            quarters = archive.quarters.loc[labels.index]
            expected = compare(labels, quarters, quarters, scores.n_orders)["ap20"]
            records += ranking_metrics(month, scores, apply_rule(scores, sample, recipe), labels, expected)
        table = pd.DataFrame(records)
        temporary = output.with_suffix(".tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(output)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    local_data = ROOT if (ROOT / "train.csv").is_file() else ROOT / "data"
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("LAMODA_DATA", local_data)))
    parser.add_argument("--cache", type=Path, default=ROOT / ".checkpoints/basket_fragility")
    parser.add_argument("--output", type=Path, default=ROOT / "research/results/basket_fragility.csv")
    parser.add_argument("--months", type=int, nargs="+", choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    parser.add_argument("--part", choices=["all", "prepare", "score", "freeze", "report"], default="all")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.part in ("all", "prepare"):
            prep_months = sorted(set(args.months) | {1}) if args.part == "all" else args.months
            prepare(args.data, args.cache, prep_months)
        if args.part == "all":
            score_month(args.cache, 1)
            freeze(args.cache)
            for month in args.months:
                if month != 1:
                    score_month(args.cache, month)
        if args.part == "score":
            for month in args.months:
                if month != 1 and not (args.cache / "january_rule.json").exists():
                    raise ValueError("Freeze the January-only rule before scoring later cohorts")
                score_month(args.cache, month)
        if args.part == "freeze":
            freeze(args.cache)
        if args.part in ("all", "report"):
            rows = report(args.cache, args.output, args.months)
            if rows.section.eq("ranking").any():
                table = rows.query("section == 'ranking'").pivot(index="variant", columns="month", values="ap20")
                print(table.to_string(float_format="%.5f"))


if __name__ == "__main__":
    main()
