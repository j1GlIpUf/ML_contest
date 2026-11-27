"""Fit purchase ranking models on histories of category sequences."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from research.motif_clocks import habitual_sequences, motif_clocks
from research.sequence_rules import baskets, candidate_calendar, replay, weighted_gaps


def build_motifs(orders, cohort):
    blocks = []
    for _, customers in cohort.groupby('scoring_date'):
        b = baskets(orders, customers)
        pieces = [motif_clocks(b), habitual_sequences(b), weighted_gaps(b)]
        for key in ['main', 'mode', 'signature']:
            pieces.extend([replay(b, key), candidate_calendar(b, key)])
        blocks.append(pd.concat(pieces, axis=1).reindex(customers.customer_id))
    return pd.concat(blocks).reindex(cohort.customer_id).astype('float32')


def mechanism(frame):
    due = -frame.rhythm_order_all_median_delta
    regular = frame.n_orders.le(3) | frame.clock_volume_top_n.eq(frame.n_orders)
    in_window = due.between(-14, 104)
    return np.select([frame.n_orders.eq(1), regular & in_window, regular,
                      in_window, frame.motif_mode_3_support.gt(0)], [0, 1, 2, 3, 4], default=5)


def motif_experts(fit, labels, query, shared):
    """Shared training plus a larger weight for customers in the same calendar regime."""
    core = ['n_orders', 'recency', 'rhythm_order_all_median_delta', 'clock_volume_top_n']
    cols = list(dict.fromkeys([*shared, *core, *[c for c in fit if c.startswith('motif_mode_')]]))
    fit_group = mechanism(fit)
    query_group = mechanism(query)
    pred = np.empty(len(query))
    for group in np.unique(query_group):
        local = fit_group == group
        query_mask = query_group == group
        model = HistGradientBoostingClassifier(max_iter=230, max_leaf_nodes=15,
            learning_rate=.045, l2_regularization=6, min_samples_leaf=40,
            early_stopping=False, random_state=42)
        with threadpool_limits(limits=4):
            model.fit(fit[cols], labels, sample_weight=np.where(local, 1., .18))
            pred[query_mask] = model.predict_proba(query.loc[query_mask, cols])[:, 1]
    return pred
