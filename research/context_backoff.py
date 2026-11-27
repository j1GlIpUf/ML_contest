"""Pool evidence across lengths of the last shopping-category sequence."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


def context_tables(core, motifs):
    rec = core.recency
    due = -core.rhythm_order_all_median_delta
    fields = ['support','share','confidence','entropy','branches','same','repeat_p','past_likelihood',
              'past_prefix_n','nonoverlap_n','cycle_due','step_due','unigram_due','bigram_due',
              'unigram_global_due','bigram_global_due','gap_per_order_due','unigram_n','bigram_n']
    tables = []
    for depth in range(1, 5):
        pre = f'motif_mode_{depth}_'
        ctx = motifs[[pre + f for f in fields]].set_axis(fields, axis=1).copy()
        ctx['depth'] = depth
        ctx['step_rate'] = ctx.support.fillna(0) / (core.n_orders-depth).clip(lower=1)
        ctx['available'] = ctx.support.gt(0).astype(float)
        ctx['personal_due'] = due
        ctx['personal_center'] = (due - 45).abs()
        ctx['in_window'] = due.between(-14,104).astype(float)
        for state in ['personal','bigram_global','unigram_global','cycle','step']:
            days = ctx[state+'_due']
            peak = np.exp(-.5 * ((days-45) / 25)**2)
            edge = np.exp(-.5 * ((days-1) / 8)**2) + np.exp(-.5 * ((days-89) / 8)**2)
            ctx[state+'_match'] = peak * ctx.step_rate
            ctx[state+'_evidence'] = peak * ctx.repeat_p
            ctx[state+'_edge'] = edge * ctx.available
        ctx['last_to_first'] = (ctx.cycle_due + rec) / (core.age_days + 1)
        tables.append(ctx.astype('float32'))
    return tables


def backoff_features(core, motifs):
    tables = context_tables(core, motifs)
    out = []
    for min_count in [1, 2, 3]:
        support = np.stack([z.support.fillna(0) for z in tables], axis=1)
        valid = support >= min_count
        index = np.where(valid, np.arange(4), -1).max(axis=1)
        values = np.stack([z.to_numpy() for z in tables], axis=1)
        selected = values[np.arange(len(core)), index.clip(0)]
        selected[index < 0] = np.nan
        df = pd.DataFrame(selected, index=core.index, columns=tables[0].columns)
        out.append(df.add_prefix(f'backoff_{min_count}_'))
    return pd.concat(out, axis=1)


def pooled_context(fit, labels, query, train_contexts, query_contexts):
    # Four views of the same label stay together in all train/evaluation splits.
    # The longest supported context receives the largest share of the vote.
    def views(frame, contexts):
        evidence = np.column_stack([z.support.reindex(frame.index).fillna(0).to_numpy() for z in contexts])
        weights = (1 + np.minimum(evidence, 4)) * (1 + (evidence > 0) * np.arange(4))
        weights = weights / weights.sum(axis=1, keepdims=True)
        frames = [frame.join(z.loc[frame.index].add_prefix('view_')) for z in contexts]
        return pd.concat(frames),weights.T
    a, w = views(fit,train_contexts)
    b, v = views(query,query_contexts)
    model = HistGradientBoostingClassifier(max_iter=240, learning_rate=.055,max_leaf_nodes=23,
        min_samples_leaf=100,l2_regularization=7,early_stopping=False,random_state=48)
    with threadpool_limits(limits=4):
        model.fit(a,np.tile(labels,4),sample_weight=w.reshape(-1))
        pred=model.predict_proba(b)[:,1].reshape(4,-1)
    return (pred*v).sum(axis=0)
