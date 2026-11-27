"""Learn soft customer groups from shopping history before fitting local predictors."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits


def gate_features(core, motifs):
    due = -core.rhythm_order_all_median_delta
    frame = pd.DataFrame(index=core.index)
    frame['orders'] = np.log1p(core.n_orders)
    frame['recency'] = np.log1p(core.recency)
    frame['pace'] = np.log1p((core.recency + due).clip(lower=0))
    frame['center'] = ((due-45).abs()/45).clip(upper=4)
    frame['in_window'] = due.between(-14,104).astype(float)
    frame['purity'] = core.clock_volume_top_n/core.n_orders
    for depth in [2,3]:
        pre=f'motif_mode_{depth}_'
        frame[f'log_support{depth}'] = np.log1p(motifs[pre+'support'].fillna(0))
        frame[f'freq{depth}'] = motifs[pre+'share']
        frame[f'repeat{depth}'] = motifs[pre+'repeat_p']
    return frame.fillna(-1.)


def learned_experts(fit, labels, query, gf, gq, group_type='mixture', oof=None):
    with threadpool_limits(limits=4):
        if group_type == 'mixture':
            scaler=StandardScaler().fit(gf)
            model=GaussianMixture(n_components=7,covariance_type='diag',reg_covar=.15,
                                  n_init=3,random_state=47).fit(scaler.transform(gf))
            train_gate=model.predict_proba(scaler.transform(gf))
            query_gate=model.predict_proba(scaler.transform(gq))
        else:
            tree=DecisionTreeRegressor(max_leaf_nodes=7,min_samples_leaf=max(500,len(fit)//30),
                                       random_state=47)
            tree.fit(gf,labels - oof,sample_weight=.3+.7*np.asarray(oof))
            leaves=np.unique(tree.apply(gf))
            train_gate=tree.apply(gf)[:,None]==leaves
            query_gate=tree.apply(gq)[:,None]==leaves
        scores=np.zeros(len(query))
        for expert in range(train_gate.shape[1]):
            weight=.16+.84*train_gate[:,expert]
            active=query_gate[:,expert]>1e-6
            if not active.any():
                continue
            hist=HistGradientBoostingClassifier(max_iter=210,max_leaf_nodes=16,min_samples_leaf=45,
                l2_regularization=6,learning_rate=.05,early_stopping=False,random_state=47)
            hist.fit(fit,labels,sample_weight=weight)
            scores[active]+=query_gate[active,expert]*hist.predict_proba(query.loc[active])[:,1]
    return scores / query_gate.clip(min=0).sum(axis=1)
