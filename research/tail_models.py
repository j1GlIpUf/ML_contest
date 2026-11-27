"""Fit head rerankers and weighted classifiers from fold-local forecast and feature tables."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit
from research.tail_analysis import calibrate, head_meta, meta_features, splice


def booster(kind):
    if kind=='lightgbm':
        return LGBMClassifier(n_estimators=300, learning_rate=.035,num_leaves=15,min_child_samples=55,
                              colsample_bytree=.4,reg_lambda=8,n_jobs=3,verbosity=-1,random_state=26)
    if kind=='catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=360,depth=5,learning_rate=.045,l2_leaf_reg=8,
                                  verbose=False,allow_writing_files=False,thread_count=3,random_seed=26)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=320,max_depth=4,learning_rate=.045,min_child_weight=12,reg_lambda=9,
                         colsample_bytree=.45,subsample=.9,tree_method='hist',n_jobs=3,random_state=26)


def importance_columns(x, y, limit=160):
    model=booster('lightgbm')
    model.fit(x,y)
    gain=pd.Series(model.booster_.feature_importance(importance_type='gain'),index=x.columns)
    return gain.sort_values(ascending=False,kind='stable').iloc[:limit]


def experiment(features, segments, dates, labels, scores, fit, query):
    prior=scores.loc[query,'quarters']
    oof=scores.loc[fit,'quarters']
    rank=oof.groupby(dates.loc[fit]).rank(pct=True,method='first')
    query_rank=prior.rank(pct=True,method='first')
    pool=fit[rank.gt(.6)]
    qpool=query[query_rank.gt(.6)]
    y=labels.loc[fit].astype(int)
    predictions={}
    group=segments.length.astype(str)+'_'+segments.mono.astype(str)
    for method in ['platt','temperature','shift']:
        predictions[method]=calibrate(oof,y,group.loc[fit],prior,group.loc[query],method)
    meta=meta_features(scores,segments)
    predictions['meta40']=head_meta(meta.loc[fit],meta.loc[query],y,pool,qpool,prior)
    importance=importance_columns(features.loc[pool],y.loc[pool])
    chosen=list(importance.index)
    data=features[chosen].copy()
    data['rank_pct']=pd.concat([rank,query_rank]).reindex(data.index)
    for frac in [.35,.40,.45]:
        train_ids=fit[rank.gt(1-frac)]
        query_ids=query[query_rank.gt(1-frac)]
        tree=booster('lightgbm')
        tree.fit(data.loc[train_ids],y.loc[train_ids])
        p=pd.Series(tree.predict_proba(data.loc[query_ids])[:,1],index=query_ids)
        key=f'head{int(frac*100)}'
        predictions[key]=splice(prior,p,query_ids)
        predictions[key+'_half']=(prior+predictions[key])/2
    # Fit a small log-odds correction on the candidate pool. Predictors see
    # per-month percentiles for every member; the intercept curve is obtained
    # without labels from the training OOF score distribution.
    train_ranks=scores.loc[fit].groupby(dates.loc[fit]).rank(pct=True)
    query_ranks=scores.loc[query].rank(pct=True)
    relative=pd.concat([train_ranks,query_ranks]).sub(pd.concat([rank,query_rank]),axis=0)
    extended=data.join(relative.add_prefix('rank_difference_'))
    grid=np.arange(1,len(fit)+1)/len(fit)
    ordered=np.sort(oof.to_numpy().clip(1e-5,1-1e-5))
    initial=logit(np.interp(rank.loc[pool],grid,ordered))
    intercept=logit(np.interp(query_rank.loc[qpool],grid,ordered))
    residual=booster('lightgbm').set_params(n_estimators=160,num_leaves=8,reg_lambda=25)
    residual.fit(extended.loc[pool],y.loc[pool],init_score=initial)
    offset=residual.booster_.predict(extended.loc[qpool],raw_score=True,num_threads=3)
    raw=pd.Series(intercept+.5*offset,index=qpool)
    predictions['offset40']=splice(prior,raw,qpool)
    original_top=query[query_rank.gt(.8)]
    predictions['offset_top20']=splice(prior,raw.loc[original_top],original_top)
    # Drop fresh indicators and reselect the same number of columns for the ablation.
    without=[c for c in features if not c.startswith('warning_')]
    reduced=importance_columns(features.loc[pool,without],y.loc[pool])
    sub=features[reduced.index].assign(rank_pct=data.rank_pct)
    tree=booster('lightgbm').fit(sub.loc[pool],y.loc[pool])
    raw=pd.Series(tree.predict_proba(sub.loc[qpool])[:,1],index=qpool)
    predictions['head40_no_warnings']=splice(prior,raw,qpool)
    data=features[chosen]
    for kind in ['lightgbm','catboost','xgboost']:
        for mode,w in [('plain',None),('weighted',1+8*oof**3)]:
            model=booster(kind)
            model.fit(data.loc[fit],y,sample_weight=w)
            prob=pd.Series(model.predict_proba(data.loc[query])[:,1],index=query)
            predictions[kind+'_'+mode]=prob
            predictions[kind+'_'+mode+'_half']=(prob+prior)/2
    return pd.DataFrame(predictions,index=query),importance
