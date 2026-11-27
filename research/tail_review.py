"""Separate changes of membership from reordering the original top-K audience."""
import numpy as np
import pandas as pd
from research.paired_validation import compare_predictions
from research.rerank import compare


def review(cache,choice):
    c=pd.read_parquet(cache/'cohorts.parquet')
    seg=pd.read_parquet(cache/'segments.parquet')
    warning_audit(cache)
    paired_warning_test(cache)
    results=[]
    uncertainty=[]
    for month in [2,3]:
        scores=pd.read_parquet(cache/f'scores_{month}.parquet')
        ref=scores.quarters
        y=c.target.loc[ref.index].astype(int)
        n=seg.n_orders
        base=compare(y,ref,ref,n)
        original=ref.sort_values(ascending=False,kind='stable').index[:5000]
        ceiling=y.loc[original].sum()/min(5000,y.sum())
        candidates=pd.read_parquet(cache/f'predictions_{month}.parquet')
        for name,p in candidates.items():
            order=p.loc[original].sort_values(ascending=False,kind='stable').index
            ranked=y.loc[order].to_numpy()
            fixed=(ranked*(ranked.cumsum()/np.arange(1,1+len(ranked)))).sum()/min(5000,y.sum())
            ap=compare(y,p,ref,n)['ap20']
            top2=p.sort_values(ascending=False,kind='stable').index[:2000]
            results.append(dict(month=month,model=name,ap20=ap,fixed_members_ap20=fixed,
                order_gain=fixed-base['ap20'],remaining_gain=ap-fixed,
                fraction_of_order_loss_recovered=(fixed-base['ap20'])/(ceiling-base['ap20']),
                nonbuyers_top2000=int((1-y.loc[top2]).sum())))
        if choice in candidates:
            lo,hi=compare_predictions(y,ref,candidates[choice],repetitions=1000)
            uncertainty.append(dict(month=month,model=choice,delta_lower=lo,delta_upper=hi))
    pd.DataFrame(results).to_csv('research/results/tail_order_effect.csv',index=False)
    pd.DataFrame(uncertainty).to_csv('research/results/tail_paired.csv',index=False)


def warning_audit(cache):
    cohorts=pd.read_parquet(cache/'cohorts.parquet')
    output=[]
    for month in [2,3]:
        frame=pd.read_parquet(cache/f'features_{month}.parquet')
        prior=pd.read_parquet(cache/f'scores_{month}.parquet').quarters.sort_values(ascending=False,kind='stable')
        for limit in [2000,5000]:
            ids=prior.index[:limit]
            ids=ids[frame.n_orders.loc[ids].between(4,15)]
            residual=prior.loc[ids]-cohorts.target.loc[ids]
            for name,values in frame.filter(regex='^warning_').loc[ids].items():
                valid=values.dropna()
                if valid.nunique()<2:
                    continue
                if valid.nunique()==2:
                    low=valid.index[valid==valid.min()]
                    high=valid.index[valid==valid.max()]
                else:
                    low=valid.index[valid<=valid.quantile(.25)]
                    high=valid.index[valid>=valid.quantile(.75)]
                    if not low.intersection(high).empty:
                        continue
                output.append(dict(month=month,top=limit,feature=name,low_n=len(low),high_n=len(high),
                    excess_false_low=residual.loc[low].mean(),excess_false_high=residual.loc[high].mean()))
    pd.DataFrame(output).to_csv('research/results/tail_warning_audit.csv',index=False)


def paired_warning_test(cache):
    """Add the warning block to identical predictors with deterministic full-column trees."""
    from research.tail_analysis import splice
    from research.tail_models import booster
    c=pd.read_parquet(cache/'cohorts.parquet')
    features=pd.concat([pd.read_parquet(cache/f'features_{m}.parquet') for m in [1,2,3]])
    january=pd.concat([pd.read_parquet(cache/f'january_fold_{i}.parquet').quarters for i in range(5)])
    scores=pd.concat([january,*[pd.read_parquet(cache/f'scores_{m}.parquet').quarters for m in [2,3]]])
    rank=scores.groupby(c.scoring_date.reindex(scores.index)).rank(pct=True,method='first')
    features['rank_pct']=rank
    warning=list(features.filter(regex='^warning_'))
    rows=[]
    for m in [2,3]:
        a=c.index[(c.scoring_date.dt.month<m)&rank.reindex(c.index).gt(.6)]
        b=c.index[c.scoring_date.dt.month.eq(m)]
        pool=b[rank.loc[b].gt(.6)]
        chosen=pd.read_csv(f'research/results/tail_selected_{m}.csv').feature
        base=[col for col in chosen if col not in warning]+['rank_pct']
        prior=scores.loc[b]
        for name,cols in [('fixed_core',base),('fixed_core_plus_warnings',base+warning)]:
            model=booster('lightgbm').set_params(colsample_bytree=1.)
            model.fit(features.loc[a,cols],c.target.loc[a].astype(int))
            p=pd.Series(model.predict_proba(features.loc[pool,cols])[:,1],index=pool)
            candidate=splice(prior,p,pool)
            rows.append(dict(month=m,model=name,**compare(c.target.loc[b].astype(int),candidate,prior,features.n_orders)))
        pd.DataFrame(rows).to_csv('research/results/tail_warning_ablation.csv',index=False)
