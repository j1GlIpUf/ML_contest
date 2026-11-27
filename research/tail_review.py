"""Separate changes of membership from reordering the original top-K audience."""
import numpy as np
import pandas as pd
from research.paired_validation import compare_predictions
from research.rerank import compare


def review(cache,choice):
    c=pd.read_parquet(cache/'cohorts.parquet')
    seg=pd.read_parquet(cache/'segments.parquet')
    warning_audit(cache)
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
