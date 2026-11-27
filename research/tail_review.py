"""Separate changes of membership from reordering the original top-K audience."""
import numpy as np
import pandas as pd
from research.paired_validation import compare_predictions
from research.rerank import compare


def review(cache,choice):
    c=pd.read_parquet(cache/'cohorts.parquet')
    seg=pd.read_parquet(cache/'segments.parquet')
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
