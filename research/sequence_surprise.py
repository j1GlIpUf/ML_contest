"""Evaluate the customer's own older category and timing forecasts before each order."""
import numpy as np
import pandas as pd


def surprise_features(receipts):
    b=receipts.sort_values(['customer_id','age','order_id'],ascending=[True,False,True]).reset_index(drop=True).copy()
    b['state']=pd.Categorical(b['mode']).codes.astype('int64')+1
    g=b.groupby('customer_id',sort=False)
    n=g.cumcount()
    gap=-g.age.diff()
    log_gap=np.log1p(gap)
    b['log_gap']=log_gap
    prev=g.state.shift()
    b['route']=prev.fillna(0).astype(int)*10+b.state
    vocab=int(b.state.max())
    # Population-free prior on the initial observation; subsequent probabilities
    # use only the prefix ending immediately before the order being evaluated.
    prob=(b.groupby(['customer_id','state']).cumcount()+1/vocab)/(n+1)
    b['surprise0']=-np.log(prob)
    cols=['surprise0']
    context=pd.Series(0,index=b.index)
    for size in range(1,5):
        context=context*10+g.state.shift(size).fillna(0).astype(int)
        b['context']=context
        prefix=b.groupby(['customer_id','context'],sort=False)
        rule=b.groupby(['customer_id','context','state'],sort=False)
        cases=prefix.cumcount()
        hits=rule.cumcount()
        next_prob=(hits+2*prob)/(cases+2)
        b[f'lift{size}']=np.log(next_prob/prob).where(n.ge(size))
        b[f'surprise{size}']=-np.log(next_prob).where(n.ge(size))
        b[f'novel{size}']=((hits==0)&(cases>0)).where(n.ge(size)).astype(float)
        prob=next_prob
        cols += [f'lift{size}',f'surprise{size}',f'novel{size}']
    rec=g.age.min()
    parts=[]
    prefix_median=g.log_gap.expanding().median().droplevel(0).reindex(b.index)
    for name,groups in [('person',['customer_id']),('state',['customer_id','state']),('route',['customer_id','route'])]:
        subset=b.groupby(groups,sort=False)
        known=subset.log_gap.expanding().median().droplevel(list(range(len(groups)))).reindex(b.index)
        previous=known.groupby([b[k] for k in groups],sort=False).shift()
        background=prefix_median.groupby(b.customer_id).shift()
        prediction=previous.fillna(background)
        residual=log_gap-prediction
        b[f'{name}_bias']=residual
        b[f'{name}_error']=residual.abs()
        b[f'{name}_in_week']=((np.expm1(prediction)-gap).abs()<=7).where(prediction.notna()).astype(float)
        cols += [f'{name}_bias',f'{name}_error',f'{name}_in_week']
        # How does historical prediction error shift the current expected purchase date?
        correction=b.groupby('customer_id')[f'{name}_bias'].median().clip(-2,2)
        projected=np.expm1(g.log_gap.median()+correction)-rec
        parts.append(projected.rename('audit_'+name+'_corrected_due'))
    pos=g.cumcount(ascending=False)
    for label,mask in [('all',pos.ge(0)),('tail',pos.lt(3)),('previous',pos.between(3,7))]:
        z=b[mask].groupby('customer_id')[cols].agg(['mean','last'])
        z.columns=[f'audit_{label}_{a}_{s}' for a,s in z.columns]
        parts.append(z)
    return pd.concat(parts,axis=1).astype('float32')
