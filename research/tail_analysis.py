"""Decompose fixed-audience AP loss and compare alternative score transforms."""
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODELS = ['motifs','context','quarters','phase','regional']


def diagnostics(y, scores, segments):
    order = scores.sort_values(ascending=False,kind='stable').index[:int(np.ceil(.2*len(y)))]
    hits = y.loc[order].to_numpy()
    k = len(hits)
    denom = min(k,int(y.sum()))
    rank = np.arange(1,k+1)
    precision = hits.cumsum()/rank
    # Attribute every inverted pair (earlier nonbuyer, later buyer) once.
    fp_loss = (1-hits)*np.cumsum((hits/rank)[::-1])[::-1]/denom
    buyer_loss = hits*(1-precision)/denom
    assert np.isclose(fp_loss.sum(), buyer_loss.sum())
    rows = []
    for lo,hi in [(1,500),(501,1000),(1001,2000),(2001,3500),(3501,k)]:
        mask = (rank>=lo)&(rank<=hi)
        rows.append(dict(first=lo,last=hi,buyers=hits[mask].sum(),false_positives=(1-hits[mask]).sum(),
                         loss_by_false_positive=fp_loss[mask].sum(),loss_by_later_buyer=buyer_loss[mask].sum()))
    bands = []
    for name,idx in [('1-1500',order[:1500]),('1501-5000',order[1500:]),('p>0.65',scores.index[scores>.65])]:
        d = segments.loc[idx,['length','mono']].assign(pred=scores.loc[idx],y=y.loc[idx])
        for fields in [['length'],['mono'],['length','mono']]:
            g=d.groupby(fields,observed=True).agg(n=('y','size'),buyers=('y','sum'),mean_score=('pred','mean'),rate=('y','mean')).reset_index()
            g['band']=name
            g['gap']=g.rate-g.mean_score
            bands.append(g)
    return pd.DataFrame(rows),pd.concat(bands,ignore_index=True)


def blends(scores):
    p=scores[MODELS].clip(1e-6,1-1e-6)
    l=pd.DataFrame(logit(p),index=p.index,columns=p.columns)
    return pd.DataFrame({'arithmetic':p.mean(axis=1),'logit':expit(l.mean(axis=1)),
        'geometric':np.exp(np.log(p).mean(axis=1)), 'complement':-np.expm1(np.log1p(-p).mean(axis=1)),
        'dispersion':p.mean(axis=1)-.5*p.std(axis=1,ddof=0),
        'logit_dispersion':expit(l.mean(axis=1)-.5*l.std(axis=1,ddof=0))},index=p.index)


def calibrate(train_scores, y, train_group, scores, group, method):
    corrected=scores.copy()
    for key in group.unique():
        fit=(train_group==key)&(train_scores>.65)
        query=(group==key)&(scores>.65)
        if fit.sum()<30 or not query.any():
            continue
        x=logit(train_scores[fit].clip(1e-6,1-1e-6)).to_numpy()
        target=y[fit].to_numpy()
        def coefficients(params):
            return (1. if method=='shift' else np.exp(params[0]),
                    0. if method=='temperature' else params[-1])
        def objective(params):
            slope,intercept=coefficients(params)
            z=slope*x+intercept
            return (np.logaddexp(0,z)-target*z).sum()+15*(params@params)
        opt=minimize(objective,np.zeros(2 if method=='platt' else 1),method='BFGS')
        slope,intercept=coefficients(opt.x)
        corrected.loc[query]=expit(slope*logit(scores[query].clip(1e-6,1-1e-6))+intercept)
    return corrected


def splice(prior, new, query):
    """Permute only the candidate pool, preserving the original boundary and tail."""
    ordered = new.sort_values(ascending=True,kind='stable').index
    result = prior.copy()
    result.loc[ordered] = np.sort(prior.loc[query].to_numpy())
    return result


def meta_features(scores,segments):
    logits=logit(scores[MODELS].clip(1e-6,1-1e-6))
    return logits.assign(dispersion=logits.std(axis=1),minimum=scores[MODELS].min(axis=1),
                         count=np.log1p(segments.n_orders),mono=segments.mono.astype(float),due=segments.due_sign)


def head_meta(a,b,y,train_pool,query_pool,prior):
    scaler=StandardScaler().fit(a.loc[train_pool])
    model=LogisticRegression(C=.1,max_iter=500).fit(scaler.transform(a.loc[train_pool]),y.loc[train_pool])
    raw=pd.Series(model.predict_proba(scaler.transform(b.loc[query_pool]))[:,1],index=query_pool)
    return splice(prior,raw,query_pool)


def local_oracles(y, scores):
    """Perfectly reorder labels within specified blocks, preserving every other position."""
    k=int(np.ceil(.2*len(y)))
    ranked=y.loc[scores.sort_values(ascending=False,kind='stable').index[:k]].to_numpy()
    ranks=np.arange(1,k+1)
    denominator=min(k,int(y.sum()))
    def ap(values):
        return ((values.cumsum()/ranks)@values)/denominator
    original=ap(ranked)
    bins=[(1,500),(501,1000),(1001,2000),(2001,3500),(3501,k)]
    cases=[(f'{lo}-{hi}',[(lo,hi)]) for lo,hi in bins]
    cases.extend([('five_blocks',bins),('1-1500',[(1,1500)]),('1501-5000',[(1501,k)]),('all',[(1,k)])])
    rows=[]
    for name,blocks in cases:
        perfect=ranked.copy()
        for lo,hi in blocks:
            perfect[lo-1:hi]=np.sort(perfect[lo-1:hi])[::-1]
        ceiling=ap(perfect)
        rows.append(dict(block=name,oracle_ap=ceiling,gain=ceiling-original))
    return pd.DataFrame(rows)
