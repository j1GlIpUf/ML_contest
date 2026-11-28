"""Estimate the next purchase date from overlapping multi-order spans and analog paths."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from research.rerank import compare
from research.tail_models import booster
from research.sequence_rules import baskets

ROOT=Path(__file__).resolve().parents[1]


def span_features(receipts,horizon):
    b=receipts.sort_values(['customer_id','age','order_id'],ascending=[True,False,True]).copy()
    g=b.groupby('customer_id',sort=False)
    recent=g.age.min()
    position=g.cumcount(ascending=False)
    observations=[]
    preferred=[]
    for length in [1,2,3,4,5,6,8,10,12]:
        # The L-step span ending at the prospective next order starts at
        # the Lth latest known order. Compare with past completed L-step spans.
        span=g.age.shift(length)-b.age
        pivot=b.loc[position.eq(length-1)].set_index('customer_id').age
        due=span-b.customer_id.map(pivot)
        sub=pd.DataFrame({'customer_id':b.customer_id,'span':span,'due':due}).dropna()
        sub['inside']=sub.due.between(0,horizon-1)
        sub['central']=np.exp(-.5*((sub.due-horizon/2)/20)**2)
        history=sub.groupby('customer_id')
        stats=history.agg(median=('due','median'),mean=('due','mean'),latest=('due','last'),
                    probability=('inside','mean'),central=('central','mean'),n=('span','size'))
        dispersion=(sub.span-sub.customer_id.map(history.span.median())).abs().groupby(sub.customer_id).median()
        stats['mad']=dispersion
        stats['center']=(stats['median']-horizon/2).abs()
        stats['agreement']=stats.central/(1+dispersion/30)
        observations.append(stats.add_prefix(f'span_{length}_'))
        preferred.append(stats.assign(length=length).reset_index())
    candidates=pd.concat(preferred)
    candidates=candidates[candidates.n>=3].assign(cost=lambda d:(d.mad+12)*(1+2/d.n))
    best=candidates.sort_values(['cost','n'],ascending=[True,False]).drop_duplicates('customer_id').set_index('customer_id')
    observations.append(best[['length','median','latest','central','probability']].add_prefix('span_precise_'))
    pred=pd.concat([v.filter(regex='_median$') for v in observations[:-1]],axis=1)
    observations.append(pd.DataFrame({'span_consensus':pred.median(axis=1),'span_dispersion':pred.std(axis=1),
                          'span_vote':(pred.ge(0)&pred.lt(horizon)).sum(axis=1)/pred.count(axis=1).clip(lower=1)}))
    # Compare the shape of earlier shopping paths to the current path using
    # multiple durations. Each historical continuation has an observed outcome.
    pieces=[]
    latest=g.tail(1)
    for depth in [1,2,4,6]:
        mismatch=pd.Series(0.,index=b.index)
        available=pd.Series(True,index=b.index)
        for step in range(depth):
            duration=g.age.shift(step+1)-b.age
            last=b.customer_id.map(pd.Series(duration.loc[latest.index].to_numpy(),index=latest.customer_id))
            available &= duration.notna()&last.notna()
            mismatch+=(np.log1p(duration)-np.log1p(last)).abs()/(step+1)
        cand=b.loc[available & position.gt(0),['customer_id','age','mode']].copy()
        if cand.empty: continue
        cand['dist']=mismatch.loc[cand.index]
        cand['same']=cand['mode'].eq(cand.customer_id.map(latest.set_index('customer_id')['mode']))
        cand['weight']=np.exp(-2*cand.dist)*np.where(cand.same,1.,.4)
        next_dates=np.stack([(b.age-g.age.shift(-k)-b.customer_id.map(recent)).loc[cand.index] for k in range(1,7)],axis=1)
        cand['hit']=((next_dates>=0)&(next_dates<horizon)).any(axis=1)
        cand['complete']=cand.age-2*cand.customer_id.map(recent)>=horizon
        cand['due']=next_dates[:,0]
        cand['usable']=cand.hit|cand.complete
        cand['weighted_hit']=cand.hit*cand.weight
        cand['weighted_total']=cand.usable*cand.weight
        groups=cand.groupby('customer_id')
        stats=groups.agg(support=('due','size'),similarity=('weight','max'),hit_mass=('weighted_hit','sum'),
                         total=('weighted_total','sum'),min_distance=('dist','min'))
        stats['likelihood']=(stats.hit_mass+.5)/(stats.total+1.)
        closest=cand.sort_values(['weight','age'],ascending=[False,True]).drop_duplicates('customer_id').set_index('customer_id')
        stats['closest_due']=closest.due
        stats['closest_hit']=closest.hit.astype(float)
        pieces.append(stats.add_prefix(f'analog_{depth}_'))
    return pd.concat(observations+pieces,axis=1).reindex(recent.index).astype('float32')


def structure(cache,work):
    c=pd.read_parquet(cache/'cohorts.parquet')
    orders=pd.read_parquet(ROOT/'orders.parquet').merge(pd.read_parquet(ROOT/'products.parquet'),on='product_id',validate='many_to_one')
    orders['order_date']=pd.to_datetime(orders.order_date)
    blocks=[]
    for date,cohort in c.groupby('scoring_date'):
        dst=work/f'spans_{date.month}.parquet'
        if not dst.exists():
            b=baskets(orders,cohort.reset_index())
            h=(date+pd.DateOffset(months=3)-date).days
            span_features(b,h).to_parquet(dst)
        blocks.append(pd.read_parquet(dst))
    return c,pd.concat(blocks)


def run(cache,work):
    work.mkdir(exist_ok=True,parents=True)
    c,extra=structure(cache,work)
    return compare_blocks(cache,work,c,extra,{'span':'^span_','analog':'^analog_','both':'.'},'span_clocks')


def compare_blocks(cache,work,c,extra,patterns,name):
    rows=[]
    for m in [2,3,4]:
        a=c.index[c.scoring_date.dt.month<m]; b=c.index[c.scoring_date.dt.month==m]
        cols=pd.read_csv(ROOT/f'research/results/tail_selected_{m}.csv').feature.tolist()
        core=pd.concat([pd.read_parquet(cache/f'features_{i}.parquet',columns=cols) for i in range(1,m+1)])
        tail=pd.read_parquet(ROOT/f'research/oof/tail_predictions_{m}.parquet').lightgbm_plain_half.loc[b]
        quarters=pd.read_parquet(ROOT/f'research/oof/scores_{m}.parquet').quarters.loc[b]
        n=pd.read_parquet(cache/f'features_{m}.parquet',columns=['n_orders']).n_orders
        for key,regex in patterns.items():
            if m==4 and key!=winner['variant']:continue
            x=core.join(extra.filter(regex=regex))
            model=booster('lightgbm').fit(x.loc[a],c.target.loc[a].astype(int))
            p=pd.Series(model.predict_proba(x.loc[b])[:,1],index=b)
            # Baseline comparison replaces the previous LightGBM contribution.
            updated=.5*(quarters+p)
            updated.rename('score').to_frame().to_parquet(work/f'{name}_{key}_{m}.parquet')
            if m<4:
                for w in [.5,1.]:
                    rows.append(dict(month=m,variant=key,weight=w,**compare(c.target.loc[b].astype(int),w*updated+(1-w)*tail,tail,n)))
                pd.DataFrame(rows).to_csv(ROOT/f'research/results/{name}.csv',index=False)
        if m==2:
            winner=max(rows,key=lambda v:v['ap20'])
            print(name,'February choice',winner,flush=True)
        if m==3:
            march=next(v for v in rows if v['month']==3 and v['variant']==winner['variant'] and v['weight']==winner['weight'])
            print(name,'March',march,flush=True)
            if winner['ap20']<=.7785233428848994 or march['ap20']<=.785118034003418:break
        if m==4:
            pred=tail*(1-winner['weight'])+updated*winner['weight']
            pred.rename('score').to_frame().to_parquet(work/f'{name}_april.parquet')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=Path('.checkpoints/tail6_notebook'))
    parser.add_argument('--work',type=Path,default=Path('.checkpoints/round7'))
    args=parser.parse_args();run(args.cache,args.work)
