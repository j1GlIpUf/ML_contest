"""Fit catalog sale-date profiles on other customers and measure purchase timing."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from research.exp_span_clocks import ROOT
from research.rerank import compare
from research.tail_models import booster


def profile(reference,query,keys,prefix):
    grouped=reference.groupby(keys,sort=False)
    table=grouped.agg(median=('day','median'),first=('day','min'),last=('day','max'),
        sd=('day','std'),n=('day','size'),price=('paid_price','median'),
        sin=('season_sin','mean'),cos=('season_cos','mean'),discount=('discount_percent','mean'))
    z=query.join(table,on=keys)
    z['offset']=z.day-z['median']
    z['in_season']=z.season_cos*z.cos+z.season_sin*z.sin
    z['scaled']=z.offset/(z.sd.fillna(30)+14)
    z['out_of_range']=(z.day-z['last']).clip(lower=0)+(z['first']-z.day).clip(lower=0)
    z['price_change']=np.log1p(z.paid_price)-np.log1p(z.price)
    z['promo_change']=z.discount_percent-z.discount
    z['observed']=z.n.notna().astype(float)
    z['log_n']=np.log1p(z.n)
    cols=['offset','in_season','scaled','out_of_range','price_change','promo_change','observed','log_n']
    order=z.groupby(['customer_id','order_date','order_id'],sort=True)[cols].mean().reset_index()
    g=order.groupby('customer_id')
    all_stats=g[cols].agg(['mean','std','median'])
    all_stats.columns=['_'.join(pair) for pair in all_stats.columns]
    last=g.tail(1).set_index('customer_id')[cols].add_suffix('_latest')
    recent=g.tail(3).groupby('customer_id')[cols].mean().add_suffix('_recent')
    result=all_stats.join(last).join(recent)
    result['consistent_offset']=result.offset_median/(result.offset_std+14)
    return result.add_prefix(prefix+'_').astype('float32')


def create_features(cache,work):
    c=pd.read_parquet(cache/'cohorts.parquet')
    raw=pd.read_parquet(ROOT/'orders.parquet').merge(pd.read_parquet(ROOT/'products.parquet'),on='product_id',validate='many_to_one')
    raw['order_date']=pd.to_datetime(raw.order_date)
    raw=raw.join(c.scoring_date,on='customer_id')
    raw=raw[(raw.scoring_date-raw.order_date).dt.days>=14].copy()
    raw['day']=(raw.order_date-pd.Timestamp('2019-01-01')).dt.days
    angle=raw.day*2*np.pi/365.25
    raw['season_sin']=np.sin(angle);raw['season_cos']=np.cos(angle)
    for m in [2,3,4]:
        path=work/f'catalog_dates_matrix_{m}.parquet'
        if path.exists():continue
        fit=c.index[c.scoring_date.dt.month<m];query=c.index[c.scoring_date.dt.month==m]
        chunks=[]
        for a,b in [*((fit[i],fit[j]) for i,j in KFold(3,shuffle=True,random_state=26).split(fit)),(fit,query)]:
            reference=raw.loc[raw.customer_id.isin(a)]
            held=raw.loc[raw.customer_id.isin(b)]
            # Match the catalog reference's observation end to the query cohort.
            for cutoff,part in held.groupby('scoring_date'):
                hist=reference[reference.order_date<=cutoff-pd.Timedelta(days=14)]
                chunks.append(pd.concat([profile(hist,part,['product_id'],'sku'),
                                         profile(hist,part,['brand','category2','gender_product'],'family')],axis=1))
        pd.concat(chunks,verify_integrity=True).to_parquet(path)
    return c


def run(cache,work):
    work.mkdir(exist_ok=True,parents=True)
    c=create_features(cache,work);rows=[]
    for m in [2,3,4]:
        a=c.index[c.scoring_date.dt.month<m];b=c.index[c.scoring_date.dt.month==m]
        extra=pd.read_parquet(work/f'catalog_dates_matrix_{m}.parquet')
        cols=pd.read_csv(ROOT/f'research/results/tail_selected_{m}.csv').feature.tolist()
        core=pd.concat([pd.read_parquet(cache/f'features_{i}.parquet',columns=cols) for i in range(1,m+1)])
        tail=pd.read_parquet(ROOT/f'research/oof/tail_predictions_{m}.parquet').lightgbm_plain_half.loc[b]
        quarters=pd.read_parquet(ROOT/f'research/oof/scores_{m}.parquet').quarters.loc[b]
        n=pd.read_parquet(cache/f'features_{m}.parquet',columns=['n_orders']).n_orders
        for spec,regex in [('sku','^sku_'),('family','^family_'),('both','.')]:
            if m==4 and spec!=winner['variant']:continue
            x=core.join(extra.filter(regex=regex))
            model=booster('lightgbm').fit(x.loc[a],c.target.loc[a].astype(int))
            update=.5*(quarters+model.predict_proba(x.loc[b])[:,1])
            update.rename('score').to_frame().to_parquet(work/f'catalog_dates_{spec}_{m}.parquet')
            if m<4:
                for w in [.5,1.]:
                    rows.append(dict(month=m,variant=spec,weight=w,**compare(c.target.loc[b].astype(int),update*w+tail*(1-w),tail,n)))
                pd.DataFrame(rows).to_csv(ROOT/'research/results/catalog_dates.csv',index=False)
        if m==2:
            winner=max(rows,key=lambda v:v['ap20']);print('February',winner,flush=True)
        if m==3:
            march=next(v for v in rows if v['month']==3 and v['variant']==winner['variant'] and v['weight']==winner['weight'])
            print('March',march,flush=True)
            if winner['ap20']<=.7785233428848994 or march['ap20']<=.785118034003418:break
        if m==4:
            (update*winner['weight']+tail*(1-winner['weight'])).rename('score').to_frame().to_parquet(work/'catalog_dates_april.parquet')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=Path('.checkpoints/tail6_notebook'))
    parser.add_argument('--work',type=Path,default=Path('.checkpoints/round7'))
    args=parser.parse_args();run(args.cache,args.work)
