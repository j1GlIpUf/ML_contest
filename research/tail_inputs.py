"""Build the tail experiment's feature tables from the executed context notebook."""
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from research.category_phase import category_phase
from research.trajectory import trajectory_features
from research.sequence_rules import baskets
from research.tail_stage1 import january_oof, stage_one
from research.tail_features import warning_features
from research.tail_analysis import diagnostics, blends, local_oracles
from research.rerank import compare


def prepare(ns,cache,rebuild_scores=False):
    cache=Path(cache)
    cache.mkdir(exist_ok=True,parents=True)
    train,test,orders=ns['train'],ns['test'],ns['orders']
    c=pd.concat([train,test]).set_index('customer_id')
    core=ns['core'].loc[c.index]
    cats=orders.merge(c[['scoring_date']],on='customer_id',validate='many_to_one')
    mono=cats[(cats.scoring_date-cats.order_date).dt.days>=14].groupby('customer_id').category.nunique().eq(1)
    seg=pd.DataFrame({'n_orders':core.n_orders,'mono':mono,'due_sign':np.sign(-core.rhythm_order_all_median_delta).fillna(0)}).loc[c.index]
    seg['length']=pd.cut(seg.n_orders,[0,1,3,6,15,np.inf],labels=['1','2-3','4-6','7-15','16+'])
    seg.to_parquet(cache/'segments.parquet')
    c.to_parquet(cache/'cohorts.parquet')
    phases=[]
    for cutoff,rows in c.groupby('scoring_date'):
        cohort=rows.reset_index()
        histories=baskets(orders,cohort)
        phase=category_phase(histories)
        phases.append(phase)
        horizon=(cutoff+pd.DateOffset(months=3)-cutoff).days
        frame=core.loc[rows.index].astype('float32')
        for block in [ns['history'],trajectory_features(histories,horizon),ns['personal'],phase,
                      warning_features(orders,cohort,frame)]:
            frame=frame.join(block)
        frame.astype('float32').to_parquet(cache/f'features_{cutoff.month}.parquet')
    phases=pd.concat(phases)
    if rebuild_scores:
        january_oof(c.target.dropna().astype(int),c.scoring_date,core,ns['transition_train'].columns,
                    ns['extra_motifs'],ns['personal'],phases,ns,cache)
        for cutoff,rows in list(c.groupby('scoring_date'))[1:]:
            fit=c.index[c.scoring_date<cutoff]
            scores=stage_one(fit,rows.index,c.target.dropna().astype(int),c.scoring_date,core,
                ns['transition_train'].columns,ns['extra_motifs'],ns['personal'],phases,ns)
            scores.to_parquet(cache/f'scores_{cutoff.month}.parquet')
    else:
        for path in sorted(Path('research/oof').glob('scores_*.parquet')):
            if path.resolve() != (cache/path.name).resolve():
                shutil.copyfile(path,cache/path.name)
    diagnostic_reports(cache)


def diagnostic_reports(cache):
    c=pd.read_parquet(cache/'cohorts.parquet')
    seg=pd.read_parquet(cache/'segments.parquet')
    losses,bands,mixes,oracles,spread=[],[],[],[],[]
    for month in [2,3]:
        s=pd.read_parquet(cache/f'scores_{month}.parquet')
        y=c.target.loc[s.index].astype(int)
        ranked=s.quarters.sort_values(ascending=False,kind='stable').index
        for lo,hi in [(1,1500),(1501,5000),(5001,10000)]:
            subset=s.loc[ranked[lo-1:hi]]
            disagreement=subset.std(axis=1,ddof=0)
            hits=y.loc[subset.index].astype(bool)
            spread.append(dict(month=month,first=lo,last=hi,model_sd_buyers=disagreement[hits].mean(),
                model_sd_nonbuyers=disagreement[~hits].mean(),
                min_pairwise_spearman=subset.corr(method='spearman').min().min()))
        oracles.append(local_oracles(y,s.quarters).assign(month=month))
        a,b=diagnostics(y,s.quarters,seg)
        losses.append(a.assign(month=month));bands.append(b.assign(month=month))
        for name,p in blends(s).assign(quarters=s.quarters).items():
            mixes.append(dict(month=month,model=name,**compare(y,p,s.quarters,seg.n_orders)))
        for suffix,frame in [('disagreement',pd.DataFrame(spread)),('oracles',pd.concat(oracles)),('loss',pd.concat(losses)),('segments',pd.concat(bands)),('blends',pd.DataFrame(mixes))]:
            frame.to_csv(f'research/results/tail_{suffix}.csv',index=False)
