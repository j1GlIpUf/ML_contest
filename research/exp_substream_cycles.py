"""Project recurring category cycles inside client-specific filtered shopping streams."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from research.exp_span_clocks import ROOT,compare_blocks
from research.motif_clocks import motif_clocks
from research.sequence_rules import baskets


def cycle_projection(frame):
    frame=frame.sort_values(['customer_id','age','order_id'],ascending=[True,False,True]).copy()
    g=frame.groupby('customer_id')
    last=g.age.min()
    outputs=[]
    for window in [3,5,8,15,75]:
        z=g.tail(window).copy()
        h=z.groupby('customer_id')
        z['x']=h.cumcount()
        z['y']=-z.age
        xc=z.x-h.x.transform('mean')
        yc=z.y-h.y.transform('mean')
        slope=(xc*yc).groupby(z.customer_id).sum()/(xc*xc).groupby(z.customer_id).sum().replace(0,np.nan)
        prediction=h.y.mean()+slope*(h.x.max()+1-h.x.mean())
        residual=yc-xc*z.customer_id.map(slope)
        mad=residual.abs().groupby(z.customer_id).median()
        out=pd.DataFrame({'next':prediction,'anchored':slope-last,'spread':mad,
            'end_error':residual.groupby(z.customer_id).last(),'window_conf':np.exp(-.5*((prediction-45)/(mad+12))**2)/(mad+12)})
        outputs.append(out.add_prefix(f'line_{window}_'))
    return pd.concat(outputs,axis=1)


def filtered_streams(b):
    family=b.groupby(['customer_id','mode']).age.agg(count='size',age='min').reset_index()
    family=family.sort_values(['count','age','mode'],ascending=[False,True,True])
    family['poprank']=family.groupby('customer_id').cumcount()
    family['recrank']=family.groupby('customer_id').age.rank(method='first').astype(int)-1
    b=b.merge(family[['customer_id','mode','poprank','recrank']],on=['customer_id','mode'],validate='many_to_one',sort=False)
    last=b.groupby('customer_id').age.min()
    pieces=[cycle_projection(b).add_prefix('trend_all_')]
    for label,stream in [('most_common',b[b.poprank==0]),('recent',b[b.recrank==0]),
                         ('common_pair',b[b.poprank<2]),('recent_pair',b[b.recrank<2]),('other',b[b.recrank>0])]:
        pieces.append(cycle_projection(stream).add_prefix('trend_'+label+'_'))
        motifs=motif_clocks(stream,depth=3)
        relevant=[f'motif_mode_{k}_{f}' for k in [1,2,3] for f in ['support','confidence','bigram_global_due','unigram_global_due','step_due','cycle_due','share']]
        pieces.append(motifs[relevant].add_prefix('filtered_'+label+'_'))
    # Binary sequences preserve other purchases and distinguish membership in
    # a frequently used category from the identity of the intervening basket.
    for label,code in [('usual',b.poprank==0),('latest',b.recrank==0),('mixed',b.ncats>1)]:
        x=b.assign(mode=code.astype(int))
        z=motif_clocks(x,depth=4)
        fields=[col for col in z if col.endswith(('_support','_confidence','_step_due','_cycle_due','_bigram_global_due','_order_period','_share'))]
        pieces.append(z[fields].add_prefix(f'presence_{label}_'))
    return pd.concat(pieces,axis=1).reindex(last.index).astype('float32')


def run(cache,work):
    work.mkdir(exist_ok=True,parents=True)
    c=pd.read_parquet(cache/'cohorts.parquet')
    orders=pd.read_parquet(ROOT/'orders.parquet').merge(pd.read_parquet(ROOT/'products.parquet'),on='product_id',validate='many_to_one')
    orders['order_date']=pd.to_datetime(orders.order_date)
    parts=[]
    for date,rows in c.groupby('scoring_date'):
        path=work/f'substreams_{date.month}.parquet'
        if not path.exists():filtered_streams(baskets(orders,rows.reset_index())).to_parquet(path)
        parts.append(pd.read_parquet(path))
    compare_blocks(cache,work,c,pd.concat(parts),{'trend':'^trend_','filtered':'^filtered_','presence':'^presence_'},'substream_cycles')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,default=Path('.checkpoints/tail6_notebook'))
    p.add_argument('--work',type=Path,default=Path('.checkpoints/round7'))
    args=p.parse_args();run(args.cache,args.work)
