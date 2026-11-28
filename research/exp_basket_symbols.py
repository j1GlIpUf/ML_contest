"""Find recurring sequences of transaction sizes, promotions and relative spending."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from research.exp_span_clocks import ROOT,compare_blocks
from research.motif_clocks import motif_clocks
from research.sequence_rules import baskets


def symbols(orders,cohort):
    b=baskets(orders,cohort)
    fields=orders.groupby(['customer_id','order_id']).agg(amount=('paid_price','sum'),promo=('discount_percent','mean'),
                                                          loyalty=('loyalty_percent','mean'))
    b=b.join(fields,on=['customer_id','order_id'])
    group=b.groupby('customer_id')
    b['size']=np.minimum(b.lines,4)
    b['relative_spend']=np.floor((group.amount.rank(pct=True,method='average')*3).clip(upper=2.999)).astype(int)
    b['promotion']=pd.cut(b.promo,[-np.inf,0,25,50,np.inf],labels=False)
    b['benefit']=pd.cut(b.loyalty,[-np.inf,0,10,20,np.inf],labels=False)
    b['width']=np.minimum(b.ncats,3)
    b['basket_shape']=b['size']*4+b.width
    b['relative_cost_type']=b['mode'].astype(str)+'_'+b.relative_spend.astype(str)
    b['volume_type']=b['mode'].astype(str)+'_'+b['size'].astype(str)
    outputs=[]
    for key in ['size','basket_shape','relative_spend','promotion','benefit','relative_cost_type','volume_type']:
        info=motif_clocks(b,key,depth=3)
        fields=('support','share','cycle_due','step_due','unigram_global_due','bigram_global_due','same','past_likelihood',
                'gap_per_order_due','confidence','pooled_step_due')
        cols=[f'motif_{key}_{n}_{v}' for n in [1,2,3] for v in fields]
        outputs.append(info[cols].add_prefix('symbol_'))
    return pd.concat(outputs,axis=1).astype('float32')


def run(cache,work):
    work.mkdir(parents=True,exist_ok=True)
    c=pd.read_parquet(cache/'cohorts.parquet')
    orders=pd.read_parquet(ROOT/'orders.parquet').merge(pd.read_parquet(ROOT/'products.parquet'),on='product_id',validate='many_to_one')
    orders['order_date']=pd.to_datetime(orders.order_date)
    pieces=[]
    for date,rows in c.groupby('scoring_date'):
        path=work/f'symbols_{date.month}.parquet'
        if not path.exists():symbols(orders,rows.reset_index()).to_parquet(path)
        pieces.append(pd.read_parquet(path))
    compare_blocks(cache,work,c,pd.concat(pieces),{'size':'motif_(?:size|basket_shape)_','price':'motif_(?:relative_spend|promotion|benefit)_',
                   'joint':'motif_(?:relative_cost_type|volume_type)_'},'basket_symbols')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,default=Path('.checkpoints/tail6_notebook'))
    p.add_argument('--work',type=Path,default=Path('.checkpoints/round7'))
    args=p.parse_args();run(args.cache,args.work)
