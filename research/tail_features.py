"""Receipt-level features for detecting false positives among frequent buyers."""
import numpy as np
import pandas as pd


def warning_features(orders, cohort, core):
    h=orders.merge(cohort[['customer_id','scoring_date']],on='customer_id',validate='many_to_one')
    h['age']=(h.scoring_date-h.order_date).dt.days
    h=h.loc[h.age>=14]
    receipts=h.groupby(['customer_id','age','order_id']).agg(spend=('paid_price','sum'),units=('product_id','size'),
        discount=('discount_percent','mean'),loyalty=('loyalty_percent','mean')).reset_index()
    receipts=receipts.sort_values(['customer_id','age','order_id'],ascending=[True,False,True])
    g=receipts.groupby('customer_id',sort=False)
    receipts['gap']=-g.age.diff()
    year=receipts[receipts.age<=378].groupby('customer_id')
    out=pd.DataFrame(index=core.index)
    due=-core.rhythm_order_all_median_delta
    out['blind_long_cycle']=((core.recency+due)>70)&due.between(-13,-1)
    for width in [45,60]:
        recent=receipts[receipts.age.le(13+width)].groupby('customer_id')
        for field in ['spend','units']:
            out[f'{field}_{width}_pace']=(recent[field].sum().reindex(out.index,fill_value=0)/width)/(year[field].sum()/365).replace(0,np.nan)
    last=g.tail(1).set_index('customer_id')
    receipts['previous_spend']=g.spend.shift()
    prev=g.tail(1).set_index('customer_id').previous_spend
    out['spend_ratio']=last.spend/prev.replace(0,np.nan)
    out['session_finished']=(core.session_median_wait_last_size>=core.session_median_wait_normal_size)&out.spend_ratio.lt(1)
    for width in [180,365]:
        maximum=receipts.loc[receipts.age<=width+13].groupby('customer_id').loyalty.max()
        out[f'loyalty_drop_{width}']=last.loyalty-maximum
    recent=last.gap
    receipts['previous_gap']=g.gap.shift()
    receipts['older_gap']=g.gap.shift(2)
    final=g.tail(1).set_index('customer_id')
    out['gap_ratio_prev']=(recent+1)/(final.previous_gap+1)
    out['gap_ratio_typical']=(recent+1)/(g.gap.median()+1)
    out['gaps_growing']=(recent>final.previous_gap)&(final.previous_gap>final.older_gap)
    bins=pd.cut(receipts.gap,[-1,0,3,7,14,30,60,120,365,np.inf])
    counts=receipts.groupby([receipts.customer_id,bins],observed=True).size()
    freq=counts/counts.groupby(level=0).transform('sum')
    out['gap_entropy']=(-freq*np.log(freq)).groupby(level=0).sum()
    out['burst_share']=receipts.gap.le(3).where(receipts.gap.notna()).groupby(receipts.customer_id).mean()
    out['discount_above_median']=(receipts.discount>receipts.discount.median()).groupby(receipts.customer_id).mean()
    out['discount_above_p90']=(receipts.discount>receipts.discount.quantile(.9)).groupby(receipts.customer_id).mean()
    return out.add_prefix('warning_').astype('float32')
