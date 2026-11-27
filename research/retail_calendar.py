"""Express personal order intervals in units of observed market activity."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def retail_calendar(receipts, horizon):
    # The reference is the label-free cohort at its own cutoff. Include the blind
    # days and next quarter only by repeating the previous year's demand curve.
    parts=[]
    for key in ['all','mode','gender_product']:
        h=receipts[['customer_id','age','order_id']].copy()
        h['key']='all' if key=='all' else receipts[key].to_numpy()
        tables=[]
        for _,sub in h.groupby('key',sort=False):
            day=sub.age.to_numpy(dtype=int)
            length=max(746,day.max()+1)
            counts=np.bincount(day,minlength=length).astype(float)
            history=counts[14:]
            mean=max(history.mean(),.01)
            # A sparse department borrows a flat daily rate; all series average one.
            rates=(gaussian_filter1d(history,2,mode='nearest')+3)/(mean+3)
            cum=np.r_[0,np.cumsum(rates)]
            projected=np.arange(-horizon+1,14)+365
            forecast_rates=rates[projected-14]
            end_of_blind=forecast_rates[-13:].sum()
            future=forecast_rates[:-13].sum()
            q=sub.sort_values(['customer_id','age','order_id'],ascending=[True,False,True]).copy()
            q['age_in_sales']=cum[q.age.to_numpy()-14]+end_of_blind
            q['sale_rate']=rates[q.age.to_numpy()-14]
            g=q.groupby('customer_id')
            q['wait']=-g.age_in_sales.diff()
            q['typical_rate']=q.sale_rate.gt(1.4).astype(float)
            stats=g.agg(sales_age=('age_in_sales','last'), median=('wait','median'), avg=('wait','mean'),
                last_period=('wait','last'), max_demand=('sale_rate','max'), last_demand=('sale_rate','last'),
                mean_demand=('sale_rate','mean'), peak_share=('typical_rate','mean'), n=('age','size'))
            for measure in ['median','avg','last_period']:
                due=(stats[measure]-stats.sales_age)/max(1,future)
                stats[measure+'_quarter']=due
                stats[measure+'_center']=(due-.5).abs()
            tables.append(stats.reset_index())
        merged=pd.concat(tables,ignore_index=True)
        if key=='all':
            part=merged.set_index('customer_id')
        else:
            g=merged.groupby('customer_id')
            part=g[['median_center','avg_center','last_period_center']].min().add_prefix('nearest_')
            favorite=merged.sort_values(['n','sales_age'],ascending=[False,True]).drop_duplicates('customer_id').set_index('customer_id')
            recent=merged.sort_values('sales_age').drop_duplicates('customer_id').set_index('customer_id')
            subset=['sales_age','median_quarter','avg_quarter','last_demand','mean_demand','peak_share']
            part=part.join(favorite[subset].add_prefix('favorite_')).join(recent[subset].add_prefix('recent_'))
        parts.append(part.add_prefix('retail_'+key+'_'))
    return pd.concat(parts,axis=1).astype('float32')
