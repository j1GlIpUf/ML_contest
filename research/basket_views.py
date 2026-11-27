"""Sequence evidence under different observable definitions of a multi-item order."""
import pandas as pd

from research.motif_clocks import motif_clocks


def basket_views(orders, cohort):
    h = orders.merge(cohort[['customer_id','scoring_date']],on='customer_id',validate='many_to_one')
    h['age'] = (h.scoring_date-h.order_date).dt.days
    h = h.loc[h.age.ge(14)].copy()
    ids = ['customer_id','order_id','age']
    by = h.groupby(ids + ['category'],sort=False).agg(
        units=('paid_price','size'), value=('paid_price','sum')).reset_index()
    price = h.sort_values(['paid_price','product_id'],ascending=[False,True],kind='stable')
    views = {}
    for label, frame in [('first_item',h),('low_sku',h.sort_values('product_id',kind='stable')),
                         ('costliest',price),('spend',by.sort_values(['value','category'],ascending=[False,True])),
                         ('lex',by.sort_values('category',kind='stable'))]:
        views[label] = frame.drop_duplicates(ids)[ids+['category']].rename(columns={'category':'mode'})
    ordered = h.sort_values(['customer_id','age','order_id','product_id'],ascending=[True,False,True,True],kind='stable')
    for label, frame in [('all_items',ordered),('category_visits',ordered.drop_duplicates(ids+['category']))]:
        frame = frame[[*ids,'category']].copy().rename(columns={'category':'mode'})
        # A unique within-customer sequence position preserves the order of
        # the item observations when several of them share a date and receipt.
        frame['order_id'] = frame.groupby('customer_id',sort=False).cumcount()
        views[label] = frame
    parts = []
    for name, frame in views.items():
        stats = motif_clocks(frame,depth=3)
        fields = ['support','share','past_likelihood','branches','repeat_p','confidence','same',
                  'pooled_step_due','bigram_global_due','unigram_global_due']
        cols = [f'motif_mode_{i}_{key}' for i in [1,2,3] for key in fields]
        parts.append(stats[cols].add_prefix(name+'_'))
    return pd.concat(parts,axis=1).reindex(cohort.customer_id).astype('float32')
