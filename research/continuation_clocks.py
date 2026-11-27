"""Joint clocks for every previously observed continuation of the current suffix."""
import numpy as np
import pandas as pd


def continuation_clocks(receipts):
    b=receipts.sort_values(['customer_id','age','order_id'],ascending=[True,False,True]).copy()
    b['state']=pd.Categorical(b['mode']).codes.astype('int64')+1
    radix=int(b.state.max())+1
    g=b.groupby('customer_id',sort=False)
    b['prefix']=b.state
    b['dest']=g.state.shift(-1)
    b['target_age']=g.age.shift(-1)
    b['wait']=b.age-b.target_age
    b['position']=g.cumcount()
    period=g.wait.median()
    age=g.age.min()
    n_orders=g.age.size()
    parts=[]
    for depth in range(1,6):
        if depth>1:
            b['prefix']=g.prefix.shift().fillna(0).astype('int64')*radix+b.state
        latest=g.prefix.transform('last')
        extract=b[(b.position>=depth-1)&b.prefix.eq(latest)&b.dest.notna()].copy()
        gg=extract.groupby(['customer_id','dest'],sort=False)
        extract['spacing']=-gg.target_age.diff()
        extract['steps']=gg.position.diff()
        d=gg.agg(n=('age','size'),anchor=('age','min'),completion=('target_age','min'),step=('wait','median'),
            last_step=('wait','last'),gap=('spacing','median'),hop=('steps','median'),oldest=('target_age','max')).reset_index()
        if d.empty:
            continue
        grp=d.groupby('customer_id',sort=False)
        pers=d.customer_id.map(period)
        rec=d.customer_id.map(age)
        d['share']=d.n/grp.n.transform('sum')
        d['global_completion']=pers-d.completion
        d['global_anchor']=pers-d.anchor
        d['return_due']=d.gap-d.completion
        d['end_to_next']=d.gap-d.anchor+d.step
        d['scaled']=pers*d.hop-d.completion
        d['count_due']=pers*((d.customer_id.map(n_orders)-depth)/d.n)-d.completion
        d['context_replay']=d.step-rec
        d['adjusted']=pers+(d.last_step-d.step)-rec
        st=grp.agg(candidates=('dest','size'),observations=('n','sum'))
        for variable in ['global_completion','global_anchor','return_due','end_to_next','scaled','count_due','context_replay','adjusted']:
            d['window']=np.exp(-.5*((d[variable]-45)/22)**2)*d.share
            st[variable+'_mass']=grp.window.sum(min_count=1)
            st[variable+'_most']=grp.window.max()
        for name,sort,asc in [('dominant',['n','completion','dest'],[False,True,True]),
                              ('last',['completion','dest'],[True,True])]:
            mode=d.sort_values(sort,ascending=asc).drop_duplicates('customer_id').set_index('customer_id')
            for k in ['share','global_completion','global_anchor','return_due','count_due','context_replay','adjusted','scaled','last_step','step','n']:
                st[name+'_'+k]=mode[k]
        parts.append(st.add_prefix(f'next_{depth}_'))
    return pd.concat(parts,axis=1).astype('float32')
