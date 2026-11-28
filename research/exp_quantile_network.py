"""Compare a tabular neural net with linear and piecewise-linear numerical inputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from torch import nn
from research.rerank import compare

ROOT=Path(__file__).resolve().parents[1]


def encode(fit, query, bins):
    train_parts, test_parts=[],[]
    for col in fit:
        a=fit[col].to_numpy(); b=query[col].to_numpy()
        finite=a[np.isfinite(a)]
        if not len(finite):
            continue
        edges=np.unique(np.quantile(finite,np.linspace(0,1,bins+1)))
        if len(edges)<2:
            if np.isfinite(a).all():
                continue
            train_parts.append(np.isfinite(a)[:,None]); test_parts.append(np.isfinite(b)[:,None])
            continue
        def transform(values):
            present=np.isfinite(values)
            pos=np.interp(np.where(present,values,edges[0]),edges,np.arange(len(edges)))
            steps=np.clip(pos[:,None]-np.arange(len(edges)-1),0,1)
            return np.column_stack([steps*present[:,None],~present])
        train_parts.append(transform(a)); test_parts.append(transform(b))
    return [torch.from_numpy(np.concatenate(p,axis=1).astype('float32')) for p in [train_parts,test_parts]]


def neural_fit(x,y,z,seed):
    torch.manual_seed(seed)
    model=nn.Sequential(nn.Linear(x.shape[1],192),nn.LayerNorm(192),nn.SiLU(),nn.Dropout(.15),
                        nn.Linear(192,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.12),nn.Linear(96,1))
    opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.02)
    target=torch.tensor(y.to_numpy(),dtype=torch.float32); saved=[]
    for epoch in range(20):
        model.train()
        if epoch==12:
            for g in opt.param_groups: g['lr']=.0003
        for ids in torch.randperm(len(x)).split(384):
            loss=nn.functional.binary_cross_entropy_with_logits(model(x[ids]).flatten(),target[ids])
            opt.zero_grad();loss.backward();opt.step()
        if epoch in [7,11,15,19]:
            model.eval()
            with torch.no_grad():
                saved.append(torch.cat([model(v).flatten() for v in z.split(1024)]).numpy())
    return expit(np.mean(saved,axis=0))


def run(cache,work):
    torch.set_num_threads(3)
    work.mkdir(exist_ok=True,parents=True)
    c=pd.read_parquet(cache/'cohorts.parquet')
    rows=[]
    for m in [2,3,4]:
        a=c.index[c.scoring_date.dt.month<m]; b=c.index[c.scoring_date.dt.month==m]
        cols=pd.read_csv(ROOT/f'research/results/tail_selected_{m}.csv').feature.tolist()
        fit=pd.concat([pd.read_parquet(cache/f'features_{i}.parquet',columns=cols) for i in range(1,m)]).loc[a]
        query=pd.read_parquet(cache/f'features_{m}.parquet',columns=cols).loc[b]
        n=pd.read_parquet(cache/f'features_{m}.parquet',columns=['n_orders']).n_orders
        tail=pd.read_parquet(ROOT/f'research/oof/tail_predictions_{m}.parquet').lightgbm_plain_half.loc[b]
        for bins in [1,12]:
            dest=work/f'net_{bins}_{m}.parquet'
            if dest.exists(): p=pd.read_parquet(dest).score.loc[b]
            else:
                x,z=encode(fit,query,bins)
                p=pd.Series(np.mean([neural_fit(x,c.target.loc[a],z,seed) for seed in [26,42]],axis=0),index=b,name='score')
                p.to_frame().to_parquet(dest)
            if m<4:
                for weight in [.25,.5,1.]:
                    rows.append(dict(month=m,bins=bins,weight=weight,**compare(c.target.loc[b].astype(int),p*weight+tail*(1-weight),tail,n)))
                pd.DataFrame(rows).to_csv(ROOT/'research/results/quantile_network.csv',index=False)
        if m==2:
            winner=max(rows,key=lambda r:r['ap20'])
            print('February selection:',winner,flush=True)
        if m==3:
            print('March:',[r for r in rows if r['month']==3 and r['bins']==winner['bins'] and r['weight']==winner['weight']],flush=True)
            if not all(r['ap20']>v for r,v in zip([r for r in rows if r['bins']==winner['bins'] and r['weight']==winner['weight']],[.7785233428848994,.785118034003418])):
                break


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,default=Path('.checkpoints/tail6_notebook'))
    p.add_argument('--work',type=Path,default=Path('.checkpoints/round7'))
    args=p.parse_args();run(args.cache,args.work)
