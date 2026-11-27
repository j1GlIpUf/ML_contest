"""Learn low-rank interactions between discretized history features on OOF residuals."""
import numpy as np
from scipy.special import expit, logit


def factorization_forecast(fit, labels, oof, query, prior, seed=41):
    import torch
    from torch import nn
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    a,b=[],[]
    start=0
    for column in fit.columns:
        data=fit[column].to_numpy(dtype='float64')
        finite=data[np.isfinite(data)]
        if len(finite)<100 or np.min(finite)==np.max(finite):
            continue
        edges=np.unique(np.quantile(finite,np.linspace(.025,.975,15)))
        def encode(v):
            return start + np.where(np.isfinite(v),1+np.searchsorted(edges,v),0)
        a.append(encode(data));b.append(encode(query[column].to_numpy()))
        start+=len(edges)+2
    aa=torch.tensor(np.stack(a,axis=1),dtype=torch.long);bb=torch.tensor(np.stack(b,axis=1),dtype=torch.long)
    base=torch.tensor(logit(np.clip(np.asarray(oof),1e-5,1-1e-5)),dtype=torch.float32)
    target=torch.tensor(np.asarray(labels),dtype=torch.float32)
    width=aa.shape[1]
    class FM(nn.Module):
        def __init__(self):
            super().__init__()
            self.terms=nn.Embedding(start,13)
            nn.init.normal_(self.terms.weight,std=.012)
        def forward(self,idx):
            code=self.terms(idx)
            linear=code[:,:,0].sum(dim=1)/np.sqrt(width)
            factors=code[:,:,1:]
            interactions=(factors.sum(dim=1).square()-factors.square().sum(dim=1)).sum(dim=1)/(2*width)
            return linear+interactions
    model=FM();opt=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.08)
    snapshots=[]
    for epoch in range(12):
        for idx in torch.randperm(len(aa)).split(384):
            delta=model(aa[idx])
            weight=.4+.6*torch.sigmoid(base[idx])
            loss=nn.functional.binary_cross_entropy_with_logits(base[idx]+delta,target[idx],weight=weight)
            loss=loss+.035*delta.square().mean()
            opt.zero_grad();loss.backward();opt.step()
        if epoch in [2,4,7,11]:
            with torch.no_grad():
                snapshots.append(torch.cat([model(p) for p in bb.split(512)]).numpy())
    return expit(logit(np.clip(np.asarray(prior),1e-5,1-1e-5))+np.mean(snapshots,axis=0))
