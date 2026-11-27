"""Learn which past baskets make useful anchors for the next purchase date."""
import numpy as np
import pandas as pd


def purchase_anchors(receipts, history=24):
    """Produce one feature vector per recent basket. Padding is entirely zero."""
    b = receipts.sort_values(['customer_id', 'age', 'order_id'], ascending=[True, False, True]).reset_index(drop=True)
    people = b.groupby('customer_id', sort=False)
    types = b.groupby(['customer_id', 'mode'], sort=False)
    recency = people.age.transform('min')
    gap = -people.age.diff()
    period = gap.groupby(b.customer_id).transform('median')
    mean_gap = gap.groupby(b.customer_id).transform('mean')
    category_gap = -types.age.diff()
    type_period = category_gap.groupby([b.customer_id, b['mode']]).transform('median')
    category_age = types.age.transform('min')
    number = people.cumcount(ascending=False)
    size = people.age.transform('size')
    features = pd.DataFrame(index=b.index)
    features['observed'] = 1.
    features['position'] = np.log1p(number)/3
    features['total_orders'] = np.log1p(size)/3
    features['units'] = np.log1p(b.lines)/2
    features['time_since'] = np.log1p(b.age)/6
    features['category_share'] = types.age.transform('size')/size
    features['is_latest'] = number.eq(0)
    features['is_latest_category'] = types.cumcount(ascending=False).eq(0)
    features['matches_last'] = b['mode'].eq(people['mode'].transform('last'))
    # A past order can be reached after the last 1–4 observed categories.
    matches = pd.Series(True, index=b.index)
    for lag in range(1, 5):
        tail_category = b['mode'] if lag == 1 else people['mode'].shift(lag-1)
        last = tail_category.groupby(b.customer_id).transform('last')
        matches = matches & people['mode'].shift(lag).eq(last)
        features[f'context_{lag}'] = matches
        features[f'context_share_{lag}'] = matches.groupby(b.customer_id).transform('sum')/size
    for name, due in [('median', period-b.age), ('mean', mean_gap-b.age),
                      ('category', type_period-category_age), ('category_anchor', period-category_age),
                      ('next', gap-recency), ('personal', period-recency)]:
        features[name+'_timing'] = np.arcsinh(due/30).clip(-4, 4)/3
        features[name+'_center'] = np.exp(-.5*((due-45)/25)**2)
        features[name+'_inside'] = due.between(0,89)
    idx = people.size().index
    tensor = np.zeros((len(idx), history, len(features.columns)), dtype=np.float32)
    use = number.lt(history)
    tensor[idx.get_indexer(b.loc[use, 'customer_id']), number[use]] = features.loc[use].fillna(0).to_numpy('float32')
    return idx, tensor


def anchor_adjustment(fit, target, oof, query, prior, seed=47):
    """Fit a regularized correction using out-of-fold prior probabilities."""
    import torch
    from torch import nn
    from scipy.special import expit, logit

    torch.set_num_threads(3)
    torch.manual_seed(seed)
    fit = torch.as_tensor(fit)
    query = torch.as_tensor(query)
    base = torch.tensor(logit(np.clip(np.asarray(oof), 1e-5, 1-1e-5)), dtype=torch.float32)
    target = torch.tensor(np.asarray(target), dtype=torch.float32)

    class BasketModel(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.encode = nn.Sequential(nn.Linear(width, 40), nn.SiLU(), nn.Linear(40, 32), nn.SiLU())
            self.attention = nn.Linear(32, 4)
            self.predict = nn.Sequential(nn.Linear(128, 32), nn.SiLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.predict[-1].bias)
            nn.init.zeros_(self.predict[-1].weight)

        def forward(self, baskets):
            mask = baskets[:, :, :1] == 0
            features = self.encode(baskets)
            weights = self.attention(features).masked_fill(mask, -1e6).softmax(dim=1)
            pooled = weights.transpose(1, 2) @ features
            return self.predict(pooled.flatten(1)).squeeze(-1)

    model = BasketModel(fit.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.025)
    snapshots = []
    for epoch in range(14):
        for rows in torch.randperm(len(fit)).split(512):
            delta = model(fit[rows])
            loss = nn.functional.binary_cross_entropy_with_logits(base[rows]+delta, target[rows],
                                                                weight=.3+.7*torch.sigmoid(base[rows]))
            opt.zero_grad()
            (loss+.02*delta.square().mean()).backward()
            opt.step()
        if epoch in [5, 9, 13]:
            with torch.no_grad():
                snapshots.append(torch.cat([model(batch) for batch in query.split(512)]).numpy())
    correction = np.mean(snapshots, axis=0)
    return expit(logit(np.clip(np.asarray(prior), 1e-5, 1-1e-5))+.5*correction)
