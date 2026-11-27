"""Transfer calendar forecast errors among customers with matching category patterns."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def history_signature(receipts):
    b = receipts.sort_values(['customer_id','age','order_id'],ascending=[True,False,True])
    people = b.groupby('customer_id',sort=False)
    ages = pd.DataFrame({j:people.age.shift(j) for j in range(6)}).groupby(b.customer_id).tail(1)
    states = pd.DataFrame({j:people['mode'].shift(j) for j in range(6)}).groupby(b.customer_id).tail(1)
    ids = people.tail(1).customer_id
    ages.index = states.index = pd.Index(ids, name='customer_id')
    size = people.size().reindex(ages.index)
    interval = (-people.age.diff()).groupby(b.customer_id).median().reindex(ages.index)
    # Normalize names within each suffix. ABAC thus matches two shoppers
    # regardless of the literal categories behind A, B and C.
    role = np.zeros((len(ages), 6),dtype=np.int64)
    for j in range(1,6):
        equal = states.iloc[:,:j].eq(states[j],axis=0).to_numpy()
        role[:,j] = np.where(equal.any(1), equal.argmax(1), j)
    frame = pd.DataFrame(index=ages.index)
    for length in [3, 4, 6]:
        code = role[:,:length] @ (7**np.arange(length))
        frame[f'shape{length}'] = np.where(size>=length, code, -1)
    frame['log_count'] = np.log1p(size)/1.2
    frame['recency'] = np.log1p(ages[0])/2
    frame['pace'] = np.log1p(interval)/2
    for k in range(4):
        due = interval-ages[k]
        frame[f'phase{k}'] = np.arcsinh((due-45)/22)/2
        frame[f'center{k}'] = np.exp(-.5*((due-45)/25)**2)
    return frame


def neighbor_correction(training, y, oof, query, prior, k=48):
    columns = [c for c in training.columns if not c.startswith('shape')]
    def space(frame, scores):
        a = frame[columns].fillna(-4.).to_numpy()
        return np.column_stack([a, 2*scores.to_numpy()])

    train = space(training,oof)
    test = space(query,prior)
    residual = (y-oof).to_numpy()
    numerator = np.zeros(len(query))
    denominator = np.zeros(len(query))
    for length in [3,4,6]:
        key = f'shape{length}'
        by_shape = training.groupby(key,sort=False).indices
        for shape, rows in query.groupby(key,sort=False).indices.items():
            members = by_shape.get(shape)
            if shape<0 or members is None or len(members)<12:
                continue
            tree = cKDTree(train[members])
            distance, selected = tree.query(test[rows], k=min(k,len(members)))
            weight = np.exp(-distance**2/1.5)
            numerator[rows] += (weight*residual[members[selected]]).sum(axis=1)
            denominator[rows] += weight.sum(axis=1)
    delta = numerator/(denominator+40)
    return np.clip(prior + .5*delta,0,1)
