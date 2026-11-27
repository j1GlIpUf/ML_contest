"""Population shopping-transition probabilities and out-of-sample basket embeddings."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


def fit_language(reference, key):
    vocab = pd.Index(sorted(reference[key].unique()))
    n = len(vocab) + 1
    code = pd.Series(vocab.get_indexer(reference[key]) + 1, index=reference.index)
    by = code.groupby(reference.customer_id)
    sources, destinations, weights = [], [], []
    for lag in (1, 2, 3):
        prior = by.shift(lag)
        mask = prior.notna()
        sources.extend(prior[mask].astype(int))
        destinations.extend(code[mask])
        weights.extend(np.full(mask.sum(), 1 / lag))
    counts = coo_matrix((weights, (sources, destinations)), shape=(n,n)).tocsr()
    counts.sum_duplicates()
    popularity = np.bincount(code, minlength=n) + 1.
    popularity /= popularity.sum()
    mass = np.asarray(counts.sum(axis=1)).ravel()
    rows, cols = counts.nonzero()
    pmi = np.maximum(0., np.log((counts.data + .01) / (np.maximum(mass[rows],1) * popularity[cols])))
    positive = coo_matrix((pmi,(rows,cols)),shape=(n,n)).tocsr()
    with threadpool_limits(limits=2):
        encoder = TruncatedSVD(n_components=min(24,n-1),random_state=31)
        vectors = normalize(encoder.fit_transform(positive))
    return vocab, counts, popularity, mass, vectors


def language_features(receipts, key, fitted):
    vocab, matrix, popularity, masses, vectors = fitted
    frame = receipts[['customer_id','age',key]].copy()
    code = vocab.get_indexer(frame[key]) + 1
    frame['token'] = code
    by = frame.groupby('customer_id',sort=False)
    tail = by.cumcount(ascending=False)
    prev = by.token.shift().fillna(0).astype(int).to_numpy()
    count = np.asarray(matrix[prev,code]).ravel()
    frame['log_transition_lift'] = np.log((count + 8 * popularity[code]) / ((masses[prev] + 8) * popularity[code]))
    frame['last_step_cosine'] = (vectors[code] * vectors[prev]).sum(axis=1)
    frame['item_surprise'] = -np.log(popularity[code])
    # For each receipt, form the preference vector from strictly earlier baskets.
    emb = pd.DataFrame(vectors[code], index=frame.index)
    prev_profile = emb.groupby(frame.customer_id,sort=False).cumsum().to_numpy() - emb.to_numpy()
    frame['profile_cosine'] = (normalize(prev_profile) * vectors[code]).sum(axis=1)
    frame.loc[by.cumcount().eq(0), ['log_transition_lift','last_step_cosine','profile_cosine']] = np.nan
    fields=['log_transition_lift','last_step_cosine','profile_cosine','item_surprise']
    parts = []
    for label,mask in [('all',tail.ge(0)),('three',tail.lt(3)),('older',tail.ge(3)),('year',frame.age.lt(379))]:
        z=frame[mask].groupby('customer_id')[fields].agg(['mean','std','last'])
        z.columns = [f'language_{key}_{label}_{a}_{b}' for a,b in z.columns]
        parts.append(z)
    return pd.concat(parts,axis=1).astype('float32')
