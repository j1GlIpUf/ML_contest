"""Transfer out-of-sample ranking residuals between customers with similar purchases."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit, logit
from sklearn.preprocessing import normalize


def catalog_vectors(orders, cohort):
    h=orders.merge(cohort[['customer_id','scoring_date']],on='customer_id',validate='many_to_one')
    age=(h.scoring_date-h.order_date).dt.days
    h=h.loc[age.ge(14)].copy()
    h['recency_weight']=.3+.7*np.exp(-(age.loc[h.index]-14)/180)
    h['family']=h.brand+'|'+h.category2+'|'+h.gender_product
    ix=pd.Index(cohort.customer_id).get_indexer(h.customer_id)
    blocks=[]
    for key in ['product_id','family']:
        tokens=pd.Categorical(h[key])
        block=csr_matrix((h.recency_weight.to_numpy(),(ix,tokens.codes.astype(int))),
                         shape=(len(cohort),len(tokens.categories)),dtype=np.float32)
        block.data=np.log1p(block.data)
        blocks.append(block)
    return hstack(blocks,format='csr')


def neighbor_adjustment(fit, labels, oof, query, prior, k=64):
    df=np.bincount(fit.indices,minlength=fit.shape[1])
    idf=np.log1p(len(labels)/(df+1))
    idf[(df < 2) | (df > .05 * len(labels))] = 0
    a=normalize(fit.multiply(idf)).tocsr();a.eliminate_zeros()
    b=normalize(query.multiply(idf)).tocsr();b.eliminate_zeros()
    residual=labels.to_numpy()-np.asarray(oof)
    variance=np.asarray(oof)*(1-np.asarray(oof))
    shifts=np.zeros(len(prior))
    for start in range(0,b.shape[0],128):
        matches=(b[start:start+128]@a.T).tocsr()
        for j in range(matches.shape[0]):
            left,right=matches.indptr[j:j+2]
            indices=matches.indices[left:right];weights=matches.data[left:right]
            if len(indices)>k:
                take=np.argpartition(weights,-k)[-k:]
                indices=indices[take];weights=weights[take]
            w=weights**2
            shifts[start+j]=(w@residual[indices])/(5+w@variance[indices])
    return expit(logit(np.clip(np.asarray(prior),1e-5,1-1e-5))+shifts.clip(-1,1))
