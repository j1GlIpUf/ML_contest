"""Run after building the input tables using lamoda_tail.ipynb."""
from pathlib import Path
import json
import hashlib
from importlib.metadata import version
import numpy as np
import pandas as pd
from research.rerank import compare
from research.tail_models import experiment
from research.tail_analysis import blends

CACHE=Path('.checkpoints/tail6')
RESULTS=Path('research/results')


def run(cache=CACHE):
    cohorts=pd.read_parquet(cache/'cohorts.parquet')
    dates=cohorts.scoring_date
    segments=pd.read_parquet(cache/'segments.parquet')
    january=pd.concat([pd.read_parquet(cache/f'january_fold_{i}.parquet') for i in range(5)])
    scores=pd.concat([january.drop(columns='fold'),*[pd.read_parquet(cache/f'scores_{m}.parquet') for m in [2,3,4]]]).loc[cohorts.index]
    assert scores.index.is_unique and scores.notna().all().all()
    features=pd.concat([pd.read_parquet(cache/f'features_{m}.parquet') for m in range(1,5)]).loc[cohorts.index]
    digest=hashlib.sha256()
    for source in ['run_tail','tail_analysis','tail_models','tail_features']:
        digest.update((Path(__file__).parent/(source+'.py')).read_bytes())
    for library in ['lightgbm','catboost','scikit-learn','pandas']:
        digest.update(version(library).encode())
    import xgboost
    digest.update(xgboost.__version__.encode())
    for frame in [features,scores,segments,cohorts]:
        digest.update(repr(frame.columns.tolist()).encode())
        digest.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
    fingerprint=digest.hexdigest()
    rows=[]
    winner=None
    improves_both=True
    for month in [2,3,4]:
        fit=dates.index[dates.dt.month<month]
        query=dates.index[dates.dt.month==month]
        ref=scores.loc[query,'quarters']
        path=cache/f'predictions_{month}.parquet'
        stamp=path.with_suffix('.sha256')
        if path.exists() and stamp.exists() and stamp.read_text()==fingerprint:
            predictions=pd.read_parquet(path).loc[query]
        else:
            predictions,importance=experiment(features,segments,dates,cohorts.target,scores,fit,query)
            predictions=predictions.join(blends(scores.loc[query]))
            predictions.to_parquet(path.with_suffix('.tmp'))
            path.with_suffix('.tmp').replace(path)
            stamp.write_text(fingerprint)
            importance.rename('gain').rename_axis('feature').to_csv(RESULTS/f'tail_selected_{month}.csv')
        if month==4:
            return predictions[winner].rename('score'),winner
        for name,p in predictions.items():
            rows.append(dict(month=month,model=name,**compare(cohorts.target.loc[query].astype(int),p,ref,segments.n_orders)))
        results=pd.DataFrame(rows)
        results.to_csv(RESULTS/'tail_models.csv',index=False)
        if month==2:
            winner=results.loc[results.ap20.idxmax(),'model']
            (RESULTS/'tail_february_choice.json').write_text(json.dumps({'model':winner,'rule':'highest February AP among the planned candidates; require both months above quarters'},indent=2)+'\n')
        print(month,winner,results[results.model.eq(winner)].to_string(index=False),flush=True)
        record=results.iloc[-len(predictions.columns):].set_index('model').loc[winner]
        control_ap=compare(cohorts.target.loc[query].astype(int),ref,ref,segments.n_orders)['ap20']
        improves_both = improves_both and record.ap20 > control_ap
        if month == 3 and not improves_both:
            april=dates.index[dates.dt.month.eq(4)]
            reference=pd.read_csv('submission_quarters.csv',float_precision='round_trip').set_index('customer_id').score
            return reference.loc[april],'quarters'
    raise RuntimeError('Missing April cohort.')
