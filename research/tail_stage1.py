"""Cross-fit all five stage-one forecasts with the original fold-local models."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from research.motif_model import motif_experts
from research.window_models import forecast
from research.behavior_experts import gate_features, learned_experts
from research.context_backoff import backoff_features


def stage_one(fit, query, labels, dates, core, transitions, extras, quarters, phase, recipes):
    y = labels.loc[fit]
    chosen = list(recipes['select_by_gain'](core.loc[fit], y, 110))
    rank_predictions = []
    for columns in (transitions, core.columns):
        predictions = recipes['predictions'](core.loc[fit, columns], y, dates.loc[fit], core.loc[query, columns])
        rank_predictions.append(predictions['Смесь с ранжировщиком 60%'])
    regional = recipes['regional_forecast'](core.loc[fit], y, core.loc[query])
    first = .25*(rank_predictions[0]+rank_predictions[1])+.5*regional
    features = core.join(extras)
    general_cols = chosen + extras.columns.tolist()
    general = forecast(features.loc[fit, general_cols], y, features.loc[query, general_cols])
    local = motif_experts(features.loc[fit], y, features.loc[query], chosen)
    motifs = .25*first+.25*general+.5*local
    history = extras.filter(regex='^motif_mode_')
    gates = gate_features(core, history)
    features = core[chosen].join(history)
    group = learned_experts(features.loc[fit], y, features.loc[query], gates.loc[fit], gates.loc[query])
    features = features.join(backoff_features(core, history))
    scores = {'motifs': motifs, 'regional': regional}
    for name, block in [('context', None), ('quarters', quarters), ('phase', phase)]:
        if block is not None:
            features = features.join(block)
        pred = forecast(features.loc[fit], y, features.loc[query])
        scores[name] = .5*motifs+.25*(group+pred)
    return pd.DataFrame(scores, index=query)


def january_oof(labels, dates, core, transitions, extras, quarters, phase, recipes, directory):
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    january = dates.index[dates.eq(dates.min())]
    core, extras = core.loc[january], extras.loc[january]
    splits = StratifiedKFold(5, shuffle=True, random_state=26)
    folds = []
    for i, (a,b) in enumerate(splits.split(january, labels.loc[january])):
        path = directory/f'january_fold_{i}.parquet'
        if not path.exists():
            query = january[b]
            scores = stage_one(january[a], query, labels, dates, core, transitions, extras,
                               quarters.loc[january], phase.loc[january], recipes)
            scores['fold'] = i
            scores.to_parquet(path.with_suffix('.tmp'))
            os.replace(path.with_suffix('.tmp'),path)
        fold = pd.read_parquet(path)
        assert fold.index.equals(january[b]) and np.isfinite(fold.to_numpy()).all()
        folds.append(fold)
        print('January fold',i+1,'of 5 complete',flush=True)
    return pd.concat(folds).loc[january]
