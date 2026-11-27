"""Population purchase and dropout rates fitted from historical shopping days."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln, hyp2f1


def exposure(receipts, start, span, category=False):
    keys = ['customer_id', 'mode'] if category else ['customer_id']
    ages = receipts.loc[receipts.age.between(start, start + span - 1), [*keys, 'age']].drop_duplicates()
    days = ages.groupby(keys, observed=True).age.agg(['size', 'max', 'min'])
    return pd.DataFrame({'x': days['size'] - 1, 'tx': (days['max'] - days['min']) / 7,
                         'T': (days['max'] - start) / 7}, index=days.index)


def fit_population(observations):
    unique, frequency = np.unique(observations[['x', 'tx', 'T']].to_numpy(), axis=0, return_counts=True)
    x, tx, T = unique.T
    def objective(log_parameters):
        r, alpha, a, b = np.exp(log_parameters)
        both = gammaln(r + x) - gammaln(r) + r * np.log(alpha) + betaln(a, b + x) - betaln(a, b)
        active = -(r + x) * np.log(alpha + T)
        dead = np.full(len(x), -np.inf)
        yes = x > 0
        dead[yes] = np.log(a / (b + x[yes] - 1)) - (r + x[yes]) * np.log(alpha + tx[yes])
        return -np.average(both + np.logaddexp(active, dead), weights=frequency)
    params = minimize(objective, np.log([.7, 10., 1., 4.]), method='L-BFGS-B',
                      bounds=[(-5, 5), (-5, 9), (-5, 6), (-5, 8)])
    if not params.success:
        raise RuntimeError(params.message)
    return np.exp(params.x)


def purchase_probability(observations, params, delay, length):
    r, alpha, a, b = params
    x, tx, T = observations[['x', 'tx', 'T']].to_numpy().T
    shape = r + x
    rate = alpha + T
    dead_log_odds = np.log(a / np.maximum(b + x - 1, 1e-12)) + shape * np.log(rate / (alpha + tx))
    alive = np.where(x > 0, expit(-dead_log_odds), 1.)
    delay, length = delay / 7., length / 7.
    # Integrate exp(-lambda*p*delay) * (1-exp(-lambda*length)) over the
    # customer's Gamma purchase rate and Beta dropout probability.
    survival = hyp2f1(shape, a, a + b + x, -delay / rate)
    empty = (rate / (rate + length)) ** shape * hyp2f1(shape, a, a + b + x, -delay / (rate + length))
    return pd.DataFrame({'alive': alive, 'rate': shape / rate, 'prob': alive * (survival - empty)},
                        index=observations.index)


def lifetime_features(receipts, reference, horizon):
    parts = []
    for span in [365, 730]:
        for category in [False, True]:
            params = fit_population(exposure(reference, 14, span, category))
            for offset in [0, 90]:
                z = exposure(receipts, 14 + offset, span - offset, category)
                scores = purchase_probability(z, params, 13 if offset == 0 else 0,
                                              horizon if offset == 0 else 90)
                if offset:
                    keys = ['customer_id', 'mode'] if category else ['customer_id']
                    observed = receipts[receipts.age.between(14, 103)].groupby(keys).size()
                    scores['surprise'] = observed.reindex(scores.index, fill_value=0).gt(0).astype(float) - scores.prob
                if category:
                    scores['log_no_buy'] = np.log1p(-scores.prob.clip(upper=1 - 1e-12))
                    g = scores.groupby(level=0)
                    aggregated = g[['alive', 'rate', 'prob']].max()
                    aggregated['parallel'] = -np.expm1(g.log_no_buy.sum())
                    if offset:
                        aggregated['surprise'] = g.surprise.mean()
                    scores = aggregated
                parts.append(scores.add_prefix(f'lt_{span}_{int(category)}_{offset}_'))
    return pd.concat(parts, axis=1).astype('float32')
