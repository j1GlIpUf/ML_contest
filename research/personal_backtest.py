"""Test alternative renewal forecasts on earlier, fully observed customer quarters."""
import numpy as np
import pandas as pd


def _clocks(receipts, offset, horizon):
    data = receipts.loc[receipts.age.ge(offset + 14)].copy()
    data['age'] -= offset
    people = data.groupby('customer_id', sort=False)
    data['wait'] = -people.age.diff()
    data['step'] = people.cumcount()
    data['return'] = -data.groupby(['customer_id', 'mode']).age.diff()
    data['prev'] = people['mode'].shift()
    age = people.age.min()
    period = people.wait.median()
    latest = people.tail(1).set_index('customer_id')
    # Estimate time for a return to the dominant or last category and time
    # after a repeat of the latest category. Everything here predates offset.
    types = data.groupby(['customer_id', 'mode'], sort=False).agg(
        n=('age', 'size'), age=('age', 'min'), cycle=('return', 'median')).reset_index()
    dominant = types.sort_values(['n','age','mode'], ascending=[False,True,True]).drop_duplicates('customer_id').set_index('customer_id')
    next_step = data[data.prev.eq(data.customer_id.map(latest['mode']))].groupby('customer_id').wait.median()
    dates = pd.DataFrame({
        'median': period-age, 'mean': people.wait.mean()-age,
        'recent': people.tail(3).groupby('customer_id').wait.median()-age,
        'last_category': latest['return'].fillna(period)-age,
        'top': dominant.cycle.fillna(period)-dominant.age,
        'conditional': next_step.fillna(period)-age,
    })
    width = np.maximum(12., .35 * period.fillna(60).clip(15,180))
    # Smoothed probability of a noisy renewal date lying in the scoring window.
    from scipy.special import ndtr
    low = dates.div(width, axis=0)
    high = (dates-horizon).div(width, axis=0)
    scores = pd.DataFrame(ndtr(low)-ndtr(high), index=dates.index, columns=dates.columns)
    return scores, age


def personal_backtest(receipts, horizon=90):
    """Historical quarters use the same 14-day cutoff as the production snapshot."""
    receipts = receipts.sort_values(['customer_id','age','order_id'], ascending=[True,False,True])
    current, age = _clocks(receipts, 0, horizon)
    size = current.shape
    weights = np.zeros(size)
    error = np.zeros(size)
    bias = np.zeros(size)
    positive = np.zeros(len(current))
    total = np.zeros(len(current))
    # A date at offset 104 leaves the complete outcome inside visible history.
    for offset in range(horizon + 14, int(receipts.age.max())-14, 30):
        historical, previous_age = _clocks(receipts, offset, horizon)
        if historical.empty:
            continue
        actual = receipts.loc[receipts.age.le(offset) & receipts.age.gt(offset-horizon), 'customer_id'].unique()
        outcome = current.index.isin(actual)
        val = historical.reindex(current.index).to_numpy()
        old_age = previous_age.reindex(current.index).to_numpy()
        # Weight a past forecast by how closely inactivity matched today's.
        similarity = np.exp(-np.abs(np.log1p(old_age)-np.log1p(age.to_numpy())))
        similarity = np.nan_to_num(similarity)
        present = np.isfinite(val)
        w = similarity[:, None] * present
        residual = np.where(present, outcome[:,None]-val, 0.)
        weights += w
        error += w*residual**2
        bias += w*residual
        total += similarity
        positive += similarity*outcome
    # These smoothed statistics remain defined when the customer has little
    # or no eligible history. The number of relevant comparisons is explicit.
    brier = (error+1)/(weights+4)
    bias = bias/(weights+4)
    now = current.to_numpy()
    confidence = np.exp(-8*brier)*np.isfinite(now)
    normalizer = confidence.sum(1).clip(1e-7)
    output = pd.DataFrame(index=current.index)
    output['comparisons'] = total
    output['historic_rate'] = (positive+1)/(total+2)
    output['consensus'] = np.nansum(confidence*now,axis=1)/normalizer
    output['corrected'] = np.nansum(confidence*(now+bias).clip(0,1),axis=1)/normalizer
    for j, name in enumerate(current.columns):
        output[name+'_historic_loss'] = brier[:,j]
        output[name+'_bias'] = bias[:,j]
        output[name+'_adjusted'] = (now[:,j]+bias[:,j]).clip(0,1)
    return output.add_prefix('quarter_').astype('float32')
