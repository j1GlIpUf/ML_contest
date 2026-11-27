"""Fit a regular purchase phase separately for each category of each customer."""
import numpy as np
import pandas as pd


def category_phase(receipts):
    dates = receipts[['customer_id','mode','age']].drop_duplicates().sort_values(['customer_id','mode','age'])
    grouped = dates.groupby(['customer_id','mode'], sort=False)
    counts = grouped.size().to_numpy()
    stops = counts.cumsum()
    starts = stops-counts
    ages = dates.age.to_numpy()
    last = ages[starts]
    span = ages[stops-1]-last
    since_last = ages-np.repeat(last,counts)
    # Short cycles fit arbitrary noisy histories more often. Compare inlier
    # proportions to their expected frequency under a uniform phase.
    best = np.full(len(counts),-np.inf)
    picked = np.full((len(counts),5), np.nan)
    for period in range(21, 395, 3):
        residual = np.mod(since_last+period/2,period)-period/2
        sigma = np.clip(period*.09,3,14)
        evidence = np.exp(-.5*(residual/sigma)**2)
        hit = (np.add.reduceat(evidence,starts)-1)/(counts-1).clip(1)
        chance = np.sqrt(2*np.pi)*sigma/period
        lift = (hit-chance) * np.sqrt(counts-1)
        coherent = (counts>=3) & (span >= 1.5*period)
        phase_shift = np.add.reduceat(evidence*residual,starts)/np.add.reduceat(evidence,starts)
        # The estimated phase is expressed in days from the scoring date.
        next_purchase = np.mod(-last-phase_shift,period)
        center = np.abs(np.mod(45-next_purchase+period/2,period)-period/2)
        eligible = coherent & (lift>best)
        vals = np.column_stack([np.full(len(counts),period),lift,next_purchase,center,hit])
        picked[eligible] = vals[eligible]
        best[eligible] = lift[eligible]
    phase = pd.DataFrame(picked, index=grouped.size().index,
                         columns=['period','evidence','next','center','fit']).reset_index()
    phase['n'] = counts
    phase['age'] = last
    phase['cycles_since'] = phase.age/phase.period
    phase['window'] = phase.next.lt(90).where(phase.period.notna()).astype(float)
    keys = ['period','evidence','next','center','n','age','cycles_since','fit','window']
    parts = []
    for label, ranking, directions in [('strongest',['evidence','n','age'],[False,False,True]),
                                       ('latest',['age','n'],[True,False]),
                                       ('frequent',['n','age'],[False,True])]:
        selected = phase.sort_values(ranking,ascending=directions).drop_duplicates('customer_id')
        parts.append(selected.set_index('customer_id')[keys].add_prefix('phase_'+label+'_'))
    return pd.concat(parts,axis=1).astype('float32')
