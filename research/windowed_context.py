"""Recover the longest recurrent category pattern using only a recent history."""
import numpy as np
import pandas as pd

from research.motif_clocks import motif_clocks


def windowed_context(receipts):
    ordered = receipts.sort_values(['customer_id', 'age', 'order_id'], ascending=[True, False, True])
    rank = ordered.groupby('customer_id', sort=False).cumcount(ascending=False)
    frames = []
    fields = ['support', 'confidence', 'repeat_p', 'pooled_step_due', 'gap_per_order_due',
              'unigram_age', 'unigram_due', 'bigram_due', 'cycle_due', 'order_period']
    for name, eligible in [('year', ordered.age.le(378)), ('half_year', ordered.age.le(195)),
                           ('eight_orders', rank.lt(8))]:
        window = ordered.loc[eligible]
        motifs = motif_clocks(window)
        groups = window.groupby('customer_id', sort=False)
        mean_wait = -groups.age.diff().groupby(window.customer_id).mean()
        across_depths = np.stack([
            motifs[[f'motif_mode_{depth}_{field}' for field in fields]].to_numpy()
            for depth in range(1, 5)
        ], axis=1)
        out = pd.DataFrame(index=motifs.index)
        for support in [1, 2]:
            available = across_depths[:, :, 0] >= support
            longest = np.where(available, np.arange(4), -1).max(axis=1)
            chosen = across_depths[np.arange(len(motifs)), longest.clip(0)].copy()
            chosen[longest < 0] = np.nan
            prefix = f'{name}_{support}_'
            for col, values in zip(fields, chosen.T):
                out[prefix+col] = values
            out[prefix+'depth'] = longest + 1
            out[prefix+'mean_due_next'] = mean_wait - out[prefix+'unigram_age']
        frames.append(out)
    return pd.concat(frames, axis=1).astype('float32')
