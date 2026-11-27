"""Compare the observed last purchase categories with alternative orderings in older baskets."""
from itertools import permutations

import numpy as np
import pandas as pd


def order_evidence(receipts):
    b = receipts.sort_values(['customer_id', 'age', 'order_id'], ascending=[True, False, True])
    ids = b.customer_id.to_numpy()
    code = pd.Categorical(b['mode']).codes.astype('int64') + 1
    base = code.max() + 1
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ids)])
    origin = np.repeat(starts, counts)
    ends = origin + np.repeat(counts, counts) - 1
    ix = np.arange(len(ids))
    final_age = b.age.to_numpy()[ends]
    recency = final_age[starts]
    gap = -b.groupby('customer_id').age.diff()
    period = gap.groupby(b.customer_id, sort=False).median().to_numpy()
    age = b.age.to_numpy()
    table = {}
    for offsets in [(0, 1), (0, 1, 2), (0, 2), (0, 1, 3), (0, 2, 3), (0, 1, 2, 3)]:
        length = len(offsets)
        label = ''.join(map(str, offsets))
        target = np.column_stack([code[np.maximum(origin, ends - k)] for k in offsets])
        observed = sum(code[np.maximum(ix - k, origin)] * base ** j for j, k in enumerate(offsets))
        allowed = ((ix - origin) >= max(offsets)) & (ix < ends)
        for perm in permutations(range(length)):
            if length == 4 and perm not in [(0, 1, 2, 3), (3, 2, 1, 0), (1, 0, 2, 3), (3, 0, 1, 2)]:
                continue
            desc = 'ord_' + label + '_' + ''.join(map(str, perm))
            wanted = sum(target[:, k] * base ** j for j, k in enumerate(perm))
            matched = (observed == wanted) & allowed
            n = np.add.reduceat(matched.astype(int), starts)
            table[desc + '_count'] = n
            table[desc + '_share'] = n / np.maximum(1, counts - max(offsets) - 1)
            nearest = np.minimum.reduceat(np.where(matched, age, np.inf), starts)
            nearest[n == 0] = np.nan
            table[desc + '_due'] = period - nearest
            table[desc + '_elapsed'] = nearest - recency
        # Count non-contiguous occurrences of the last k states among strictly older
        # receipts by updating prefix counts once per observed event, from right to left.
        if offsets == tuple(range(length)):
            reversed_order = [code[np.maximum(starts, starts + counts - 1 - k)] for k in offsets[::-1]]
            for reverse in [False, True]:
                sought = reversed_order[::-1] if reverse else reversed_order
                acc = np.zeros((len(starts), length + 1))
                acc[:, 0] = 1
                for j in range(int(counts.max()) - 1):
                    active = j < counts - 1
                    symbol = code[np.minimum(starts + j, starts + counts - 1)]
                    for k in range(length, 0, -1):
                        acc[:, k] += acc[:, k - 1] * (active & (symbol == sought[k - 1]))
                opportunities = np.ones(len(counts))
                for j in range(length):
                    opportunities *= np.maximum(0, counts - 1 - j) / (j + 1)
                name = f'subseq_{length}_' + ('reverse' if reverse else 'forward')
                table[name + '_log_n'] = np.log1p(acc[:, -1])
                table[name + '_share'] = acc[:, -1] / np.maximum(1, opportunities)
    return pd.DataFrame(table, index=pd.Index(ids[starts], name='customer_id')).astype('float32')
