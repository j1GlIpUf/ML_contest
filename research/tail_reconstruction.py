"""Reconstruct the final baskets from an older prefix with a frozen transition model."""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


def tail_reconstruction(receipts):
    b=receipts.sort_values(['customer_id','age','order_id'],ascending=[True,False,True]).copy()
    b['symbol']=pd.Categorical(b['mode']).codes
    rows=[]
    for uid,sub in b.groupby('customer_id',sort=False):
        states=sub.symbol.tolist()
        ages=sub.age.to_numpy(dtype=float)
        record={'customer_id':uid}
        lengths=[k for k in [1,2,3,5] if len(states)>=k+2]
        if not lengths:
            rows.append(record)
            continue
        transitions=defaultdict(Counter)
        waits=defaultdict(list)
        for i, actual in enumerate(states):
            if i>0:
                for depth in range(min(i,3)+1):
                    ctx=tuple(states[i-depth:i])
                    transitions[ctx][actual]+=1
                    waits[(ctx,actual)].append(ages[i-1]-ages[i])
            length=len(states)-i-1
            if length not in lengths:
                continue
            simulated=states[max(0,i-2):i+1]
            eta=ages[i]
            step_time_errors=[];steps_right=[];confidences=[];used=[];first_error=None
            for j in range(length):
                for depth in range(min(3,len(simulated)),-1,-1):
                    ctx=tuple(simulated[-depth:]) if depth else ()
                    counts=transitions.get(ctx)
                    if counts:
                        break
                # The largest count wins; Counter preserves the earliest observed
                # successor on ties. The prefix dictionary stays frozen throughout.
                predicted,votes=counts.most_common(1)[0]
                eta-=np.median(waits[(ctx,predicted)])
                truth=states[i+j+1]
                same=truth==predicted
                if not same and first_error is None:
                    first_error=j
                steps_right.append(same)
                step_time_errors.append(eta-ages[i+j+1])
                confidences.append(votes/sum(counts.values()))
                used.append(depth)
                simulated.append(predicted)
            tag=f'tail_{length}_'
            record.update({tag+'accuracy':np.mean(steps_right),tag+'first_error':length if first_error is None else first_error,
                tag+'final_right':steps_right[-1],tag+'final_bias':step_time_errors[-1],
                tag+'mean_abs_error':np.mean(np.abs(step_time_errors)),tag+'median_confidence':np.median(confidences),
                tag+'max_depth':max(used),tag+'final_depth':used[-1],tag+'final_age':eta})
        rows.append(record)
    return pd.DataFrame.from_records(rows).set_index('customer_id').astype('float32')
