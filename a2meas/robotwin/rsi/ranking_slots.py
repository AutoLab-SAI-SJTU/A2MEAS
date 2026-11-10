"""Budgeted ranking fallback using the same visible slots and original transfer primitive."""
import numpy as np

def retained_slot_plan(kin,state,points,rotations,remaining,transfer):
    points=np.asarray(points,dtype=float).copy();state=np.asarray(state).copy()
    if points.shape!=(3,3)or not np.isfinite(points).all():raise ValueError('Three finite visible rank points required')
    slots=np.sort(points[:,0])
    if np.any(np.diff(slots)<.06):raise ValueError('Visible slots overlap')
    goals=points.copy();goals[:,0]=slots;goals[:,1]=np.median(points[:,1])
    best=None;expanded=0
    def search(current,layout,segments,moves,cost):
        nonlocal best,expanded
        wrong=[i for i in range(3)if np.linalg.norm(layout[i,:2]-goals[i,:2])>.024]
        if not wrong:
            if not segments:return
            if best is None or cost<best[0]:best=(cost,segments,moves)
            return
        if len(moves)>=7 or expanded>=64:return
        free=[i for i in wrong if not any(j!=i and np.linalg.norm(layout[j,:2]-goals[i,:2])<.055 for j in range(3))]
        options=[]
        if free:options=[(i,goals[i],False)for i in free]
        else:
            seen=set()
            for i in wrong:
                for j in range(3):
                    if j==i or np.linalg.norm(layout[j,:2]-goals[i,:2])>=.055:continue
                    for x in [-.19,.19]:
                        dest=np.array([x,-.06,layout[j,2]])
                        if (j,x)not in seen and all(k==j or np.linalg.norm(dest[:2]-layout[k,:2])>.07 for k in range(3)):
                            seen.add((j,x));options.append((j,dest,True))
        for rank,dest,temporary in options:
            if expanded>=64:break
            expanded+=1
            budget=min(remaining-cost,(best[0]-cost)if best else remaining-cost)
            if budget<=10:continue
            try:actions,arm=transfer(kin,current,layout[rank],dest,rotations,budget)
            except ValueError:continue
            new_cost=cost+len(actions)
            if new_cost>=remaining-10 or(best and new_cost>=best[0]):continue
            new_layout=layout.copy();new_layout[rank]=dest
            move=dict(rank=rank,arm=arm,source=layout[rank].tolist(),destination=np.asarray(dest).tolist(),counted_steps=len(actions),temporary=temporary)
            search(actions[-1].copy(),new_layout,segments+[actions],moves+[move],new_cost)
    search(state,points,[],[],0)
    if best is None:raise ValueError('No complete visible-slot reorder within original action budget')
    return np.concatenate(best[1]),best[2],dict(goals=goals.tolist(),expanded_transfers=expanded,total_steps=best[0],remaining=int(remaining))
