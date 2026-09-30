import json, csv, statistics
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
root=Path.cwd(); budgets=[1200,460,205,95,41]; sources=[]
def load(stem,seed):
 p=next((root/'outputs/diagnostics').glob(f'*_{stem}_s{seed}/onset_test.npz')); sources.append(str(p)); z=np.load(p,allow_pickle=False); return {k:z[k] for k in z.files if k!='meta'}
def measure(a,p):
 tracks=[np.flatnonzero(a['track_id']==i) for i in np.unique(a['track_id'])]
 pos=[i for i in tracks if np.any(a['onset_offset'][i]>=0)]; neg=[i for i in tracks if not np.any(a['onset_offset'][i]>=0)]
 nwin=sum(map(len,neg)); exposure=nwin/36000; out={'auc':float(roc_auc_score(a['crosses'],p)),'n_positive':len(pos),'n_negative':len(neg),'negative_windows':nwin,'exposure_hours':exposure}
 for account in ['per_window','per_track']:
  negatives=np.concatenate([p[i] for i in neg]) if account=='per_window' else np.array([max(p[i]) for i in neg]); values,counts=np.unique(negatives,return_counts=True); tails=np.cumsum(counts[::-1])[::-1]; rows=[]
  for b in budgets:
   allowed=int(b*exposure)
   if allowed>=len(negatives): th=-float('inf')
   else:
    eligible=values[tails<=allowed]; th=float(eligible[0]) if len(eligible) else float(np.nextafter(float(values[-1]),float('inf')))
   leads=[max(a['onset_offset'][i][(a['onset_offset'][i]>=0)&(p[i]>=th)])/30 for i in pos if np.any((a['onset_offset'][i]>=0)&(p[i]>=th))]
   rows.append({'budget':b,'detected':len(leads),'detection_rate':len(leads)/len(pos),'mean_lead_s':float(np.mean(leads)) if leads else 0.,'false_alarms':int(np.sum(negatives>=th))})
  out[account]=rows
 return out
res={x:[] for x in ['R2','R3','PRIMARY']}
for seed in [42,43,44]:
 r2=load('c4_r2s',seed); r3=load('c4_r3',seed); anc=load('c4_r2a',seed)
 for key in ['track_id','onset_offset','crosses','future_observed','track_crosses']:
  assert np.array_equal(r2[key],r3[key]) and np.array_equal(r2[key],anc[key]),key
 for name,p in [('R2',r2['p_frame']),('R3',r3['p_readout']),('PRIMARY',anc['p_frame']*r2['p_frame'])]: res[name].append({'seed':seed,**measure(r2,p.astype(float))})
reported=json.load(open('outputs/diagnostics/combo_test/results.json',encoding='utf-8')); names={'R2':'R2 alone (baseline)','R3':'R3 alone','PRIMARY':next(k for k in reported if k.startswith('PRIMARY:'))}; checks=0
for name,rows in res.items():
 for actual,old in zip(rows,reported[names[name]]):
  assert abs(actual['auc']-old['auc'])<1e-12
  for acc in ['per_window','per_track']:
   for p,q in zip(actual[acc],old[acc]):
    for k in ['detected','detection_rate','mean_lead_s','false_alarms']: assert abs(p[k]-q[k])<1e-10,(name,k,p,q)
    checks+=4
summary={}
for name,rows in res.items():
 summary[name]={'auc_mean':statistics.mean(x['auc'] for x in rows),'auc_sd':statistics.stdev(x['auc'] for x in rows)}
 for acc in ['per_window','per_track']:
  summary[name][acc]=[{'budget':b,'mean':statistics.mean(x[acc][j]['detection_rate'] for x in rows),'sd':statistics.stdev(x[acc][j]['detection_rate'] for x in rows)} for j,b in enumerate(budgets)]
verdicts={}
for name in ['R3','PRIMARY']:
 verdicts[name]={}
 for acc in ['per_window','per_track']:
  wins=sum(x['mean']-y['mean']>x['sd']+y['sd'] for x,y in zip(summary[name][acc],summary['R2'][acc])); reverse=sum(y['mean']-x['mean']>x['sd']+y['sd'] for x,y in zip(summary[name][acc],summary['R2'][acc])); verdicts[name][acc]={'wins':wins,'baseline_wins':reverse,'verdict':'INCONCLUSIVE' if max(wins,reverse)<3 else 'WIN'}
out={'method':'Independent numpy implementation from NPZ arrays, allow_pickle=False; no model/checkpoint load; paired row metadata verified; checked against combo JSON','checks':checks,'sources':sources,'summary':summary,'criterion':verdicts,'per_seed':res}
(root/'graphify-out/EXPERIMENT_VERIFICATION.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'checks':checks,'criterion':verdicts,'summary':summary},indent=2))
