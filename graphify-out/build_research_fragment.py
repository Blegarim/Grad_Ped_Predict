import json, re
from pathlib import Path

root = Path.cwd()
files = json.loads((root/'graphify-out/semantic_files_1.json').read_text(encoding='utf-8-sig'))
files += [str(root/p) for p in ['paper/main.tex','paper/refs.bib','paper/project_context.md'] if (root/p).exists()]
nodes, edges, hyperedges = {}, [], []
def norm(s): return re.sub('[^a-z0-9]+','_',s.lower()).strip('_')
def stem(p): return norm(str(Path(p).relative_to(root).with_suffix('')))
def node(p, entity, label=None, typ='concept', line=None, rationale=None, **attrs):
    nid=stem(p)+'_'+norm(entity)
    n=dict(id=nid,label=label or entity,file_type=typ,source_file=p,source_location=f'line:{line}' if line else None,source_url=None,captured_at=None,author=None,contributor=None)
    if rationale: n['rationale']=rationale
    n.update(attrs); nodes[nid]=n
    return nid
def edge(a,b,rel,p,line=None,confidence='EXTRACTED',score=1.0):
    edges.append(dict(source=a,target=b,relation=rel,confidence=confidence,confidence_score=score,source_file=p,source_location=f'line:{line}' if line else None,weight=1.0))
docids={}
for p in files:
    text=Path(p).read_text(encoding='utf-8-sig'); lines=text.splitlines()
    title=next((re.sub(r'^> ?','',l).lstrip('# ').strip() for l in lines if l.startswith('# ')),Path(p).name)
    status='historical' if '\\archive\\' in p else 'mixed_age_working_document'
    if p.endswith('main.tex'): status='current_working_tree_manuscript_v4'
    docids[p]=node(p,'document',title,'paper' if p.endswith(('.tex','.bib')) else 'document',1,status=status)
    for i,l in enumerate(lines,1):
        m=re.match(r'^#{2,3} (.+)',l)
        if m:
            label=m[1].strip(); sec=node(p,label,label,'concept',i,status=status)
            edge(docids[p],sec,'references',p,i)
    # Explicit internal document links; each file source remains verbatim absolute.
    for i,l in enumerate(lines,1):
        for target in re.findall(r'\]\(([^)]+)\)',l):
            target=target.split('#')[0]
            if not target or '://' in target: continue
            resolved=str((Path(p).parent/target).resolve())
            if resolved in files:
                edge(docids[p],stem(resolved)+'_document','references',p,i)

def concept(rel,entity,line,why,typ='rationale',status='current'):
    p=str(root/rel); n=node(p,entity,entity.replace('_',' '),typ,line,why,status=status)
    edge(docids[p],n,'references',p,line); return n

stream=concept('docs/project-context-streaming-crossing-onset.md','streaming_crossing_onset',49,'Continuous dense windows ask whether onset is imminent; anchored event-relative windows ask a different question.')
hard=concept('docs/project-context-streaming-crossing-onset.md','hard_temporal_negatives',67,'An eventual crosser before the forecast horizon is a negative in streaming, omitted by event-anchored sampling.')
gap=concept('docs/project-context-streaming-crossing-onset.md','gap_decomposition',95,'Separate the portion threshold recalibration recovers from the residual ranking gap; residual is not a causal identification without matched-size control.')
legacy=concept('docs/archive/REBUILD_SCHEMATIC.md','behavior_preserving_rebuild',10,'Port legacy behavior with characterization fixtures before intentional methodological changes.',status='historical')
golden=concept('docs/archive/MIGRATION.md','golden_fixture_parity',12,'Capture old behavior before porting, distinguish exact parity from documented intentional departures.',status='historical')
audit=concept('docs/archive/HOLE_AUDIT.md','v2_validity_repair',59,'Resolve threshold leakage, future-label misuse, censoring, correlation, seeding, model selection, missing ablations and normalization.',status='historical_completed')
pose=concept('docs/POSE_ENCODER.md','pose58_intrinsic_extrinsic_geometry',147,'49 pose features remove bbox position and scale already carried by 9 motion channels; raw keypoints stored once, features built at read time.')
flat=concept('docs/POSE_ENCODER.md','flattened_pose_temporal_encoder',165,'Reuse GRU and attention instead of adding ST-GCN; partial confidence-weighted skeleton and bounded supporting-study scope favor feature engineering.')
route=concept('docs/POSE_ENCODER.md','pose_rides_motions_tensor',211,'Preserve collate and forward contracts by widening motions; no separate PoseMotionEncoder class was implemented.')
backbone=concept('docs/BACKBONE_STUDY.md','pretrained_tinyvit_choice',22,'Avoid legacy dimension collapse, tiny receptive windows and data-hungry from-scratch ViT. TinyViT baseline choice is historical, not the current paper model.',status='historical_baseline')
bn=concept('docs/RECIPE_V2_PLAN.md','frozen_batchnorm_recipe_detour',47,'Image-mode probe exonerated the feature cache; true eval freezing and/or trainable projection explain worse performance, not cache corruption.',status='historical_negative')
onset=concept('docs/METHODOLOGY.md','onset_timing_under_censoring',351,'Represent event time and observed follow-up rather than fabricate negatives for missing futures; maintain probability of onset within 32 frames.')
metric=concept('docs/METHODOLOGY.md','detection_curve_primary_metric',66,'Count detection and earliest qualifying lead time under a hard false-alarm budget; account for per-window and per-track alarms separately.')
oldwin=concept('docs/METHODOLOGY.md','single_seed_two_to_six_fold_claim',29,'Historical single-seed claim was retracted by multi-seed campaigns; this document still presents it as current.',status='stale_retracted')
seeds=concept('docs/SEED_PLAN_2026-09-21.md','preregistered_multi_seed_criterion',47,'Hazard mean must exceed binary by more than sum of sample SDs at at least 3 of 5 budgets; otherwise inconclusive.')
pf=concept('docs/RERUN_PLAN_2026-09-26.md','pixel_free_v4_canonical_recipe',86,'Remove image decoding, standardize pose58, use batch32 and shuffled chunks; select best checkpoint on validation AUC.')
override=concept('docs/RERUN_PLAN_2026-09-26.md','v4_manual_no_go_override',8,'User launched v4 despite memorization NO-GO; no memorization fix was adopted, and overfitting must be stated.')
v4=concept('paper/main.tex','v4_twenty_run_campaign',71,'Replace older image-recipe paper results with 20 pixel-free runs; 3 seeds except the two Model A runs.')
null=concept('paper/main.tex','hazard_binary_inconclusive',928,'No budget clears the preregistered spread criterion; hazard is not demonstrated to detect better; binary warns earlier on average.')
whowhen=concept('paper/main.tex','who_versus_when_diagnosis',597,'Anchored model retains eventual-crosser discrimination (AUC 0.799) but imminent-vs-distant ordering is inverted (0.414).')
comb=concept('paper/main.tex','posthoc_model_combination_negative',620,'Pre-registered anchored-streaming score combination does not materially improve the streaming baseline.')
size=concept('paper/main.tex','unrun_matched_size_control',632,'Anchored 4.9k versus streaming 88k training windows confounds protocol attribution; manuscript pending marker remains.')
memor=concept('paper/main.tex','early_peak_memorization',647,'19 of 20 runs peak within eight epochs then validation declines while training loss falls; best-checkpoint reporting must be explicit.')
look=concept('paper/main.tex','lookahead96_bin4_horizon32',755,'96-frame future covers the 64-frame confusable band beyond H32, with 24 bins and 8 bins in the readout horizon.')
sd=concept('paper/main.tex','f1_spread_prose_inconsistency',232,'Introduction claims SD exceeds mean gap, but current macros give 0.008 and 0.025 versus 0.034 gap; the source comment says gap slightly exceeds their sum.',status='unresolved_document_inconsistency')
scope=concept('paper/main.tex','model_a_seed_scope_inconsistency',505,'Setup says every configuration uses 3 seeds; matrix caption says Model A seed42 only.',status='unresolved_document_inconsistency')
stale=concept('paper/project_context.md','partially_updated_binding_context',12,'Header retracts old headline and records v4, while body retains old claims, baseline identity, completed work as open, and outdated narrative status.',status='mixed_current_and_stale')
for a,b,rel in [(stream,hard,'conceptually_related_to'),(gap,hard,'conceptually_related_to'),(legacy,golden,'implements'),(audit,golden,'conceptually_related_to'),(pose,flat,'implements'),(pose,route,'implements'),(onset,look,'implements'),(v4,pf,'implements'),(v4,null,'references'),(v4,whowhen,'references'),(v4,memor,'references'),(v4,size,'references'),(null,seeds,'implements'),(null,metric,'implements'),(v4,override,'references')]:
    edge(a,b,rel,nodes[a]['source_file'],confidence='INFERRED',score=0.95)
for a,b,score in [(audit,onset,0.85),(flat,pf,0.95),(bn,pf,0.75),(golden,seeds,0.75),(whowhen,hard,0.85),(metric,stream,0.85),(oldwin,null,0.95),(stale,sd,0.85)]:
    edge(a,b,'semantically_similar_to',nodes[a]['source_file'],confidence='INFERRED',score=score)
# Extract bibliography entities and explicit manuscript citation edges.
bib=str(root/'paper/refs.bib'); bibtext=Path(bib).read_text(encoding='utf-8'); bibids={}
for m in re.finditer(r'@\w+\{([^,]+),([\s\S]*?)(?=\n@|\Z)',bibtext):
    key,body=m.groups(); title=re.search(r'title\s*=\s*\{([\s\S]*?)\},?\n\s*\w+\s*=',body)
    label=re.sub(r'[{}\s]+',' ',title.group(1)).strip() if title else key
    bibids[key]=node(bib,key,label,'paper',bibtext[:m.start()].count('\n')+1)
    edge(docids[bib],bibids[key],'references',bib)
main=str(root/'paper/main.tex')
for i,l in enumerate(Path(main).read_text(encoding='utf-8').splitlines(),1):
    for m in re.finditer(r'\\cite\{([^}]+)\}',l):
        for key in m.group(1).split(','):
            if key in bibids: edge(docids[main],bibids[key],'cites',main,i)
hyperedges.append(dict(id='paper_main_valid_comparison_contract',label='Comparable onset objective evaluation',nodes=[onset,look,metric,seeds,null],relation='form',confidence='INFERRED',confidence_score=0.95,source_file=main))
hyperedges.append(dict(id='docs_rerun_plan_2026_09_26_v4_recipe',label='v4 canonical recipe and limitations',nodes=[pf,override,v4,memor],relation='participate_in',confidence='INFERRED',confidence_score=0.95,source_file=str(root/'docs/RERUN_PLAN_2026-09-26.md')))
result=dict(nodes=list(nodes.values()),edges=edges,hyperedges=hyperedges,input_tokens=0,output_tokens=0)
assert all(e['source'] in nodes and e['target'] in nodes for e in edges)
(root/'graphify-out/.graphify_chunk_1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'nodes':len(nodes),'edges':len(edges),'hyperedges':len(hyperedges),'token_accounting':'unavailable; 0 sentinel'}))
