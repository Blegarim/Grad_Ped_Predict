# Local Git history evidence

Captured 2026-09-29. Local refs only; no fetch. Commit statements are historical evidence, not independent experiment replication.

## Snapshot

```text
 M outputs/runs/index.csv
 M paper/main.tex
?? graphify-out/
```

## Refs and branches

```text
  claude/multitask-loss-imbalance-yzFQi                        ae693e0 [origin/claude/multitask-loss-imbalance-yzFQi] Port unified multitask loss (Prompt 3.1, B3/B1/B4/B8)
  main                                                         de35f65 [origin/main] tbh i dont even know man its the paper and the v1 run and the reshuffling and 20 billion other things
  p9-cutover                                                   1482f72 P9 cutover: retire rebuild scaffolding, stand up v1.0 clean baseline
* pixel-free-rerun                                             fd507a2 Combination test (pre-registered): anchored who x streaming when is inconclusive
  plan/prompt-7.1-onnx                                         d5dc067 Add Prompt 7.1 sub-plan: ONNX export (export/onnx.py + scripts/export_onnx.py)
  pose-encoder-arm                                             3e08daa [origin/pose-encoder-arm: ahead 1] Untrack pose_cache/ npz binaries (accidentally committed by lab PC)
  remotes/origin/HEAD                                          -> origin/main
  remotes/origin/claude/ensemble-registry-rebuild-mWURl        fe674d4 Verify Prompt 2.4 parity in-sandbox; fix circular import + scope fixture
  remotes/origin/claude/eval-module-rebuild-plan-sqE1j         bdcf0ac Implement Prompt 5.2: efficiency benchmark (eval/benchmark.py)
  remotes/origin/claude/hole-audit-research-plan-f2zeka        16b2289 Add concise broad-audience research proposal (PROPOSAL.md)
  remotes/origin/claude/logging-csv-run-dir-Gxgv4              8cea434 Loosen trainer golden post-step weight parity to BLAS-portable atol
  remotes/origin/claude/logging-run-dir-setup-yqAR7            95cdfa8 Add run-dir layout + cross-run CSV index (Prompt 4.5)
  remotes/origin/claude/multitask-loss-imbalance-yzFQi         ae693e0 Port unified multitask loss (Prompt 3.1, B3/B1/B4/B8)
  remotes/origin/claude/onnx-export-rebuild-wyt58b             d351c16 Implement ONNX export + parity checks (Prompt 7.1)
  remotes/origin/claude/pedpredict-ablations-port-NK7Ys        5bd6ecf Port ablation models (Prompt 2.5, B4/B11)
  remotes/origin/claude/pedpredict-callbacks-rebuild-rOfKe     8cea434 Loosen trainer golden post-step weight parity to BLAS-portable atol
  remotes/origin/claude/pedpredict-cross-attention-heads-MqVLa 7bcc799 Port CrossAttentionModule + heads (Prompt 2.3, B4)
  remotes/origin/claude/pedpredict-sampler-weights-diIw1       e80a96a 1.6
  remotes/origin/claude/pedpredict-viz-module-h9kkpo           790fd7b Implement viz/qualitative.py (Prompt 6.2): GT/comparison/attention overlays
  remotes/origin/claude/pedpredict-viz-rebuild-pWPaN           3c1f555 Fix pre-existing ruff nits in test_callbacks.py
  remotes/origin/claude/waiting-for-training-56s2zq            61df1b9 Add backbone candidate study (A1/A2)
  remotes/origin/main                                          de35f65 tbh i dont even know man its the paper and the v1 run and the reshuffling and 20 billion other things
  remotes/origin/pose-encoder-arm                              d51e37e m
```

## Tags

```text
legacy-archive  Pre-cutover snapshot: OLD repo + rebuild scaffolding (SCHEMATIC, MIGRATION, golden capture sources)
rerun-v4-code   Memorization ladder stage: metadata-only reshuffle, gate AUC summary
v1.0.0          Clean Phase-A baseline: standalone pedestrian behavior prediction (PIE)
```

## Full reachable log

```text
fd507a2777c175450d574c45bcd5bc2ee9969e1f 2026-09-29T10:29:48+09:00  (HEAD -> pixel-free-rerun) Combination test (pre-registered): anchored who x streaming when is inconclusive
a9f67ff7a70749f33fafc8ba4465befb4c913222 2026-09-29T10:23:36+09:00  Pre-register the anchored-who x streaming-when combination test (before any number)
78c5d1bbee187bd924e53eeb02c1ef89d44f93a5 2026-09-29T10:21:25+09:00  Log the v4 pixel-free campaign: artifacts, generated tables, results ledger
8c8be99e29ebbe1052c4e15b661202c1e587eb99 2026-09-27T11:06:01+09:00  Plan: campaign launched by manual override (gate NO-GO), memorization ladder held
d4fac3c9fae209a0eb3e298c006865a0a5f3de0b 2026-09-27T07:12:29+09:00  (tag: rerun-v4-code) Memorization ladder stage: metadata-only reshuffle, gate AUC summary
8513b288de77f55dac33770170172f76c7d25ef0 2026-09-27T06:51:00+09:00  Select best.pth on val crosses_auc; campaign stops reusing F1-selected runs
5aca9d760ba6e5613a96f1a1e6f2dc8b466b50df 2026-09-26T15:11:04+09:00  Pixel-free re-run campaign: warm-up fix, resume fix, gate, config-diff check
de35f65c94a8e367b0a9ef912938855846151825 2026-09-25T10:48:51+09:00  (origin/main, origin/HEAD, main) tbh i dont even know man its the paper and the v1 run and the reshuffling and 20 billion other things
691e06ae35317ac4b8484e9edf6a8fb44704afdd 2026-09-13T20:19:38+09:00  cleanup config, default config locked, clean up setup.md
bfc541478c55c8446f18a021ffe6264fba65cb94 2026-09-13T19:32:29+09:00  interrupted training sess
5f8eecd0b53102761820b30ee47639a89a84695a 2026-09-11T13:07:26+09:00  setup markdown
cdc56c878138055430bbe642c701a5ac21a7de84 2026-09-11T12:42:24+09:00  bug fix
c75fa6edc3313d488d4deda00485bb46601eef20 2026-09-11T11:55:15+09:00  paper, resume mechanism, etc.
23986a918dfb959182780fd398828182600adb21 2026-09-04T10:10:18+09:00  bug fix
6f409afe17230e58cf79cf417da68c97b3da9c8e 2026-08-27T16:30:38+09:00  result
b9a381fd2eed77ff61a6f6f98371da99e265a559 2026-08-27T16:25:04+09:00  new head. new direction. new day. new me.
fa21c61d75e200cb6f50009170b55e42e5ae6c8c 2026-08-19T13:11:07+09:00  docs: consolidate to 6 maintained + 3 frozen + archive; re-pin stats gate
dc2c1bba12f21c8ebabb097ae975b2fc09206a40 2026-08-19T12:57:44+09:00  docs: track METHODOLOGY.md + RESULTS_MATRIX.md
59ee062a187d640c43ae5744585072116a402b84 2026-07-22T14:13:59+09:00  m
1a5ee038f740e2724a34f4b4b20248e3dac53e2b 2026-07-22T12:41:40+09:00  roadmap. run_arm.py
9ace948cf01d4f6b0cc7a4c2797e180b80f96f5d 2026-07-21T15:23:32+09:00  Merge branch 'main' of https://github.com/Blegarim/Grad_Ped_Predict
8a258db64dbd9a355ae3ed0cc7fe4dd744a69d93 2026-07-21T15:19:30+09:00  cross only finish matrix
90425323e31fb5bde6b422d9fbf7a7421fc00e78 2026-07-15T14:10:57+09:00  deleted obsolete run
d4bffa9fd2e85f2740fd7513e3526c9156747cec 2026-07-15T13:32:30+09:00  m
b6486946f7ec591c98bf0fd76df7e10cd3098cff 2026-07-15T11:50:29+09:00  cross only frozen vit pose full retrain
8d2c559b210de057a4667ba8674eaec94d5b802c 2026-07-14T13:32:05+09:00  cross-only promoted to first class
e35296058499b3db1c82af150322edb1d5de6460 2026-07-14T11:51:50+09:00  training sess frozen vit pose full crosses only
a4c38807cde41dad7a70982efe46f113ff480142 2026-07-12T15:56:08+09:00  threshold overwrite fix
b7f6b14cbec96b6f1f18624798a0960adc24bfc1 2026-07-12T15:19:29+09:00  eval for the last session
80b0d80b6db18825701896c68fc79f944d805536 2026-07-12T14:27:13+09:00  training sesison pose full freeze vit streaming data. error on epoch 24
da5f8faea0fa242b37d4927d10a0350a41e20f79 2026-07-10T15:10:21+09:00  eval: add protocol column to eval_log.csv (streaming | anchored)
3e08daa1d3c6bccb938688f3706794209d13cf92 2026-07-10T15:00:15+09:00  (pose-encoder-arm) Untrack pose_cache/ npz binaries (accidentally committed by lab PC)
d51e37e8413cf049663e349b8e0ad2f9a076f2ee 2026-07-10T14:57:47+09:00  (origin/pose-encoder-arm) m
20342d88b438d7aa4a615507ca79d340d4464d83 2026-07-10T14:56:28+09:00  trainin sess with pose full frozen vit. the result is GLARING in what were tryna do lol
ecf2f7cbd60fc9ebf2066303d8719cd0ffee0a9b 2026-07-10T14:19:27+09:00  eval changes: read config yaml every run
0ebb7e2082c64420c75e18fcd7a180d5b72dcaa4 2026-07-09T15:49:06+09:00  build incremental add all, all_benchmark, exhaustive
9b8fca897e7606dc0859f7e8d4b30cad45b52a4d 2026-07-08T12:19:23+09:00  extract_pose: stream frames in-memory from PIE_clips (no staged images)
9cd1eb1aa70ef0993a6295c8efaecf658b44728a 2026-07-08T11:53:20+09:00  pose arm doc-sync: CLAUDE/README/setup + POSE_ENCODER status + implementation notes
b2ffa6e53d3171b4d98717a57fec9a91242e60ec 2026-07-08T11:49:47+09:00  pose arm step 4: extract_pose.py (bbox-conditioned dwpose -> npz cache, --dry-run)
2ba235ed5cf62179e84a433aad4a8d099be98994 2026-07-08T11:46:41+09:00  pose arm step 3: pose through ProcessedSample/meta/read path, flip coupling, loader wiring
4f3bbf333cca74e1ea84454df23390445b287330 2026-07-07T18:58:31+09:00  pose arm step 2: motion_norm=none, PoseFullModel, pose_kinematics/pose_full registry + export wiring
545d1d7fe96a25f14f584f4d16d13c9d87eb60de 2026-07-07T18:55:11+09:00  pose arm step 1: data/pose.py feature math + synthetic tests
bcdca590c9db4863d0a1dfd367e5c753c4ab6b4c 2026-07-07T18:52:31+09:00  pose arm step 0: PoseCfg scaffold + design doc
0cb737390b1885dad93085e33efcedd1f9771fe7 2026-07-07T18:38:46+09:00  training sess freeze backbone #1
d9eea4e42fd20d58a62da6fdbb1fbef0ff1d8bf4 2026-07-07T13:06:28+09:00  freeze backbone
c68fb26c85f359f948fde76d7669d56e2bcec85c 2026-07-07T12:17:24+09:00  training session, pretrained vit #1
d88edbefdcbbe237a2610f3bac90e6c7d7d11f09 2026-07-06T16:42:06+09:00  pretrained vit, runtime augmentation, doc changes.
72b12a2efddc042ed61e48b311361ac8b8dc139e 2026-07-06T15:09:22+09:00  benchmark training sess #1
d126c5a0bd2988441620a1aeb197757c51773fe4 2026-07-02T16:28:24+09:00  pivot.
1f6af021010a5fd2dc9dd0097e4f540770ca8a18 2026-07-01T12:50:34+09:00  unfinished training session #4
9b864d831a1e92aba2a39e8795a6a1cd1c041028 2026-06-30T11:23:57+09:00  vit architecture change
bf69c6a1ee9a10a2b02563f17cb70688195dbb34 2026-06-30T10:55:06+09:00  gradient accumulation, fusion residual, calibration script
5f4410e24f0d347276cb8a7e1af9d60e407b79bb 2026-06-25T10:00:27+09:00  cross attention residual connection
3eddd961313f5a9c63271b5869227fac78b459bc 2026-06-25T08:36:06+09:00  registry + rename wave
d3af4827b912b614a20eaa6b11de87a1d17cd832 2026-06-25T08:32:58+09:00  m
e73ea24d35dc1295062eb78dad91a04f450889f1 2026-06-24T10:54:17+09:00  dataset statistics, unfinished training sess
bb234309b85f078b0884b18529b8800eec80c15d 2026-06-22T14:12:17+09:00  little train default config toggle
4d960943387c0313ee6e647ecd6895a2796b1592 2026-06-22T14:04:32+09:00  warmup -> cosine lr strategy
edbb78ef31839045cec550896e7c188bb51d3892 2026-06-22T13:02:04+09:00  training session #2 - sampler tuned down
fb40def989944796a9eba05cd16092ef2a2891ca 2026-06-19T13:56:12+09:00  sweep distribution
b7f318149e2295e373a1918250c1d44784798e5c 2026-06-19T13:19:52+09:00  initial training session
903140e5f7e2f7deb4671dc360b6499577c449f4 2026-06-19T13:18:20+09:00  skills usage threshold, gitignore
52cf083ca5732f9ff2ac25e94d3fb8b2f93eea1d 2026-06-19T12:02:03+09:00  setup
61df1b91f6ed1e5d10b99f5de0da27fa95c40cf1 2026-06-18T01:58:53Z  (origin/claude/waiting-for-training-56s2zq) Add backbone candidate study (A1/A2)
e95a2a12c713811116cfbc45d1421d1568eacc07 2026-06-16T14:29:52+09:00  augment from base lmdb
f181079fc84d868dbe279c67ec35bd1ec3b2705b 2026-06-15T21:22:14+09:00  temp set4 fix
33749c081ddbe673bd14e9efeca0fce36863bb0b 2026-06-15T20:12:34+09:00  jpeg
2e5106107c6f63db80b97ddb81b9cc88f5dbb3fb 2026-06-15T20:04:58+09:00  bug fixes
00ed66eef9b45a5b7e379c60332c10bd25eb877a 2026-06-15T19:41:14+09:00  Consolidate setup.md as the canonical end-to-end runbook; drop V2 runbook
3811f2ae156414f46f6300f9f6b51b652e89f949 2026-06-15T19:10:08+09:00  Update setup.md + runbook for v2 low-storage build
be295585c751722828430ccf8141b1204e461c5c 2026-06-15T19:03:49+09:00  Adapt research plan + audit progress to merged step 2
6ba317bc00289edd7d2ee379f3eb9131e49dc5eb 2026-06-15T19:01:45+09:00  Add broad-audience research proposal (PROPOSAL.md)
66c6000d13d6c017acfb0be681df514331fa8324 2026-06-15T18:57:05+09:00  hole audit #2
16b228944c1cec84c848328a85b1946acc22f6c0 2026-06-14T07:52:02Z  (origin/claude/hole-audit-research-plan-f2zeka) Add concise broad-audience research proposal (PROPOSAL.md)
fe4da59626cc0938bfbfc9be305d4ee4e08321b7 2026-06-14T07:47:12Z  Reconcile RESEARCH_PLAN with resolved hole audit
4c2120bc13095c300c3e6c222b5d12d854f7bebd 2026-06-11T17:51:49+09:00  hole_audit #1, data-independent code fix
5524350a0fd0267c0c6b61fb7ed3cfbfce0e79c3 2026-06-11T13:23:17+09:00  Merge branch 'main' of https://github.com/Blegarim/Grad_Ped_Predict
cc8ec6fcf2ff227995f6585652da98bf23529687 2026-06-11T13:22:22+09:00  HOLE_AUDIT.md
a789f7b6bf3f8064591351ccc75478bde97dada3 2026-06-10T21:00:08+09:00  session 1 set up
49fa001bd3e045d89cedce0eab849d657e8ed9bb 2026-06-10T16:16:16+09:00  tqdm
7603ff35ff88d118d0d5f1e32734b623b6c0611c 2026-06-10T14:34:25+09:00  split clips to frame shell
a9735c9ac2b91ec0bf2a3ebb0971fce2460f5e6b 2026-06-10T13:34:39+09:00  test and setup.md
0b1fc3a2b06965a845f4a3b6e4476ef7687a5fff 2026-06-09T21:29:34+09:00  setup.md
1482f72b265b2b62c0c58589584d42c290f6d0f3 2026-06-09T21:12:40+09:00  (tag: v1.0.0, p9-cutover) P9 cutover: retire rebuild scaffolding, stand up v1.0 clean baseline
aa0a0e6a9e8fdc57afe28185f07de6e72ba51856 2026-06-09T20:48:52+09:00  (tag: legacy-archive) training bug fixed, viz/export lint
4f2206a44fe0adf0d78c1e4c3292de2d08d4d714 2026-06-09T12:00:38+09:00  8.2
1232350cf3eb70a1c7339a61a159016cd189c4bf 2026-06-09T11:42:00+09:00  8.1
bb5a7517dcb9971b50700e386fdc516f2c481a0c 2026-06-09T11:28:06+09:00  Merge claude/onnx-export-rebuild-wyt58b: Implement ONNX export + parity checks (Prompt 7.1)
d351c16ceedc614d50217c945538f332eb1ea383 2026-06-09T02:03:49Z  (origin/claude/onnx-export-rebuild-wyt58b) Implement ONNX export + parity checks (Prompt 7.1)
0d55702287c8329893ff3754ccd80f61b8ee95bf 2026-06-09T00:08:10Z  Merge claude/pedpredict-viz-module-h9kkpo: Prompts 5.1–6.2 + viz/qualitative.py
790fd7b780ad160472c6668814222ff23e30c69c 2026-06-08T21:52:29Z  (origin/claude/pedpredict-viz-module-h9kkpo) Implement viz/qualitative.py (Prompt 6.2): GT/comparison/attention overlays
0c3d3199089067d38deea2a21ade983c014efbc0 2026-06-08T12:24:45+09:00  5.3
d5dc067325a95a999b090abac12b87da1a49aa8b 2026-06-08T11:52:28+09:00  (plan/prompt-7.1-onnx) Add Prompt 7.1 sub-plan: ONNX export (export/onnx.py + scripts/export_onnx.py)
3c1f555a5d81d5c720942f6a841f1097b24937f9 2026-06-08T01:04:35Z  (origin/claude/pedpredict-viz-rebuild-pWPaN) Fix pre-existing ruff nits in test_callbacks.py
9fba24e3491ee3d119a362fbedeed1f6cef62caa 2026-06-07T07:16:19Z  Port quantitative plots to new run-dir artifacts (Prompt 6.1)
bdcf0aca5d0b833d9911fdea09ec99c83cd8ef40 2026-06-06T13:15:24Z  (origin/claude/eval-module-rebuild-plan-sqE1j) Implement Prompt 5.2: efficiency benchmark (eval/benchmark.py)
891d2fe7ecbe657904abcff2f894cc7612855c73 2026-06-06T12:00:01Z  Implement Prompt 5.1: evaluation pipeline (eval/evaluate.py + scripts/evaluate.py)
3072f295198c906c0d6800f386f421d5c5f1ee29 2026-06-06T11:39:04Z  Add Prompt 5.1 sub-plan: evaluation pipeline (eval/evaluate.py + scripts/evaluate.py)
c2cecf8cb430d77b72364afa2105ca2de93f51d8 2026-06-06T08:02:38Z  Merge: run-dir layout + cross-run CSV index (Prompt 4.5)
95cdfa81dd1e80e095c48ca7326a63241c39fd80 2026-06-06T08:00:53Z  (origin/claude/logging-run-dir-setup-yqAR7) Add run-dir layout + cross-run CSV index (Prompt 4.5)
8cea434fd3c943599631f5d7ba1bfda50e95f99d 2026-06-06T03:49:30Z  (origin/claude/pedpredict-callbacks-rebuild-rOfKe, origin/claude/logging-csv-run-dir-Gxgv4) Loosen trainer golden post-step weight parity to BLAS-portable atol
2351f95008146d1effc8daa51dd728594b8d219d 2026-06-05T07:38:41Z  Port Prompts 4.3 + 4.4: CheckpointManager + two-phase schedule (B1, B2-load, B11)
2cacf380bc5abca4d8b46af68e56c7ad1ec148e1 2026-06-05T06:47:37Z  Prompt 4.3: CheckpointManager, full-state resume, B2/B11 closure
faabe482064aa82d94916dac4c404313ed294a43 2026-06-05T13:58:56+09:00  4.2
181935d8d9ce86dd702385f25c04d3d2838c9729 2026-06-05T13:00:49+09:00  4.1
66e717ba0024519aa04d172f3c8b625f49866f05 2026-06-05T12:28:53+09:00  3.2
ae693e096feb026a7855b5be699b027b1add7d26 2026-06-04T14:24:15Z  (origin/claude/multitask-loss-imbalance-yzFQi, claude/multitask-loss-imbalance-yzFQi) Port unified multitask loss (Prompt 3.1, B3/B1/B4/B8)
5bd6ecf44889684f0b9a076aede7fef2069da560 2026-06-04T11:30:46Z  (origin/claude/pedpredict-ablations-port-NK7Ys) Port ablation models (Prompt 2.5, B4/B11)
fe674d41784862e110c51a78a17a4fbda409c8ce 2026-06-04T10:58:34Z  (origin/claude/ensemble-registry-rebuild-mWURl) Verify Prompt 2.4 parity in-sandbox; fix circular import + scope fixture
44c814f0b1fdbeabf4a0f25dc4519dbb68603b8b 2026-06-04T10:47:59Z  Port EnsembleModel + typed model registry (Prompt 2.4, B10)
7bcc799ff475ab81603c957dcf0baf55838b0bff 2026-06-04T05:00:07Z  (origin/claude/pedpredict-cross-attention-heads-MqVLa) Port CrossAttentionModule + heads (Prompt 2.3, B4)
9170878ca72d9fdf1a8e860ee94b0e58717b646b 2026-06-04T11:38:22+09:00  2.2. motion enc built
18e865051f9f59af4fcf1b50c669d61b9941e386 2026-06-04T11:28:53+09:00  1.7 + 2.1. vit built
e80a96a01ff0129b24960362b6a0380a552315a9 2026-06-04T00:49:58Z  (origin/claude/pedpredict-sampler-weights-diIw1) 1.6
4588acd30f3b858bb4b1686107301c550f7bef6b 2026-06-04T09:18:07+09:00  changed references from hardcoded path
da9157a26455317cc3c3e7a4da39446e748d34c1 2026-06-04T09:16:57+09:00  golden artifact cloned for direct references
468f53e9c90c3347346a3dd519b73c8d03d17e5b 2026-06-03T15:03:09+09:00  1.4, 1.5
61337602971787f8f3c357060f8875697491cb7a 2026-06-03T14:02:41+09:00  1.3
5169d9204db0fc81ceff11793a15a5ddc5e4bf1f 2026-06-03T09:02:31+09:00  1.2
c6ef28fdfd782adfad65f310589615701bc4961b 2026-06-03T07:43:45+09:00  1.1
e2c4e08f554298f392a4a25d8662e2c02be5bad2 2026-06-03T07:22:54+09:00  0.3
0a235f3e484da0067e08da1b6fa839e411421040 2026-06-03T07:09:14+09:00  0.2 — typed config system (yaml -> dataclass -> argparse)
bbb2dc0beb1f8d2a31a53ac3dd6ffeaa2097b287 2026-06-02T15:39:26+09:00  0.1
3e1e972bdd6ff4edce6f68951f73ba5555ca3f86 2026-06-02T14:48:48+09:00  Initial commit
```

## Reachable commit count

```text
124
```

## Milestone 1482f72

```text
commit 1482f72b265b2b62c0c58589584d42c290f6d0f3
Author:     Blegarim <nguyenbaoviet25072003@gmail.com>
AuthorDate: Tue Jun 9 21:12:40 2026 +0900
Commit:     Blegarim <nguyenbaoviet25072003@gmail.com>
CommitDate: Tue Jun 9 21:12:40 2026 +0900

    P9 cutover: retire rebuild scaffolding, stand up v1.0 clean baseline
    
    Complete the Phase-A migration cutover and retire the legacy/rebuild scaffolding.
    
    - Retire OLD/ (vendored legacy repo) from the working tree; preserved in the
      `legacy-archive` git tag.
    - Archive MIGRATION.md, REBUILD_SCHEMATIC.md, and the per-prompt sub-plans under
      docs/archive/; add docs/archive/legacy_baselines.md.
    - Flip CLAUDE.md / README.md to standalone docs (drop the Rebuild Context section and
      the B1-B13 band-aid inventory).
    - Strip prompt-number provenance asides from module/script/config docstrings; repoint
      surviving ledger references to docs/archive/.
    - Reframe golden fixtures as characterization tests; document that the OLD-importing
      regenerators require the legacy-archive tag (tests/_capture/README.md).
    - Add docs/PHASE_B_BACKLOG.md; bump version 0.0.0 -> 1.0.0.
    
    Gate green: ruff clean, pytest -m "not slow" = 393 passed, 3 skipped.
    
    Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

 .gitignore                                         |  12 +-
 CHANGELOG.md                                       |  34 ++
 CLAUDE.md                                          |  60 +-
 .../.claude/skills/code-review-protocol/SKILL.md   | 139 -----
 .../.claude/skills/debugging-playbook/SKILL.md     | 140 -----
 .../references/debug-checklist.md                  |  34 --
 .../.claude/skills/experiment-tracking/SKILL.md    | 122 ----
 .../references/baseline-results.md                 |  39 --
 OLD/Undergrad_thesis_project/.gitattributes        |   2 -
 OLD/Undergrad_thesis_project/.gitignore            |  25 -
 OLD/Undergrad_thesis_project/.vscode/settings.json |   3 -
 OLD/Undergrad_thesis_project/CLAUDE.md             | 152 -----
 OLD/Undergrad_thesis_project/GUIDELINE.md          |  63 --
 OLD/Undergrad_thesis_project/README.md             | 118 ----
 .../ablation_usage_example.py                      | 160 -----
 .../class_imbalance_strategies.py                  | 469 --------------
 OLD/Undergrad_thesis_project/config.py             |  43 --
 OLD/Undergrad_thesis_project/extract_frames.py     |  45 --
 .../final_ablation_verification.py                 | 132 ----
 OLD/Undergrad_thesis_project/imbalance_config.py   | 282 ---------
 OLD/Undergrad_thesis_project/label_count.py        |  64 --
 OLD/Undergrad_thesis_project/main.py               | 325 ----------
 .../models/AblationModels.py                       | 214 -------
 .../models/Cross_Attention_Module.py               |  83 ---
 .../models/Motion_Encoder.py                       | 118 ----
 .../models/Unified_Module.py                       |  31 -
 .../models/Vision_Transformer.py                   | 408 -------------
 OLD/Undergrad_thesis_project/models/__init__.py    |   0
 OLD/Undergrad_thesis_project/onnx/onnx_export.py   |  70 ---
 OLD/Undergrad_thesis_project/plots/loss_curves.png | Bin 48989 -> 0 bytes
 .../plots/per_head_f1_curves.png                   | Bin 62910 -> 0 bytes
 OLD/Undergrad_thesis_project/requirements.txt      |  50 --
 OLD/Undergrad_thesis_project/run_env.bat           |   3 -
 .../scripts/PIE_sequence_Dataset_1.py              | 201 ------
 OLD/Undergrad_thesis_project/scripts/__init__.py   |   0
 .../scripts/augment_sequences.py                   | 146 -----
 .../scripts/balance_sequences.py                   | 146 -----
 .../scripts/generate_sequences.py                  | 100 ---
 .../scripts/lmdb_dataset.py                        | 104 ----
 .../scripts/model_utils.py                         |  90 ---
 .../scripts/pedestrian_detection.py                | 130 ----
 .../scripts/plot_results.py                        | 504 ---------------
 .../scripts/preprocess_data.py                     | 111 ----
 .../scripts/preprocess_data_lmdb.py                | 168 -----
 .../scripts/split_balance_sequences_all.py         | 228 -------
 .../scripts/train_utils.py                         |  97 ---
 OLD/Undergrad_thesis_project/test.py               | 619 -------------------
 .../test_ablation_models.py                        | 181 ------
 .../test_ablation_structure_clean.py               |  61 --
 .../test_imbalance_setup.py                        | 264 --------
 OLD/Undergrad_thesis_project/train.py              | 631 -------------------
 OLD/Undergrad_thesis_project/train_two_phase.py    | 304 ---------
 .../visualize_comparison.py                        | 680 ---------------------
 OLD/Undergrad_thesis_project/visualize_gt.py       | 437 -------------
 OLD/golden/README.md                               |  46 --
 OLD/golden/sequences_test_sample.pkl               | Bin 1202103 -> 0 bytes
 OLD/golden/sequences_train_sample.pkl              | Bin 1201508 -> 0 bytes
 OLD/golden/sequences_val_sample.pkl                | Bin 1202103 -> 0 bytes
 README.md                                          |  22 +-
 configs/augment.yaml                               |   2 +-
 configs/balance.yaml                               |   4 +-
 configs/export.yaml                                |   2 +-
 configs/infer.yaml                                 |   2 +-
 configs/paths.yaml                                 |   2 +-
 configs/schedule.yaml                              |   4 +-
 configs/train.yaml                                 |   2 +-
 docs/PHASE_B_BACKLOG.md                            |  38 ++
 MIGRATION.md => docs/archive/MIGRATION.md          |   0
 .../archive/REBUILD_SCHEMATIC.md                   |   0
 docs/archive/legacy_baselines.md                   |  34 ++
 docs/{ => archive}/plans/PROMPT_5.1_evaluation.md  |   0
 docs/{ => archive}/plans/PROMPT_5.3_inference.md   |   0
 pyproject.toml                                     |   4 +-
 scripts/augment_dataset.py                         |   3 +-
 scripts/balance_dataset.py                         |   3 +-
 scripts/build_lmdb.py                              |   2 +-
 scripts/count_labels.py                            |   2 +-
 scripts/evaluate.py                                |   2 +-
 scripts/export_onnx.py                             |   2 +-
 scripts/infer_video.py                             |   2 +-
 scripts/make_sequences.py                          |   2 +-
 scripts/train.py                                   |   2 +-
 src/pedpredict/config/__init__.py                  |   2 +-
 src/pedpredict/config/loader.py                    |  24 +-
 src/pedpredict/config/schema.py                    |  26 +-
 src/pedpredict/data/augment.py                     |   4 +-
 src/pedpredict/data/balance.py                     |   6 +-
 src/pedpredict/data/collate.py                     |   2 +-
 src/pedpredict/data/lmdb_dataset.py                |   2 +-
 src/pedpredict/data/lmdb_warm.py                   |   2 +-
 src/pedpredict/data/lmdb_writer.py                 |   4 +-
 src/pedpredict/data/pie_sequences.py               |   2 +-
 src/pedpredict/data/sampler.py                     |   6 +-
 src/pedpredict/data/stats.py                       |   4 +-
 src/pedpredict/data/transforms.py                  |   6 +-
 src/pedpredict/eval/benchmark.py                   |   2 +-
 src/pedpredict/eval/evaluate.py                    |   2 +-
 src/pedpredict/eval/inference.py                   |   2 +-
 src/pedpredict/export/onnx.py                      |   4 +-
 src/pedpredict/losses/__init__.py                  |   2 +-
 src/pedpredict/losses/multitask.py                 |   6 +-
 src/pedpredict/models/ablations.py                 |   4 +-
 src/pedpredict/models/cross_attention.py           |   4 +-
 src/pedpredict/models/ensemble.py                  |   6 +-
 src/pedpredict/models/heads.py                     |   8 +-
 src/pedpredict/models/motion_encoder.py            |   4 +-
 src/pedpredict/models/registry.py                  |   2 +-
 src/pedpredict/models/vit.py                       |   2 +-
 src/pedpredict/paths.py                            |   2 +-
 src/pedpredict/training/callbacks.py               |   6 +-
 src/pedpredict/training/chunk_loader.py            |   2 +-
 src/pedpredict/training/metrics.py                 |   4 +-
 src/pedpredict/training/schedule.py                |   4 +-
 src/pedpredict/training/trainer.py                 |   4 +-
 src/pedpredict/utils/__init__.py                   |   2 +-
 src/pedpredict/utils/amp.py                        |   2 +-
 src/pedpredict/utils/device.py                     |   2 +-
 src/pedpredict/utils/memory.py                     |   2 +-
 src/pedpredict/utils/seed.py                       |   2 +-
 src/pedpredict/viz/plots.py                        |   6 +-
 src/pedpredict/viz/qualitative.py                  |   2 +-
 tests/_capture/README.md                           |  17 +
 tests/_golden.py                                   |  22 +-
 123 files changed, 264 insertions(+), 8777 deletions(-)
```

## Milestone d126c5a

```text
commit d126c5a0bd2988441620a1aeb197757c51773fe4
Author:     Blegarim <nguyenbaoviet25072003@gmail.com>
AuthorDate: Thu Jul 2 16:28:24 2026 +0900
Commit:     Blegarim <nguyenbaoviet25072003@gmail.com>
CommitDate: Thu Jul 2 16:28:24 2026 +0900

    pivot.

 CLAUDE.md                                        |  29 +++-
 configs/data.yaml                                |   3 +
 configs/model.yaml                               |  10 +-
 configs/paths.yaml                               |   2 +
 docs/HOLE_AUDIT.md                               |   8 +
 docs/RESEARCH_PLAN.md                            |  13 ++
 docs/project-context-streaming-crossing-onset.md | 188 +++++++++++++++++++++++
 docs/streaming-onset-plan.md                     | 173 +++++++++++++++++++++
 outputs/runs/index.csv                           |   3 +
 scripts/build_lmdb.py                            |  12 +-
 scripts/build_lmdb_incremental.py                |  15 +-
 scripts/train.py                                 |   8 +-
 setup.md                                         |  61 +++++++-
 src/pedpredict/config/loader.py                  |   3 +
 src/pedpredict/config/schema.py                  |  84 ++++------
 src/pedpredict/data/pie_sequences.py             |  45 ++++++
 src/pedpredict/eval/evaluate.py                  |   8 +-
 src/pedpredict/paths.py                          |  21 ++-
 src/pedpredict/training/chunk_loader.py          |  11 +-
 tests/test_config.py                             |  14 +-
 tests/test_data_shapes.py                        |  38 +++++
 tests/test_eval.py                               |   5 +-
 tests/test_model_shapes.py                       |   5 +-
 tests/test_schedule.py                           |   5 +-
 tests/test_trainer.py                            |   4 +-
 tests/test_utils.py                              |  21 ++-
 26 files changed, 694 insertions(+), 95 deletions(-)
```

## Milestone b9a381f

```text
commit b9a381fd2eed77ff61a6f6f98371da99e265a559
Author:     Blegarim <nguyenbaoviet25072003@gmail.com>
AuthorDate: Thu Aug 27 16:25:04 2026 +0900
Commit:     Blegarim <nguyenbaoviet25072003@gmail.com>
CommitDate: Thu Aug 27 16:25:04 2026 +0900

    new head. new direction. new day. new me.

 .claude/skills/comprehensive-speech/SKILL.md |  33 +++
 CLAUDE.md                                    | 116 +++++++--
 README.md                                    |  14 +-
 configs/data.yaml                            |  25 +-
 configs/model.yaml                           |  20 ++
 configs/train.yaml                           |  12 +
 docs/METHODOLOGY.md                          | 309 +++++++++++++++++++-----
 docs/THESIS_ROADMAP.md                       | 107 ++++++---
 outputs/runs/RESULTS_MATRIX.md               |  18 ++
 scripts/backfill_onset_meta.py               |  93 +++++++
 scripts/report_negative_composition.py       | 104 ++++++++
 setup.md                                     | 167 ++++++++++++-
 src/pedpredict/config/loader.py              |  40 +++
 src/pedpredict/config/schema.py              |  66 +++++
 src/pedpredict/data/collate.py               |  15 ++
 src/pedpredict/data/lmdb_dataset.py          |  25 +-
 src/pedpredict/data/lmdb_writer.py           |  16 +-
 src/pedpredict/data/onset_backfill.py        | 186 ++++++++++++++
 src/pedpredict/data/onset_stats.py           | 160 ++++++++++++
 src/pedpredict/data/onset_target.py          | 172 +++++++++++++
 src/pedpredict/data/pie_annotations.py       | 118 +++++++++
 src/pedpredict/data/pie_sequences.py         |  39 ++-
 src/pedpredict/data/transforms.py            |   9 +
 src/pedpredict/eval/evaluate.py              |  12 +-
 src/pedpredict/losses/multitask.py           |  24 +-
 src/pedpredict/losses/onset.py               | 173 +++++++++++++
 src/pedpredict/models/ablations.py           |  31 +++
 src/pedpredict/models/cross_attention.py     |  11 +
 src/pedpredict/models/heads.py               |  78 +++++-
 src/pedpredict/training/metrics.py           |  10 +-
 src/pedpredict/training/trainer.py           |   9 +-
 tests/test_data_shapes.py                    |  36 +++
 tests/test_onset_head.py                     | 219 +++++++++++++++++
 tests/test_onset_loss.py                     | 257 ++++++++++++++++++++
 tests/test_onset_plumbing.py                 | 264 ++++++++++++++++++++
 tests/test_onset_stats.py                    | 100 ++++++++
 tests/test_onset_target.py                   | 347 +++++++++++++++++++++++++++
 37 files changed, 3305 insertions(+), 130 deletions(-)
```

## Milestone de35f65

```text
commit de35f65c94a8e367b0a9ef912938855846151825
Author:     Blegarim <nguyenbaoviet25072003@gmail.com>
AuthorDate: Fri Sep 25 10:48:51 2026 +0900
Commit:     Blegarim <nguyenbaoviet25072003@gmail.com>
CommitDate: Fri Sep 25 10:48:51 2026 +0900

    tbh i dont even know man its the paper and the v1 run and the reshuffling and 20 billion other things

 CLAUDE.md                                      |   58 +-
 README.md                                      |   20 +-
 configs/augment.yaml                           |    4 +
 configs/data.yaml                              |    9 +
 configs/model.yaml                             |    5 +
 configs/paths.yaml                             |    1 +
 configs/train.yaml                             |    5 +-
 docs/AUTONOMY_PLAN_2026-09-18.md               |  198 +++++
 docs/METHODOLOGY.md                            |   60 ++
 docs/RECIPE_V2_PLAN.md                         |  457 ++++++++++
 docs/SEED_PLAN_2026-09-21.md                   |   85 ++
 docs/THESIS_ROADMAP.md                         |  110 ++-
 friend-research-guide.md                       |   53 ++
 outputs/diagnostics/detection/curves.json      |  537 ++++++++++++
 outputs/diagnostics/detection/curves.md        |   58 ++
 outputs/diagnostics/detection_true/curves.json |  275 ++++++
 outputs/diagnostics/detection_true/curves.md   |   34 +
 outputs/diagnostics/final_3v3/curves.json      |  563 ++++++++++++
 outputs/diagnostics/final_3v3/curves.md        |   82 ++
 outputs/runs/RESULTS_MATRIX.md                 |  275 +++++-
 outputs/runs/index.csv                         |   34 +
 paper/.gitignore                               |    3 +
 paper/README.md                                |   93 +-
 paper/RESULTS_TO_FILL.md                       |  180 ++++
 paper/check_numbers.sh                         |   61 ++
 paper/main.tex                                 | 1108 ++++++++++++++----------
 paper/refs.bib                                 |   11 +
 scripts/build_feature_cache.py                 |  101 +++
 scripts/diagnose_backbone_bn.py                |  255 ++++++
 scripts/dump_onset_predictions.py              |   78 ++
 scripts/filter_censored_sequences.py           |   71 ++
 scripts/report_detection_curve.py              |  125 +++
 scripts/report_onset_timing.py                 |   79 ++
 scripts/reshuffle_train_lmdb.py                |  113 +++
 setup.md                                       |   26 +-
 src/pedpredict/config/loader.py                |   76 ++
 src/pedpredict/config/schema.py                |   33 +-
 src/pedpredict/data/augment.py                 |   21 +-
 src/pedpredict/data/feature_cache.py           |  437 ++++++++++
 src/pedpredict/data/lmdb_dataset.py            |   90 +-
 src/pedpredict/data/pie_sequences.py           |   33 +-
 src/pedpredict/data/reshuffle.py               |  173 ++++
 src/pedpredict/eval/detection_curve.py         |  270 ++++++
 src/pedpredict/eval/diagnostics.py             |  171 ++++
 src/pedpredict/eval/evaluate.py                |    9 +-
 src/pedpredict/eval/onset_timing.py            |  312 +++++++
 src/pedpredict/models/timm_backbone.py         |   30 +-
 src/pedpredict/paths.py                        |    2 +
 src/pedpredict/training/chunk_loader.py        |   23 +-
 src/pedpredict/training/schedule.py            |   15 +-
 src/pedpredict/training/trainer.py             |    5 +-
 tests/test_bn_diagnostics.py                   |  104 +++
 tests/test_config.py                           |   46 +
 tests/test_data_shapes.py                      |   56 ++
 tests/test_detection_curve.py                  |  175 ++++
 tests/test_feature_cache.py                    |  223 +++++
 tests/test_frozen_backbone.py                  |  112 +++
 tests/test_onset_timing.py                     |  144 +++
 tests/test_reshuffle.py                        |  109 +++
 59 files changed, 7359 insertions(+), 537 deletions(-)
```

## Milestone 5aca9d7

```text
commit 5aca9d760ba6e5613a96f1a1e6f2dc8b466b50df
Author:     Blegarim <nguyenbaoviet25072003@gmail.com>
AuthorDate: Sat Sep 26 15:11:04 2026 +0900
Commit:     Blegarim <nguyenbaoviet25072003@gmail.com>
CommitDate: Sat Sep 26 15:11:04 2026 +0900

    Pixel-free re-run campaign: warm-up fix, resume fix, gate, config-diff check
    
    Code the research-PC re-run campaign (docs/RERUN_PLAN_2026-09-26.md) runs, plus
    the pixel-free read path + input standardization it builds on.
    
    - Pixel-free read (data.visual_input=none) and read-path standardization
      (pose.input_stats; data/input_stats.py, scripts/compute_input_stats.py).
    - Warm-up reads only _meta records when no image is decoded, and the dataset
      opens LMDB without OS readahead then: pixel-free epochs were reading the
      full 83 GB store (JPEGs included) every epoch. Page cache only; batches
      are unchanged. Backward compatible with older spawn parents.
    - Resume restores the early-stop counter and the true best epoch; last.pth
      now carries the post-update patience state.
    - Threshold sweep default widened to 0.01-0.99 @ 0.01 (anchored tuned
      thresholds sat on the old 0.10 floor).
    - eval/rerun_gate.py + scripts/rerun_gate.py: go/no-go for the campaign,
      read from the binary seeds only.
    - config/diff.py + scripts/check_run_config.py: refuse a run whose config
      differs from the reference by anything but its intended flags.
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 CLAUDE.md                               |   8 +-
 README.md                               |  18 +++
 configs/data.yaml                       |   1 +
 configs/eval.yaml                       |   6 +-
 configs/pose.yaml                       |   7 ++
 docs/RERUN_PLAN_2026-09-26.md           | 210 ++++++++++++++++++++++++++++++++
 docs/THESIS_ROADMAP.md                  |   4 +-
 scripts/check_run_config.py             |  51 ++++++++
 scripts/compute_input_stats.py          |  47 +++++++
 scripts/rerun_gate.py                   |  84 +++++++++++++
 src/pedpredict/config/diff.py           |  55 +++++++++
 src/pedpredict/config/loader.py         |  51 +++++++-
 src/pedpredict/config/schema.py         |  24 +++-
 src/pedpredict/data/augment.py          |  20 +++
 src/pedpredict/data/input_stats.py      |  90 ++++++++++++++
 src/pedpredict/data/lmdb_dataset.py     |  33 ++++-
 src/pedpredict/data/lmdb_warm.py        |  30 ++++-
 src/pedpredict/data/pose.py             |  15 ++-
 src/pedpredict/eval/rerun_gate.py       | 136 +++++++++++++++++++++
 src/pedpredict/training/callbacks.py    |  14 +++
 src/pedpredict/training/chunk_loader.py |   9 +-
 src/pedpredict/training/trainer.py      |  29 ++++-
 tests/test_callbacks.py                 |  78 ++++++++++++
 tests/test_chunk_loader.py              |  89 +++++++++++++-
 tests/test_config.py                    |   2 +-
 tests/test_config_diff.py               |  49 ++++++++
 tests/test_lmdb_dataset.py              |  20 +++
 tests/test_pose.py                      | 147 ++++++++++++++++++++++
 tests/test_rerun_gate.py                | 138 +++++++++++++++++++++
 tests/test_runtime_augment.py           |  16 +++
 30 files changed, 1445 insertions(+), 36 deletions(-)
```

## Milestone 8513b28

```text
commit 8513b288de77f55dac33770170172f76c7d25ef0
Author:     Blegarim <nguyenbaoviet25072003@gmail.com>
AuthorDate: Sun Sep 27 06:51:00 2026 +0900
Commit:     Blegarim <nguyenbaoviet25072003@gmail.com>
CommitDate: Sun Sep 27 06:51:00 2026 +0900

    Select best.pth on val crosses_auc; campaign stops reusing F1-selected runs
    
    F1 at the fixed 0.5 cut is noisy here: the sampler trains at ~29% crossing
    while val sits at ~2.8%, so the cut is arbitrary and moves with calibration
    epoch to epoch. On pf_fix_s42 it picked epoch 7 while val AUC peaked at 17,
    and it also drove early stopping.
    
    - train.selection_metric gains "crosses_auc" (rank-based, so immune to the
      prior gap); guarded like crosses_f1 (crosses must be an active task).
    - The rerun gate finds each run's selected epoch with that run's own
      selection metric instead of assuming F1.
    - Plan: every campaign arm selects on crosses_auc, so no F1-selected ladder
      run is reused (20 runs); records the 2026-09-27 gate preview (NO-GO:
      2 of 3 fix seeds peak at epoch 1, then overfit).
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 configs/train.yaml                 |  2 +-
 docs/RERUN_PLAN_2026-09-26.md      | 40 ++++++++++++++++++++++----------------
 scripts/rerun_gate.py              | 12 +++++++++++-
 src/pedpredict/config/loader.py    |  6 +++---
 src/pedpredict/config/schema.py    |  4 +++-
 src/pedpredict/eval/rerun_gate.py  |  8 ++++++--
 src/pedpredict/training/trainer.py |  2 +-
 tests/test_config.py               | 14 +++++++++----
 tests/test_rerun_gate.py           |  9 +++++++++
 tests/test_trainer.py              |  6 ++++--
 10 files changed, 71 insertions(+), 32 deletions(-)
```

## Milestone d4fac3c

```text
commit d4fac3c9fae209a0eb3e298c006865a0a5f3de0b
Author:     Blegarim <nguyenbaoviet25072003@gmail.com>
AuthorDate: Sun Sep 27 07:12:29 2026 +0900
Commit:     Blegarim <nguyenbaoviet25072003@gmail.com>
CommitDate: Sun Sep 27 07:12:29 2026 +0900

    Memorization ladder stage: metadata-only reshuffle, gate AUC summary
    
    The pf_fix preview fails on memorization (2 of 3 seeds best after epoch 1),
    so a second stage now picks a fix before the campaign: base data only /
    sampler off / lr 1e-5, 2 seeds each, adopted by a pre-registered
    validation-only rule (plan section 11).
    
    - reshuffle --meta-only: copy only the _meta records into a *_metaonly dir.
      Pixel-free runs read nothing else, so a store is a few GB, not 54-88 GB.
    - Config validation refuses a *_metaonly train dir to any model that
      decodes images.
    - rerun_gate verdict.json records the mean selected-epoch val AUC (the
      validation-only basis for choosing between passing recipes).
    - CLAUDE.md: crosses_auc selection option and the metadata-only store.
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 CLAUDE.md                        |  6 ++++--
 README.md                        |  2 ++
 docs/RERUN_PLAN_2026-09-26.md    | 40 ++++++++++++++++++++++++++++++++++++++--
 scripts/rerun_gate.py            | 12 +++++++-----
 scripts/reshuffle_train_lmdb.py  |  9 ++++++++-
 src/pedpredict/config/loader.py  | 17 +++++++++++++++++
 src/pedpredict/data/reshuffle.py | 28 +++++++++++++++++++++++-----
 tests/test_rerun_gate.py         |  4 +++-
 tests/test_reshuffle.py          | 30 ++++++++++++++++++++++++++++++
 9 files changed, 132 insertions(+), 16 deletions(-)
```

## Milestone 8c8be99

```text
commit 8c8be99e29ebbe1052c4e15b661202c1e587eb99
Author:     Blegarim <nguyenbaoviet25072003@gmail.com>
AuthorDate: Sun Sep 27 11:06:01 2026 +0900
Commit:     Blegarim <nguyenbaoviet25072003@gmail.com>
CommitDate: Sun Sep 27 11:06:01 2026 +0900

    Plan: campaign launched by manual override (gate NO-GO), memorization ladder held
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 docs/RERUN_PLAN_2026-09-26.md | 5 +++++
 1 file changed, 5 insertions(+)
```

## Milestone 78c5d1b

```text
commit 78c5d1bbee187bd924e53eeb02c1ef89d44f93a5
Author:     Blegarim <nguyenbaoviet25072003@gmail.com>
AuthorDate: Tue Sep 29 10:21:25 2026 +0900
Commit:     Blegarim <nguyenbaoviet25072003@gmail.com>
CommitDate: Tue Sep 29 10:21:25 2026 +0900

    Log the v4 pixel-free campaign: artifacts, generated tables, results ledger
    
    All 20 campaign runs (plus the recipe-development ladder) pulled from the
    research PC and verified byte-identical: run logs, configs, thresholds,
    reports. Prediction dumps and checkpoints stay local (gitignored).
    
    - scripts/report_campaign.py generates every campaign table from logged
      artifacts only: cross-protocol matrix, gap decomposition, window metrics,
      detection curves (both accountings), the pre-registered criterion,
      per-run training summary, and analyses (separability by time to onset,
      causal/centered smoothing, who-vs-when split) -> v4_report/tables/,
      with results.json (every per-seed value) and provenance README.
    - eval/campaign_report.py holds the per-dump analyses, with tests.
    - RESULTS_MATRIX.md: new canonical v4 section; v1 kept as history.
    
    Headline: R3 vs R2 inconclusive (0/5 budgets), matrix and gap replicate
    at n=3 (anchored->streaming AUC 0.516; G_prior -0.000, G_hardneg 0.180).
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 README.md                                          |    4 +
 .../diagnostics/mem_report/per_track/curves.json   |  694 +++
 outputs/diagnostics/mem_report/per_track/curves.md |   94 +
 .../diagnostics/mem_report/per_window/curves.json  |  694 +++
 .../diagnostics/mem_report/per_window/curves.md    |   94 +
 outputs/diagnostics/rerun_gate/verdict.json        |   91 +
 outputs/diagnostics/rerun_gate/verdict.md          |   15 +
 .../diagnostics/v3_report/per_track/curves.json    | 1899 ++++++++
 outputs/diagnostics/v3_report/per_track/curves.md  |  226 +
 .../diagnostics/v3_report/per_window/curves.json   | 1899 ++++++++
 outputs/diagnostics/v3_report/per_window/curves.md |  226 +
 outputs/diagnostics/v4_campaign/OVERRIDE.md        |    6 +
 outputs/diagnostics/v4_report/headline.md          |   11 +
 .../diagnostics/v4_report/per_track/curves.json    | 2200 +++++++++
 outputs/diagnostics/v4_report/per_track/curves.md  |  286 ++
 .../diagnostics/v4_report/per_window/curves.json   | 2200 +++++++++
 outputs/diagnostics/v4_report/per_window/curves.md |  286 ++
 outputs/diagnostics/v4_report/tables/README.md     |   25 +
 outputs/diagnostics/v4_report/tables/analyses.md   |   27 +
 outputs/diagnostics/v4_report/tables/criterion.md  |   83 +
 .../v4_report/tables/detection_per_track.md        |    9 +
 .../v4_report/tables/detection_per_window.md       |    9 +
 outputs/diagnostics/v4_report/tables/gap.md        |   10 +
 outputs/diagnostics/v4_report/tables/matrix.md     |   15 +
 outputs/diagnostics/v4_report/tables/results.json  | 4814 ++++++++++++++++++++
 outputs/diagnostics/v4_report/tables/training.md   |   24 +
 .../diagnostics/v4_report/tables/window_metrics.md |   27 +
 outputs/input_stats/pose58_train_benchmark.json    |  129 +
 outputs/input_stats/pose58_train_shuffled.json     |  129 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  268 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   17 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  272 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   19 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  272 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   24 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   23 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   20 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  272 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   22 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   17 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   17 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   18 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   18 +
 .../resolved_config.yaml                           |  388 ++
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   12 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   24 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   18 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   17 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   18 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   17 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   18 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   22 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   18 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   18 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  390 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   17 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  390 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   31 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   17 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   21 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   18 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   17 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   17 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   18 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   19 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   19 +
 .../eval_log.csv                                   |    5 +
 .../resolved_config.yaml                           |  388 ++
 .../thresholds_anchored.json                       |    9 +
 .../thresholds_streaming.json                      |    9 +
 .../train_distribution.json                        |   35 +
 .../train_log.csv                                  |   18 +
 outputs/runs/RESULTS_MATRIX.md                     |  100 +-
 scripts/report_campaign.py                         |  342 ++
 src/pedpredict/eval/campaign_report.py             |  109 +
 tests/test_campaign_report.py                      |   66 +
 216 files changed, 30770 insertions(+), 1 deletion(-)
```

## Milestone a9f67ff

```text
commit a9f67ff7a70749f33fafc8ba4465befb4c913222
Author:     Blegarim <nguyenbaoviet25072003@gmail.com>
AuthorDate: Tue Sep 29 10:23:36 2026 +0900
Commit:     Blegarim <nguyenbaoviet25072003@gmail.com>
CommitDate: Tue Sep 29 10:23:36 2026 +0900

    Pre-register the anchored-who x streaming-when combination test (before any number)
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 outputs/diagnostics/combo_test/PREREG.md | 26 ++++++++++++++++++++++++++
 1 file changed, 26 insertions(+)
```

## Milestone fd507a2

```text
commit fd507a2777c175450d574c45bcd5bc2ee9969e1f
Author:     Blegarim <nguyenbaoviet25072003@gmail.com>
AuthorDate: Tue Sep 29 10:29:48 2026 +0900
Commit:     Blegarim <nguyenbaoviet25072003@gmail.com>
CommitDate: Tue Sep 29 10:29:48 2026 +0900

    Combination test (pre-registered): anchored who x streaming when is inconclusive
    
    Primary p_anc x p_R2 vs R2 alone, per_window, 3 paired seeds: INCONCLUSIVE
    (42.3 vs 42.3 at 205/hr, 23.1 vs 22.9 at 41/hr; window AUC 0.757 vs 0.783).
    All secondaries (causal intent, val-fitted stack, x R3) also inconclusive.
    
    - scripts/report_combination.py: the test, exactly as PREREG.md (a9f67ff);
      streaming-val dumps of the 9 checkpoints (inference only) feed only the
      val-fitted variant.
    - eval/campaign_report.py: assert_aligned, StackedCombiner, and
      compare_curves (the pre-registered rule, now shared with
      report_campaign.py; criterion.md unchanged).
    - RESULTS_MATRIX.md: result logged under the v4 analyses.
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 README.md                                         |    2 +
 outputs/diagnostics/combo_test/results.json       | 2704 +++++++++++++++++++++
 outputs/diagnostics/combo_test/results.md         |   28 +
 outputs/diagnostics/v4_report/tables/README.md    |    2 +-
 outputs/diagnostics/v4_report/tables/results.json |  160 +-
 outputs/runs/RESULTS_MATRIX.md                    |    7 +
 scripts/report_campaign.py                        |   28 +-
 scripts/report_combination.py                     |  120 +
 src/pedpredict/eval/campaign_report.py            |   67 +-
 tests/test_campaign_report.py                     |   24 +
 10 files changed, 3043 insertions(+), 99 deletions(-)
```

## Selected diff d126c5a

```text
commit d126c5a0bd2988441620a1aeb197757c51773fe4
Author: Blegarim <nguyenbaoviet25072003@gmail.com>
Date:   Thu Jul 2 16:28:24 2026 +0900

    pivot.

diff --git a/docs/project-context-streaming-crossing-onset.md b/docs/project-context-streaming-crossing-onset.md
new file mode 100644
index 0000000..a09ea66
--- /dev/null
+++ b/docs/project-context-streaming-crossing-onset.md
@@ -0,0 +1,188 @@
+# Project Context Brief
+## Online Detection of Pedestrian Crossing-Onset: Bridging Online Action Detection into the PIE/JAAD Intention Subfield, and Decomposing the Anchored→Streaming Performance Gap
+
+> **Purpose of this document.** Self-contained context read for a new standalone research project, meant to be dropped into a fresh project thread so none of the reasoning below has to be re-derived. It is fully technical; nothing is dumbed down. It records not only the proposed angle but the *dead ends already ruled out*, so future-you does not re-tread them. It assumes familiarity with the author's existing pipeline (multimodal ViT + Motion Encoder + Cross-Attention + Ensemble, per-frame multi-task heads, LMDB data, PIE dataset).
+
+**Prepared:** July 2026
+**Working title (one line):** *The pedestrian-intention benchmark never learned to run in a stream — and the case it hides is the one that matters.*
+
+---
+
+## 0. TL;DR
+
+The dominant PIE/JAAD crossing-prediction protocol is **event-anchored**: for each pedestrian track it clips observation at the crossing onset and places a short observation window 1–2 s before a *known* event, producing a per-track binary label and a mild (~2.5:1) class ratio. This is not the condition a deployed model faces. A deployed model is a **streaming detector** monitoring every moment, where the true rate of "a crossing is about to start in the next ~1 s" is far rarer (the author's own dense sliding-window pipeline yields ~37:1) and where the dominant negatives are **hard temporal negatives** — windows of a pedestrian who *will* cross but not yet — which the anchored protocol never generates for training *or* testing.
+
+The contribution is a **bridge + diagnosis**: import the mature **Online Action Detection (OAD) / Online Detection of Action Start (ODAS)** task definition and metrics into the PIE/JAAD intention subfield (which never adopted them despite being ODAS's canonical motivating example), re-evaluate anchored-trained SOTA models under a streaming protocol, and **decompose** the resulting performance gap into (a) a *prior-shift* component that threshold recalibration fixes cheaply and (b) a *residual hard-temporal-negative* component that recalibration cannot fix because the model was never trained or tested on those windows. The decomposition is the rigorous core that elevates this above "we applied ODAS to PIE."
+
+---
+
+## 1. Origin, and the dead ends already ruled out
+
+This angle is the survivor of a long pruning process. Record of what was killed and why, so it is not revisited:
+
+1. **VRU-safety digital twin under imperfect sync (latency safety-cliff + fail-safe controller).** Rejected: the "find the latency threshold past which the twin is untrustworthy" framing is a calibration exercise, not a research question; there is no single latency number (danger depends on how fast the scene is changing, not on data age); and the corrected version (risk-aware control under delayed/dropped state) walks into mature Networked-Control-Systems / estimation territory.
+2. **"Benchmark saturation via aleatoric ceiling."** Weakened: uncertainty (epistemic/aleatoric) is already modeled in crossing prediction (evidential DL, KL/Mahalanobis heads, threshold networks); "nobody has framed the residual as irreducible" is false.
+3. **"Ego-motion shortcut inflates the leaderboard."** Occupied: **LIM ("Less Is More")** makes exactly this its central thesis (ego-vehicle speed causally confounds crossing; *removing* it helps; skeleton-only model with adversarial speed removal). IntFormer notes "most crossing cases in PIE share a similar pattern" (ego-speed). Azarmi's **CAPFI** review quantifies the driver-side bias.
+4. **"Safety-weighted / risk-stratified re-scoring of PIE."** Occupied on multiple axes: **CAPFI** stratifies PIE by pedestrian–ego distance; **"Diving Deeper" (2024)** proposes a per-sample TTE-weighted metric (and found the reweighting *marginal*); **"A Novel Benchmarking Paradigm" (2023)** stratifies PIE *by ego-motion* (speed, yaw, acceleration) — but for *trajectory regression*, not intention classification. The remaining sliver ("crossing-classification × ego-kinematic-danger axis × leaderboard ranking inverts") was judged too thin and one arm's reach from published work.
+5. **"Streaming evaluation is a new idea."** False. OAD/ODAS is a mature subfield (see §3). This is the key correction that *reshaped* the present angle from "I discovered streaming" into "the intention subfield never imported streaming."
+
+**What survived and why.** The present angle is the first that (i) originates from the author's own pipeline friction (the 37:1 imbalance from dense sliding-window sampling), (ii) has a rigorous decomposable core rather than a vibe, (iii) is a *bridge* that inherits ready-made machinery, and (iv) has a clean, honest "why nobody did this" (subfield path-dependence on the 2019 benchmark).
+
+---
+
+## 2. Background primer (concepts, defined)
+
+### 2.1 The two task formulations
+
+**Event-anchored per-track classification (the standard PIE/JAAD protocol; Rasouli/Kotseruba benchmark).**
+- One label per pedestrian *track* (instance).
+- For crossing pedestrians, observation is **clipped at the first crossing frame** (the `crossing_point` tag); for non-crossers, at the last visible frame.
+- The observation window (commonly 16 frames ≈ 0.5 s) is sampled so its **last frame sits 1–2 s (TTE 30–60 frames) before the event**, with overlap (0.5–0.8) for augmentation.
+- Class ratio ≈ 2.5:1 non-crossing:crossing (a *track-level* ratio, reflecting how many pedestrians eventually cross).
+- Train/val/test are produced by splitting the resulting samples (PIE: sets 01/02/06 train, 04/05 val, 03 test). **The test set carries the same anchored distribution and the same structural omission** (see §2.3).
+
+**Streaming online detection of crossing-onset (the proposed / author's formulation).**
+- Dense sliding window across the entire track.
+- Label per window: does a crossing **onset** fall within the next H frames (e.g., observe 1–20, predict onset in 21–50)?
+- Windows where crossing already began during observation are discarded (correct: you cannot "anticipate" an onset already underway).
+- Class ratio ≈ 37:1 (a *temporal-density* ratio, reflecting how rare an imminent onset is per observed moment).
+
+### 2.2 Why the ratios differ (mechanism — state this precisely, it is the crux)
+The anchored 2.5:1 is a **sampling artifact**: taking ~one labeled window per track, anchored right before the event, never samples the many "nothing imminent yet" windows earlier in each track. The streaming 37:1 reflects the **true base rate** of imminent onsets in continuous observation. They are not two estimates of the same quantity; they measure different things (track-level class balance vs. per-window onset density).
+
+### 2.3 The hard temporal negative (the safety-critical case the protocol omits)
+Streaming negatives are of three kinds:
+1. **Genuine non-crossers** (waiting, walking along) — also present anchored.
+2. **Hard temporal negatives** — windows of a pedestrian who *will* cross, but not for another few seconds. Same person, same appearance, same scene, labeled negative *now* only because the onset is beyond horizon H. **The anchored protocol, by clipping at onset, generates these for neither training nor testing.** These are exactly the windows a deployed detector must not false-alarm on, and they are a genuine *likelihood* (discrimination) problem, not a *prior* problem.
+3. **Junk** — occlusion, sensor noise, far/low-information windows. This is the fraction that drowns naive training on raw 37:1.
+
+**Why this matters for the author's past failure:** feeding raw, unmanaged 37:1 to a from-scratch, data-hungry ViT collapses training — the positive gradient is negligible before crossing features are learned. The fix is *curated* streaming training (emphasize hard temporal negatives, down-weight/filter junk, pretrained backbone, focal/weighted objective), not abandoning the real distribution.
+
+### 2.4 Online Action Detection (OAD) and Online Detection of Action Start (ODAS)
+Mature action-recognition subfields for streaming, per-frame settings:
+- **OAD**: per-frame labeling of a streaming video using only current + past frames (no future access). Datasets: TVSeries, THUMOS14, HDD. Methods: TRN, OadTR, RED, FATSnet. Metric: per-frame mAP; **calibrated AP (cAP/mcAP)** to handle heavy background.
+- **ODAS**: detect the *onset* of an action instance as early as possible in an untrimmed stream with large background. Methods: StartNet. Metric: **point-level AP (p-AP)** with temporal-offset tolerance.
+- **Critical fact:** ODAS/OAD papers repeatedly cite *pedestrian crossing* as the canonical safety-critical motivating example, yet evaluate on TV/sports/ego-driver-maneuver data (HDD's "crosswalk passing" is the *ego-vehicle driver's* action, not the pedestrian's). The rigor exists; it was never applied to pedestrian-crossing *intention*.
+
+### 2.5 The base-rate fallacy (why "good discrimination" ≠ "deployable")
+Discrimination (AUC) and operating-point usefulness (precision at a threshold) diverge under base-rate shift. Worked example, same model (recall 0.9, FPR 0.1):
+- At 2.5:1 (crossing ≈ 28.6%): precision ≈ 0.9·0.286 / (0.9·0.286 + 0.1·0.714) ≈ **0.78**.
+- At 37:1 (crossing ≈ 2.6%): precision ≈ 0.9·0.026 / (0.9·0.026 + 0.1·0.974) ≈ **0.19**.
+Four of five alarms become false purely from the prior. Part of this is *fixable by recalibration* (shift threshold / prior-correct); the hard-temporal-negative part is *not*.
+
+---
+
+## 3. The precise thesis
+
+> The pedestrian-crossing-intention subfield inherited a 2019 event-anchored classification protocol and never adopted the streaming/ODAS evaluation rigor developed — in a sibling subfield — for exactly this safety-critical action-start problem. Reformulating PIE crossing prediction as online detection of crossing-onset (a) exposes a true base rate (~37:1) that the anchored protocol hides, (b) reveals a class of hard temporal negatives absent from both anchored training and anchored *testing*, and (c) causes anchored-trained SOTA models to degrade. That degradation decomposes into a prior-shift component (recalibration fixes it) and a residual hard-temporal-negative component (recalibration cannot). The deliverable is a streaming evaluation protocol for the subfield plus a quantified decomposition of what the standard protocol has been systematically failing to measure.
+
+---
+
+## 4. The decomposition (the rigorous core)
+
+For a model trained anchored and evaluated streaming, total deployment gap G = G_prior + G_hardneg, where:
+- **G_prior** = the portion recovered by applying the optimal prior/threshold correction to the true streaming base rate (Bayesian prior correction or validation-set threshold re-selection at the true prior).
+- **G_hardneg** = the residual after correction, measured against a model *trained* streaming (curated) and evaluated streaming. This residual is attributable to the hard-temporal-negative discrimination problem the anchored model never learned.
+
+Headline number = the fraction of G explained by each. Interpretation:
+- If G ≈ G_prior (recalibration recovers most): the author's own intuition wins, contribution shrinks to "remember to recalibrate" (a known trick → demote to a cautionary short paper).
+- If G_hardneg is large: the protocol systematically under-trains and under-tests the safety-critical case → full paper.
+
+This decomposition is also the pre-emptive answer to the strongest reviewer objection ("just recalibrate the threshold"): the paper *measures* whether that is true.
+
+---
+
+## 5. Plan of attack (phased)
+
+**Phase 0 — Final kill-check (do first, cheap).** Search specifically for ODAS / online action detection already applied to PIE/JAAD *pedestrian-crossing intention* (not ego-vehicle maneuvers). Surfaced so far: ODAS-on-driving via HDD = ego-driver actions (different task). If a direct PIE-crossing-as-ODAS paper exists, re-scope. If not, runway is clear.
+
+**Phase 1 — Day-one empirical de-risk.** Take 2–3 published models with released code (e.g., PCPA, SF-GRU, one recent transformer). Train anchored (standard). Evaluate on both anchored and streaming test sets (start with the "eventual crossing" label to isolate prior effects). Check: does precision collapse at the true prior, and does simple threshold recalibration recover it? Result tells go/no-go in days. (The author's own per-frame multi-task heads make the streaming harness cheap.)
+
+**Phase 2 — Formalize both samplers + characterize negatives.** Implement the anchored sampler (benchmark-faithful) and the streaming sampler (dense sliding window, onset-in-horizon label). **Tag streaming negatives into the three types (§2.3) and report the composition** (fraction hard-temporal vs. junk vs. genuine non-crosser). This breakdown is itself unpublished and is a sub-contribution.
+
+**Phase 3 — Cross-protocol matrix (the spine table).** For each of N models (RNN, GRU, transformer, the author's ViT+Motion+Cross-Attn ensemble), run {train anchored, train streaming} × {test anchored, test streaming}. Report **discrimination** (AUC, average precision) *separately from* **operating-point** metrics (precision/recall/F1 at the true-prior-recalibrated threshold) *and* **ODAS metrics** (per-frame mAP / calibrated AP, point-level AP with offset tolerance).
+
+**Phase 4 — Decomposition (§4).** In the train-anchored/test-streaming cell, apply optimal prior correction, then measure residual vs. train-streaming/test-streaming. Report G_prior vs. G_hardneg fractions. This is the headline.
+
+**Phase 5 — Constructive close (turns critique into a tool).** Ship (i) the streaming ODAS-style evaluation protocol for the subfield and (ii) the curated hard-negative streaming training recipe; show it narrows G_hardneg. Benchmark-critique reviewers want a fix, not just a diagnosis.
+
+**Supporting analyses.** (a) Confirm residual failures concentrate in the "will-cross-soon" windows (validates the hard-negative story). (b) Sweep horizon H — base rate and difficulty both move with H; use this to distinguish streaming from a mere large-TTE sweep (see §7). (c) Check whether models lean *harder* on the ego-speed shortcut under streaming (quietly reconnects the LIM/CAPFI confound thread as a secondary finding).
+
+---
+
+## 6. Methods & tools
+
+| Tool / concept | Role | Priority |
+|---|---|---|
+| Author's existing PIE pipeline (ViT + Motion + Cross-Attn + Ensemble, per-frame heads, LMDB) | Streaming harness + one of the N models | Essential, in hand |
+| Anchored sampler (Rasouli/Kotseruba protocol) | Baseline formulation; must be benchmark-faithful | Essential |
+| Streaming/dense sampler (onset-in-horizon) | The proposed formulation | Essential, in hand |
+| Published baselines w/ code (PCPA, SF-GRU, a transformer) | Multi-model evidence for a *protocol* claim | Essential |
+| ODAS/OAD metrics: per-frame mAP, calibrated AP (cAP/mcAP), point-level AP | Import from OAD; the correct streaming metrics | Essential (read + implement) |
+| Prior correction / threshold recalibration (Bayesian prior shift, val-set threshold selection) | Isolates G_prior | Essential |
+| Focal loss / class-balanced loss / curated hard-negative mining | Curated streaming training (§2.3) | Essential |
+| TRN / OadTR as reference streaming architectures | Optional strong streaming baselines | Optional v2 |
+
+---
+
+## 7. Risks & reviewer objections (pre-empt in the writing)
+
+1. **"Streaming eval is old (OAD/ODAS)."** True — do not claim to invent it. Position as *importing* mature rigor into a siloed subfield that skipped it due to path dependence on the 2019 benchmark. Novelty = the bridge + the decomposition + the field-wide blind-spot demonstration, not the streaming idea.
+2. **"You just renamed the TTE sweep."** GTransPDM shows PIE accuracy → 99%+ as TTE→0; a reviewer will equate streaming with large-TTE. **Rebuttal (must show, not assert):** large-TTE is still *one anchored window per track, placed a known distance before a known event*; streaming is *dense windows, no event anchor, flooded with hard temporal negatives*. The difference is the **negative distribution**, not the horizon. Demonstrate via the negative-composition analysis (Phase 2) and the H-sweep (Phase 5b).
+3. **"Just recalibrate the threshold."** Answered structurally by the decomposition (§4) — the paper measures exactly how much recalibration recovers.
+4. **Nearest neighbor — Coupling-Intent's "two sampling settings."** Setting 1 ("all original data") includes windows long before the event and notes they are easy; that is the closest prior observation. Cite and differentiate: they used whole-track windows with an *eventual-crossing* label and dismissed early windows as boring; this work uses dense *onset-in-horizon* labeling and argues the early windows of *eventual crossers* are the hard, safety-critical negatives.
+5. **"Someone already bridged ODAS→PIE."** Residual risk; resolve in Phase 0.
+6. **Labor/reproducibility.** Retraining several published models is real work and depends on their code cooperating. Budget for it; prefer models with maintained repos.
+7. **Single-dataset validity.** PIE is Toronto/daylight/clear only. Consider replicating the core matrix on JAAD (note: JAAD lacks numeric ego-speed) to show the effect is not PIE-specific.
+
+**Realistic venues:** IV, ITSC, WACV, or an autonomous-driving / safe-ML workshop. A strong, clean decomposition could reach a main CV or robotics track.
+
+---
+
+## 8. Open sub-questions / natural sequels
+- Formalize crossing-onset detection with a proper ODAS metric suite as a *new public evaluation track* on PIE/JAAD (community artifact).
+- Time-to-cross **regression** framing as a complement that sidesteps the binary prior entirely — compare against the detection framing.
+- Cost-sensitive / decision-theoretic operating points: pick thresholds by a braking-cost vs. miss-cost model rather than F1.
+- Curriculum over the hard-temporal-negative horizon (train from easy far-from-onset to hard near-onset negatives).
+- Interaction with the ego-speed shortcut (LIM/CAPFI): does streaming make the shortcut more or less load-bearing?
+- Correlated/bursty degradation (occlusion runs) as a distinct hard-negative subtype.
+
+---
+
+## 9. Key papers to read first (grouped)
+
+**The protocol you are critiquing**
+- Rasouli et al., *PIE: A Large-Scale Dataset and Models…* (ICCV 2019) — dataset + original per-track protocol, ~2.5:1.
+- Kotseruba et al., *Benchmark for Evaluating Pedestrian Action Prediction* (WACV 2021) — the standard sampling/splits everyone follows; TTE-vs-accuracy effect; easy/medium/hard sample analysis.
+
+**Nearest neighbors (know cold; differentiate)**
+- *Coupling Intent and Action for Pedestrian Crossing Behavior Prediction* — the "two sampling settings" observation (closest prior thought); Naive-baseline-beats-SOTA note.
+- *Diving Deeper Into Pedestrian Behavior Understanding* (2024) — per-sample TTE-weighted metric (found marginal); balanced-accuracy/mAP.
+- Azarmi et al., *Feature Importance… CAPFI* (2024) and *…via Vision-Language Foundation Models* (2025) — context/distance stratification; ego-speed driver-side bias.
+- *A Novel Benchmarking Paradigm…* (2023) — ego-motion scenario stratification, but for trajectory regression.
+- *Causal Confusion… The Role of Ego-Vehicle Speed* (LIM, "Less Is More") — the ego-speed shortcut, claimed and "fixed."
+- GTransPDM — TTE→0 gives ~99% accuracy (the loaded gun for objection #2).
+
+**The rigor to import (OAD/ODAS)**
+- Xu et al., *Temporal Recurrent Network (TRN)* — online detection + anticipation.
+- Wang et al., *OadTR: Online Action Detection with Transformers* — per-frame mAP / calibrated AP.
+- Shou et al., *Online Detection of Action Start (ODAS/StartNet)* — point-level AP; cites pedestrian crossing as motivation.
+- De Geest et al. (TVSeries) — origin of OAD + calibrated AP.
+- A survey on online action detection / action anticipation — for metric definitions and framing.
+
+---
+
+## 10. Glossary (plain definitions)
+- **Event-anchored sampling:** placing the observation window a fixed time before a *known* crossing event; clips at onset; yields ~2.5:1.
+- **Streaming / online detection:** per-window (or per-frame) decision using only current+past frames, over the whole track; yields the true ~37:1 onset rate.
+- **Crossing onset:** the first frame the pedestrian begins crossing (`crossing_point`).
+- **Hard temporal negative:** a window of a pedestrian who will cross, but not within horizon H — labeled negative now; absent from anchored data.
+- **Base rate / prior:** the fraction of positive windows in the evaluated distribution.
+- **Prior shift / base-rate fallacy:** good discrimination (AUC) can coexist with poor precision when the prior is small; part fixable by recalibration.
+- **Prior correction / recalibration:** adjusting the decision threshold (or posterior) to the true deployment prior without retraining.
+- **TTE (time-to-event):** frames between last observation and the event; the anchored protocol fixes it to 1–2 s.
+- **Horizon H:** the anticipation window in the streaming formulation ("onset within next H frames").
+- **OAD (Online Action Detection):** per-frame labeling of a stream; metric per-frame mAP / calibrated AP.
+- **ODAS (Online Detection of Action Start):** detect the onset as early as possible in an untrimmed stream; metric point-level AP.
+- **Calibrated AP (cAP):** AP variant that compensates for heavy background/negative imbalance (from TVSeries/OAD).
+- **G_prior / G_hardneg:** decomposition of the anchored→streaming gap into the recalibration-fixable part and the residual hard-negative part.
```

## Selected diff b9a381f

```text
commit b9a381fd2eed77ff61a6f6f98371da99e265a559
Author: Blegarim <nguyenbaoviet25072003@gmail.com>
Date:   Thu Aug 27 16:25:04 2026 +0900

    new head. new direction. new day. new me.

diff --git a/docs/THESIS_ROADMAP.md b/docs/THESIS_ROADMAP.md
index 8a18336..4f28069 100644
--- a/docs/THESIS_ROADMAP.md
+++ b/docs/THESIS_ROADMAP.md
@@ -1,12 +1,12 @@
 # Thesis Roadmap — Streaming Crossing-Onset Detection
 
-**Prepared:** July 2026 · **Last swept:** 2026-08-19.
+**Prepared:** July 2026 · **Last swept:** 2026-08-20.
 
 > **What this document is.** The single, findable, end-to-end checklist for the whole thesis — from the
 > pre-migration prototype through to the defense — plus the supporting-study spokes at the bottom. It is
 > the **tracker**. The *why* lives in two companions and is not repeated here:
 > - **The argument** — [`project-context-streaming-crossing-onset.md`](project-context-streaming-crossing-onset.md): the thesis case, the dead ends already ruled out, the reviewer objections. Frozen reference.
-> - **The method** — [`METHODOLOGY.md`](METHODOLOGY.md): the three prongs and how they were chosen. The active working reference.
+> - **The method** — [`METHODOLOGY.md`](METHODOLOGY.md): onset timing under censoring, plus the two supporting parts and how they were chosen. The active working reference.
 >
 > The numbers live in [`RESULTS_MATRIX.md`](../outputs/runs/RESULTS_MATRIX.md), which is authoritative for
 > every figure — this file never restates a metric it does not own.
@@ -18,6 +18,15 @@ is *done* (Stage 4). But on its own that is a negative result: "the usual benchm
 became the **motivation**, and the contribution is what follows from it: *a way to train for streaming
 crossing-onset that the standard setup cannot produce.*
 
+**The method has a spine (2026-08-20).** It is no longer three parallel prongs: the contribution is to
+treat streaming crossing prediction as **onset timing under censoring** rather than yes/no classification
+at a fixed horizon. What forced the change was an objection with no good answer inside the binary framing
+— a pedestrian who walks normally and then turns abruptly is *unpredictable* a few seconds out, so part
+of the confusing population is not hard but impossible, and any claim to "handle the hard negatives"
+overclaims. Dropping the fixed cut-off removes that population by construction instead of fighting it.
+Pose-movement features and the online-action-detection material stay, in support. Full reasoning, and the
+measurements that back it, in [`METHODOLOGY.md`](METHODOLOGY.md).
+
 Three things follow, and they govern how to read the stages below:
 
 - **The centre of gravity is Stage 7**, not Stage 4. Critical path: **4 (done) → 6 → 7 → 8**.
@@ -42,11 +51,11 @@ checkable.
 | **0** | Prototype → clean rebuild (behavior-preserving port) | ✅ done |
 | **1** | Engineering audit + v2 data-contract *code* | ✅ done |
 | **2** | v2 data regeneration + baseline runs | ✅ done (streaming + anchored builds exist, trained) |
-| **3** | Streaming pivot: onset metadata + protocol switch + pose arm | 🟡 code + pose extraction done · onset fields still stranded (see below) |
+| **3** | Streaming pivot: onset metadata + protocol switch + pose arm | 🟡 code done incl. onset plumbing + head + loss (2026-08-26) · awaiting the lab-PC backfill run |
 | **4** | The decomposition (G_prior / G_hardneg) — **now the motivation, not the headline** | ✅ measured + written up ([RESULTS_MATRIX.md](../outputs/runs/RESULTS_MATRIX.md)) |
 | **5** | Streaming-leg convergence + baseline hygiene | 🟡 demoted from headline — one real config mismatch found, must be fixed |
-| **6** | Rare-event metrics + negative-composition report | 🟢 **NEXT — all 💻, no GPU, no data** |
-| **7** | **The method** (was "constructive close") + supporting studies | 🟢 **the thesis now lives here** → [METHODOLOGY.md](METHODOLOGY.md) |
+| **6** | Rare-event metrics + negative-composition report | 🟡 composition + horizon sweep ✅ done 2026-08-20 · metric suite still 💻 **NEXT** |
+| **7** | **The method** — onset timing under censoring + supporting studies | 🟢 **the thesis now lives here** → [METHODOLOGY.md](METHODOLOGY.md) |
 | **8** | Write-up, defense, release | ⬜ not started |
 
 The critical path is now **6 → 7 → 8** (Stage 4 is done). Stage 6 comes first because it is entirely
@@ -58,8 +67,10 @@ mismatch that would corrupt any comparison built on it.
 
 **The two dependencies worth knowing before planning any lab visit:**
 - The three onset fields never reach the database the trainer reads (Stage 3, last unchecked item; the gap
-  is described in CLAUDE.md § Data Pipeline, S1 bullet). Two of the four candidate method directions cannot
-  start until they do. The fix is small and mostly 💻.
+  is described in CLAUDE.md § Data Pipeline, S1 bullet). Since 2026-08-20 this blocks **the method itself**,
+  not just two of four candidate directions — onset timing has nothing to train against until the fields
+  arrive. The fix is small and mostly 💻. *(It does not block the negative-composition report, which is
+  done: that reads the sequence pkls or PIE's annotation XMLs, both upstream of the packing step.)*
 - Nothing in Stages 6–7 needs a *training* run to make progress. Metrics, features, and the negative
   census are all written and tested on the laptop; the lab machine only executes.
 
@@ -128,13 +139,14 @@ The July reframe. Code is landed; the pose extraction pass ran on the lab PC.
 - [x] 💻 `future_observed` (`n − end`) — makes the H-sweep rigorous against right-censoring
 - [x] 💻 `track_crosses` (track ever crosses) — separates genuine non-crosser from will-/already-crossed
 - [x] 💻 Tests: genuine-non-crosser, in-horizon positive, hard-temporal-negative, already-crossed
-- [ ] 💻+🖥️ **Get the three onset fields into the trainer's reach — the one blocking item left in Stage 3.**
-      They are computed but dropped before the trainer can see them; the gap and its three-part fix are
-      stated once, in the **S1 bullet of [CLAUDE.md](../CLAUDE.md) § Data Pipeline**. Blocks two of the four
-      candidate method directions. Sub-checkboxes:
-      - [ ] 💻 (a) add the three fields to `pack_meta` + the read path, so future builds carry them
-      - [ ] 💻 (b) write the backfill patch script for the existing LMDBs
-      - [ ] 🖥️ (c) run it (fast metadata pass — images untouched)
+- [x] 💻+🖥️ **Get the three onset fields into the trainer's reach.** Done 2026-08-26 except the lab-PC run.
+      - [x] 💻 (a) `pack_meta` + read path + collate (the fields ride in `labels`, so no new Trainer wiring)
+      - [x] 💻 (b) backfill script — `scripts/backfill_onset_meta.py`, verifies `track_id` + `crosses`
+            per sample before writing, aborts on a mismatched pkl
+      - [ ] 🖥️ (c) **run it** (metadata-only pass — images untouched). Stop any training job first:
+            Windows refuses a write-open while a chunk is memory-mapped. `--dry-run` verifies safely.
+      - [ ] 🖥️ (d) re-run `augment_dataset.py` so `preprocessed_train_aug` inherits the keys
+            (augmented dirs are not backfillable — oversampling breaks the positional sample→record map)
 
 **Protocol switch (`data.protocol`)**
 - [x] 💻 `data.protocol={streaming,anchored}` resolved once in `paths.protocol_lmdb_dirs`
@@ -242,35 +254,66 @@ is cheap and decides how much is on the table.
 - [ ] 💻 Unit tests on synthetic streams (no data)
 - [ ] 🖥️ Apply the suite across the matrix cells
 
-**Negative-composition report (brief §2.3, from S1)**
-- [ ] 💻 Classify every streaming negative into: genuine non-crosser / hard-temporal / already-crossed (from `track_crosses` + `onset_offset`)
-- [ ] 💻 Pick + implement a **junk** signal (bbox height below threshold / occlusion / low `future_observed`)
-- [ ] 🖥️ Report the composition (fraction hard-temporal vs junk vs genuine) — the "what the benchmark omits" figure
+**Negative-composition report (brief §2.3, from S1)** — ✅ **done 2026-08-20**, and it landed on the
+laptop rather than the lab PC: PIE's behaviour tags are a ~25 MB XML set, so the whole label contract can
+be re-derived without frames, LMDBs or a GPU. Numbers in [METHODOLOGY.md](METHODOLOGY.md) § What the
+negatives are actually made of.
+- [x] 💻 Classify every streaming negative into: genuine non-crosser / hard-temporal / already-crossed (from `track_crosses` + `onset_offset`) — `data/onset_stats.py`
+- [x] 💻 Report the composition — **hard-temporal is 25.7% of train / 39.2% of test windows**, i.e. 8.9× and 12.8× the positives; the direction is confirmed
+- [x] 💻 Bucket it by time-to-onset — two-thirds of that mass sits >5 s out; the genuinely confusable 1–3 s band is ~1.7× the positive set
+- [x] 💻 **Horizon sweep** (was filed under Stage 7) — what a longer horizon buys and costs; justifies keeping ~1 s
+- [x] 💻 CLI + unit tests + a laptop-side drift gate on the golden counts (`scripts/report_negative_composition.py`, `tests/test_onset_stats.py`)
+- [ ] 💻 Pick + implement a **junk** signal (bbox height below threshold / occlusion) — the one composition axis still unmeasured
 - [ ] 🖥️ **Confirm residual failures concentrate in the will-cross-soon windows** (validates the hard-negative story — brief §5 supporting analysis a)
+- [ ] 🖥️ **The ceiling measurement** — separability stratified by time-to-onset; the distance at which prediction decays to chance. New, and it is what keeps the claim honest.
+
+**Train/test asymmetry found while doing this.** The two splits differ structurally in what their
+negatives are made of (25.7% vs 39.2% hard-temporal; 64.3% vs 53.4% never-crossers), so a train-to-test
+difference is not purely a generalisation gap. Caveat recorded in [RESULTS_MATRIX.md](../outputs/runs/RESULTS_MATRIX.md).
 
 ---
 
 ## Stage 7 — 🟢 THE METHOD (was "constructive close") + supporting studies
 
 **This is where the thesis contribution now lives.** It used to be the optional upgrade; after the August
-reframe it is the centre. The detailed direction — three prongs, what has been tried in the literature, what
+reframe it is the centre. The detailed direction — the method and its two supporting parts, what has been tried in the literature, what
 transfers, the open decision points — is in **[`METHODOLOGY.md`](METHODOLOGY.md)**, which is the working
 reference. Kept here in summary so the tracker stays complete:
 
-1. **Pose-motion features** — turn the raw keypoints into features that describe *movement*, not just
-   posture. All current pose features are single-frame. 💻, no re-extraction needed.
-2. **Onset-time supervision** — get the three onset fields into the trainer's reach, then teach the model
-   *when*, not just *whether*. 💻 plumbing + a small lab pass.
-3. **A rare-event mechanism imported from online action detection** — that community named this exact
-   failure in 2018 and proposed fixes for it; the two literatures do not currently talk to each other.
-
-The items below are the pre-reframe version of this stage. They remain valid work; METHODOLOGY.md is what
-sequences them now.
-
-**Curated streaming recipe (narrows G_hardneg — brief Phase 5)**
+1. **Onset timing under censoring — the method.** Get the three onset fields into the trainer's reach,
+   then predict *when* the crossing starts, treating a window whose future ran out as a censored
+   observation rather than a negative. This removes the confusing case by construction, recovers the
+   windows M4 currently discards, and lets any horizon be read off afterwards. 💻 plumbing + a small lab
+   pass. **Hard constraint: it must still emit the probability of onset within 32 frames**, or the four
+   baseline runs stop being comparable.
+2. **Pose-motion features** — turn the raw keypoints into features that describe *movement*, not just
+   posture. All current pose features are single-frame. 💻, no re-extraction needed. This is what gives a
+   timing model something to read at a one-second horizon.
+3. **Online-action-detection material** — that community named this exact failure in 2018, and solved the
+   measurement problem for it. Its metrics are the instrument (Stage 6); its objectives are available
+   where they fit; its obvious alternatives (focal, class-balanced) are the comparison baselines a methods
+   thesis has to run rather than dismiss.
+
+**Onset-timing build order** (supersedes the pre-reframe recipe list below)
+- [x] 💻 S1 fields into `pack_meta` + the read path + collate
+- [x] 💻 Backfill script over the existing LMDBs — 🖥️ **still to run**
+- [x] 💻 Timing output + a censoring-aware loss (`model.onset_head`, `losses/onset.py`; default off).
+      Contracts and the three-arm weight table live in **[CLAUDE.md](../CLAUDE.md) § Onset Timing**.
+- [x] 💻 Conversion back to the baseline question — `hazard_to_horizon_logits`, pinned against the
+      generator's own `crosses` label so the four baselines stay comparable
+- [ ] 🖥️ Short smoke run to check the head does not collapse to `h ~ 0` (the predicted failure mode)
+- [ ] 🖥️ Auxiliary-arm run first (reported number still from `crosses_frame` = baselines untouched),
+      then the pure-reformulation arm
+- [ ] 💻+🖥️ Censored windows restored — needs a **regen** with the M4 filter relaxed, NOT the backfill
+      (`window_track` skips them at generation, so they were never in the pkls). Separate experiment.
+- [ ] 🖥️ Restore censored windows to training — a data-quantity change, measured separately from the objective change
+- [ ] 💻 Two literature checks before any novelty wording: onset time as a training signal, and censoring-aware modelling, both specifically for crossing prediction
+
+**Curated streaming recipe (narrows G_hardneg — brief Phase 5).** Pre-reframe; still valid work, now
+comparison baselines rather than the plan.
 - [ ] 🖥️ Pretrained backbone (RQ1) + focal/class-balanced loss + hard-negative emphasis + junk filtering
 - [ ] 🖥️ Show the recipe measurably narrows G_hardneg vs the naive streaming-trained model
-- [ ] 🖥️ **H-sweep** (near-free via S1) — base rate + difficulty move with horizon H → rebuts objection #2 ("just the TTE sweep")
+- [x] 💻 **H-sweep** — done 2026-08-20 as a label-side study (no training needed); rebuts objection #2 and justifies keeping ~1 s. The *trained* version (does difficulty actually move with H?) is still open.
 - [ ] 🖥️ Ego-speed-under-streaming leakage probe (RQ4) — does streaming lean *harder* on the shortcut?
 
 ---
@@ -313,7 +356,7 @@ never against each other. Screen at 1 seed, confirm finalists at 3, report mean
 - [ ] 💻 Related-work survey (brief §9 reading list; know the nearest neighbors cold — Coupling-Intent, Diving-Deeper, LIM, GTransPDM)
 - [ ] 💻 Draft: intro + the two-formulation background + the base-rate-fallacy primer
 - [ ] 💻 Draft: **motivation** — the two protocols, the decomposition, and why re-tuning the threshold cannot fix it (this is the old "results headline", moved forward to justify the method)
-- [ ] 💻 Draft: methods (both samplers, the protocol, the metric suite, the decomposition math, **and the three prongs from [METHODOLOGY.md](METHODOLOGY.md)**)
+- [ ] 💻 Draft: methods (both samplers, the protocol, the metric suite, the decomposition math, **and the onset-timing method from [METHODOLOGY.md](METHODOLOGY.md)**)
 - [ ] 💻 Draft: results (the method measured against the four baseline runs, the negative-composition figure, the H-sweep)
 - [ ] 💻 Draft: pre-empt the reviewer objections (brief §7) — especially "just recalibrate" (answered by §4) and "just the TTE sweep" (answered by H-sweep)
 - [ ] 💻 Discussion + limitations + the constructive close (protocol + recipe as community artifacts)
```

## Selected diff 8513b28

```text
commit 8513b288de77f55dac33770170172f76c7d25ef0
Author: Blegarim <nguyenbaoviet25072003@gmail.com>
Date:   Sun Sep 27 06:51:00 2026 +0900

    Select best.pth on val crosses_auc; campaign stops reusing F1-selected runs
    
    F1 at the fixed 0.5 cut is noisy here: the sampler trains at ~29% crossing
    while val sits at ~2.8%, so the cut is arbitrary and moves with calibration
    epoch to epoch. On pf_fix_s42 it picked epoch 7 while val AUC peaked at 17,
    and it also drove early stopping.
    
    - train.selection_metric gains "crosses_auc" (rank-based, so immune to the
      prior gap); guarded like crosses_f1 (crosses must be an active task).
    - The rerun gate finds each run's selected epoch with that run's own
      selection metric instead of assuming F1.
    - Plan: every campaign arm selects on crosses_auc, so no F1-selected ladder
      run is reused (20 runs); records the 2026-09-27 gate preview (NO-GO:
      2 of 3 fix seeds peak at epoch 1, then overfit).
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

diff --git a/docs/RERUN_PLAN_2026-09-26.md b/docs/RERUN_PLAN_2026-09-26.md
index 7295b58..0eacb14 100644
--- a/docs/RERUN_PLAN_2026-09-26.md
+++ b/docs/RERUN_PLAN_2026-09-26.md
@@ -5,6 +5,16 @@ PC waits for the pixel-free ladder (`queue_v3.sh`, tmux `queue3`) to finish, run
 once, and on GO starts `queue_v4.sh` in tmux `queue4`. Nobody has to be watching. On NO-GO nothing
 launches, and the verdict says which check failed (§7).
 
+> **2026-09-27 update.** (1) Every campaign arm now selects `best.pth` (and early-stops) on **val
+> `crosses_auc`**, not F1 at the fixed 0.5 cut. Keeping F1 (old D4) was a mistake: its revisit condition
+> fired on `pf_fix_s42` (F1 picked epoch 7, the AUC peak was epoch 17) and was not acted on. As a result no
+> ladder run is reused (they were F1-selected), and the campaign trains all **20** runs itself.
+> (2) **Gate preview on the three `pf_fix` seeds: NO-GO.** `pf_fix_s43`/`s44` peak at **epoch 1** (val AUC
+> 0.870 / 0.854, trained at a tenth of the learning rate during warmup), then fall to 0.76 / 0.73 while train
+> loss drops 0.54 → 0.06: memorization, which no selection metric fixes. The seed spread IS fixed, though:
+> detection sd 3.1 pp at 205/hr (v1: 18.5), mean 20.2% at 41/hr (v1 binary 11.9%, GBM 38.9%), test AUC 0.785.
+> Unless overridden, the campaign will not launch.
+
 **The idea.** The paper's claim is about the *training objective* (binary vs hazard), not the architecture.
 If the pixel-free model (`pose_kinematics`: the 58-number pose + motion vector per frame, no images) trains
 stably, it becomes the canonical model, and every run the paper cites is redone on it: the motivation
@@ -70,13 +80,13 @@ GBM-probe ceiling (38.9 ± 1.0% at 41/hr) are reported in `v3_report`, not gated
 
 ## 3. The canonical recipe
 
-Exactly `pf_fix` from `queue_v3.sh`, so the ladder's runs can be reused. The arm flags live in
+`pf_fix` from `queue_v3.sh` with one change: selection on val `crosses_auc` (2026-09-27). The arm flags live in
 `/workspace/setup_logs/recipes_v4.sh`:
 
 ```
 PF      eval.model_type=pose_kinematics  pose.enabled=true  model.motion_norm=none
         data.motion_dim=58  model.motion_dim=58  data.visual_input=none
-COMMON  train.active_tasks=[crosses]  train.selection_metric=crosses_f1  augment.runtime=true
+COMMON  train.active_tasks=[crosses]  train.selection_metric=crosses_auc  augment.runtime=true
         train.lr_schedule=warmup_cosine
 BS32    train.batch_size=32  train.accum_steps=1
 STREAM  paths.lmdb_train=[preprocessed_train_shuffled]  pose.input_stats=<pose58_train_shuffled.json>
@@ -95,7 +105,9 @@ of optimizer steps is identical; only the BatchNorm batch changes.
 | R3 pure | `onset_head`, `onset_report_crosses`, hazard 1.0, crosses weight 0 | + `train.loss_weight` |
 | R4 hedge | R3 + `onset_readout_weight=0.5` | as R3 |
 | R3C censored | R3 on `preprocessed_train_censored_shuffled` | as R3 + `paths.lmdb_train` |
-| Model A (3-task), seed 42 only | `active_tasks=[actions,looks,crosses]`, `selection_metric=macro_f1`, both protocols | seed, protocol, `pose.input_*`, the two task keys |
+| Model A (3-task), seed 42 only | `active_tasks=[actions,looks,crosses]`, both protocols | seed, protocol, `pose.input_*`, `train.active_tasks` |
+
+`train.selection_metric` is an allowed difference for every arm: the reference (`pf_fix_s42`) was F1-selected.
 
 The reference is `pf_fix_s42`'s `resolved_config.yaml`. Every arm is diffed against it **before it trains**
 (`scripts/check_run_config.py`). Any difference outside its column fails that arm's `_cfg` job, so it never
@@ -122,25 +134,19 @@ recipe (sampler off) was caught, and the censored folder under a binary arm is r
 
 ## 5. Run inventory
 
-**Reused from the ladder** if the config check passes (marker `queue_v4/reuse_<tag>.done`). If it fails
-(`.no`), the campaign trains a `c4_` twin instead:
-
-| ladder run | becomes |
-|---|---|
-| `pf_fix_s42`, `pf_fix_s43`, `pf_fix_s44` | R2 streaming s42/43/44 |
-| `pf_fixonset_s42` | R3 s42 |
-
-**New runs** (tags `c4_*`):
+**No ladder run is reused** (they were F1-selected; only `best`/`last` checkpoints exist, so their selection
+cannot be redone). All runs are new (tags `c4_*`):
 
 | arm | seeds | runs | feeds |
 |---|---|---|---|
-| R3 pure | 43, 44 | 2 | `tab:seedspread`, `tab:detection`, `tab:onset` (**the headline**) |
+| R2 streaming (binary baseline) | 42, 43, 44 | 3 | `tab:seedspread`, `tab:matrix`, `tab:detection` (**the headline**) |
+| R3 pure | 42, 43, 44 | 3 | `tab:seedspread`, `tab:detection`, `tab:onset` (**the headline**) |
 | R2 anchored | 42, 43, 44 | 3 | `tab:matrix`, the gap decomposition, every onset arm's anchored-trained row |
 | Model A, 3-task | 42 | 2 | `tab:matrix` rows A |
 | R1 auxiliary | 42, 43, 44 | 3 | `tab:onset`, `tab:detection` |
 | R4 hedge | 42, 43, 44 | 3 | same |
 | R3C censored | 42, 43, 44 | 3 | same + the lead-time trade |
-| **total** | | **16** | |
+| **total** | | **20** | |
 
 **Per run:** config check → train → val + test on **both** protocols → streaming-test dump → report refresh.
 
@@ -154,8 +160,8 @@ of which come from labels. The GBM probe stays as an external reference row.
 
 ## 6. Run order
 
-1. **W1, headline:** R3 s43, s44. With the reused runs this completes R2s ×3 vs R3 ×3, and the
-   pre-registered criterion is computed straight away into `v4_report/headline.md`.
+1. **W1, headline:** R2s and R3 as a matched pair per seed (42, 43, 44). The pre-registered criterion is
+   computed into `v4_report/headline.md` as soon as both arms have 3 seeds.
 2. **W2, motivation:** `anchored_stats`, R2 anchored ×3, Model A streaming + anchored.
 3. **W3, every arm at one seed:** `censored_shuffle`, R1 s42, R4 s42, R3C s42.
 4. **W4, remaining seeds:** R1, R4, R3C at 43 then 44.
@@ -202,7 +208,7 @@ Unchanged from `SEED_PLAN_2026-09-21.md`:
 ## 9. Decisions (settled 2026-09-26)
 
 D1 Model A kept at seed 42 only · D2 R3C at 3 seeds · D3 anchored arms scaled by their own training set ·
-D4 `selection_metric=crosses_f1` kept (so ladder runs are reusable) · D5 warm-up fix applied to the running
+D4 ~~`selection_metric=crosses_f1` kept~~ → **`crosses_auc`** (2026-09-27; reuse dropped) · D5 warm-up fix applied to the running
 ladder.
 
 ## 10. Out of scope
diff --git a/src/pedpredict/training/trainer.py b/src/pedpredict/training/trainer.py
index 8cc4684..52610cf 100644
--- a/src/pedpredict/training/trainer.py
+++ b/src/pedpredict/training/trainer.py
@@ -398,7 +398,7 @@ class Trainer:
     # ----------------------------------------------------------------- validation
 
     def _selection_value(self, val_loss: float, metrics: MetricResult) -> float:
-        """The minimized scalar for best-ckpt + early stop (M8). F1 metrics are negated (maximized).
+        """The minimized scalar for best-ckpt + early stop (M8). F1 / AUC metrics are negated (maximized).
 
         ``macro_f1`` averages ONLY active tasks; with a single active task there is no ``macro_f1``
         column, so it resolves to that task's own macro (``metrics.macro_f1``, which ``compute`` set to
```

## Selected diff 8c8be99

```text
commit 8c8be99e29ebbe1052c4e15b661202c1e587eb99
Author: Blegarim <nguyenbaoviet25072003@gmail.com>
Date:   Sun Sep 27 11:06:01 2026 +0900

    Plan: campaign launched by manual override (gate NO-GO), memorization ladder held
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

diff --git a/docs/RERUN_PLAN_2026-09-26.md b/docs/RERUN_PLAN_2026-09-26.md
index 0b4e299..5de978f 100644
--- a/docs/RERUN_PLAN_2026-09-26.md
+++ b/docs/RERUN_PLAN_2026-09-26.md
@@ -5,6 +5,11 @@ PC waits for the pixel-free ladder (`queue_v3.sh`, tmux `queue3`) to finish, run
 once, and on GO starts `queue_v4.sh` in tmux `queue4`. Nobody has to be watching. On NO-GO nothing
 launches, and the verdict says which check failed (§7).
 
+> **2026-09-27 02:05 UTC: LAUNCHED BY MANUAL OVERRIDE** (user decision, gate NO-GO). Recipe = hub (`pf_fix` +
+> `crosses_auc` selection), no memorization fix. Memorization ladder held (`queue_mem/HOLD`): A s42 done, B s42
+> stopped mid-run, C not run; if resumed it only records verdicts. Record: `queue_v4/OVERRIDE.md`. The paper must
+> state the model still overfits after epochs 1–2 (best checkpoint = the val-AUC peak).
+>
 > **2026-09-27 update.** (1) Every campaign arm now selects `best.pth` (and early-stops) on **val
 > `crosses_auc`**, not F1 at the fixed 0.5 cut. Keeping F1 (old D4) was a mistake: its revisit condition
 > fired on `pf_fix_s42` (F1 picked epoch 7, the AUC peak was epoch 17) and was not acted on. As a result no
```

## Selected diff a9f67ff

```text
commit a9f67ff7a70749f33fafc8ba4465befb4c913222
Author: Blegarim <nguyenbaoviet25072003@gmail.com>
Date:   Tue Sep 29 10:23:36 2026 +0900

    Pre-register the anchored-who x streaming-when combination test (before any number)
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

diff --git a/outputs/diagnostics/combo_test/PREREG.md b/outputs/diagnostics/combo_test/PREREG.md
new file mode 100644
index 0000000..c7a54d0
--- /dev/null
+++ b/outputs/diagnostics/combo_test/PREREG.md
@@ -0,0 +1,26 @@
+# Pre-registration: anchored "who" × streaming "when" (written 2026-09-29, before any combined number)
+
+**Question.** The anchored-trained model ranks eventual crossers well (per pedestrian AUC 0.799) but its
+timing is inverted (0.414); streaming-trained models time well (0.753–0.766). Does combining the two, with no
+retraining, detect crossings better than the streaming model alone?
+
+**Inputs.** Existing v4 campaign checkpoints only (no training). Test scores: the streaming-test dumps already
+logged. Validation scores: new streaming-val dumps of the same checkpoints (inference only), used only by the
+fitted variant. Seeds are paired: anchored `c4_r2a_sN` with streaming `c4_r2s_sN` (and `c4_r3_sN`).
+
+**Primary (fixed now, no fitted parameters):** per window, `p_combo = p_anc × p_str`, where `p_anc` is the
+anchored model's `p_frame` and `p_str` the streaming binary baseline's `p_frame` (R2).
+
+**Secondary (reported, not the claim):**
+- S1 — causal intent: `mean(p_anc over this pedestrian's windows so far) × p_str` (deployable, no look-ahead).
+- S2 — fitted: logistic regression on streaming **val** with features `logit p_anc`, `logit p_str`, and their
+  product; fitted per seed, applied unchanged to test.
+- S3 — the primary product with R3's hazard read-out as `p_str` instead of R2.
+
+**Metric and criterion (same as the campaign's).** Detection rate at {1200, 460, 205, 95, 41} alarms/hr,
+`per_window` accounting, streaming test, mean ± sd over the 3 paired seeds. **The combination wins if its mean
+beats R2 alone by more than the sum of the two sds at ≥ 3 of the 5 budgets.** Anything less is inconclusive.
+`per_track`, window AUC and lead time are reported alongside.
+
+**Not allowed after this point:** changing the combination rule, the pairing, the metric, the budgets or the
+criterion; picking among S1–S3 by their test results.
```

## Selected diff fd507a2

```text
commit fd507a2777c175450d574c45bcd5bc2ee9969e1f
Author: Blegarim <nguyenbaoviet25072003@gmail.com>
Date:   Tue Sep 29 10:29:48 2026 +0900

    Combination test (pre-registered): anchored who x streaming when is inconclusive
    
    Primary p_anc x p_R2 vs R2 alone, per_window, 3 paired seeds: INCONCLUSIVE
    (42.3 vs 42.3 at 205/hr, 23.1 vs 22.9 at 41/hr; window AUC 0.757 vs 0.783).
    All secondaries (causal intent, val-fitted stack, x R3) also inconclusive.
    
    - scripts/report_combination.py: the test, exactly as PREREG.md (a9f67ff);
      streaming-val dumps of the 9 checkpoints (inference only) feed only the
      val-fitted variant.
    - eval/campaign_report.py: assert_aligned, StackedCombiner, and
      compare_curves (the pre-registered rule, now shared with
      report_campaign.py; criterion.md unchanged).
    - RESULTS_MATRIX.md: result logged under the v4 analyses.
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

diff --git a/outputs/diagnostics/combo_test/results.md b/outputs/diagnostics/combo_test/results.md
new file mode 100644
index 0000000..789241f
--- /dev/null
+++ b/outputs/diagnostics/combo_test/results.md
@@ -0,0 +1,28 @@
+# Anchored who × streaming when — results (pre-registered: PREREG.md)
+
+Detection rate % (`per_window`, streaming test, mean ± sd over 3 paired seeds) and window AUC.
+
+| score | AUC | @1200/hr | @460/hr | @205/hr | @95/hr | @41/hr |
+|---|---|---|---|---|---|---|
+| R2 alone (baseline) | 0.783 ± 0.010 | 69.6 ± 6.0 | 55.0 ± 6.2 | 42.3 ± 3.2 | 32.0 ± 4.2 | 22.9 ± 3.5 |
+| anchored alone | 0.516 ± 0.010 | 35.9 ± 6.2 | 24.1 ± 2.9 | 15.1 ± 2.6 | 8.9 ± 2.9 | 4.1 ± 2.3 |
+| R3 alone | 0.793 ± 0.010 | 72.0 ± 2.8 | 59.2 ± 4.9 | 44.1 ± 4.7 | 31.4 ± 5.9 | 17.4 ± 9.1 |
+| PRIMARY: anchored × R2 | 0.757 ± 0.021 | 69.8 ± 5.2 | 54.3 ± 4.3 | 42.3 ± 2.0 | 31.2 ± 1.8 | 23.1 ± 4.3 |
+| S1: causal intent (running mean of anchored) × R2 | 0.760 ± 0.020 | 63.9 ± 9.1 | 45.5 ± 5.7 | 33.3 ± 5.2 | 22.9 ± 3.5 | 15.9 ± 4.5 |
+| S2: stacked on val (anchored, R2) | 0.781 ± 0.010 | 69.3 ± 6.3 | 54.6 ± 6.4 | 41.6 ± 4.1 | 32.2 ± 3.9 | 22.8 ± 4.5 |
+| S3: anchored × R3 | 0.750 ± 0.013 | 70.4 ± 0.7 | 55.8 ± 2.4 | 45.0 ± 1.1 | 33.7 ± 3.8 | 23.9 ± 2.1 |
+
+## Pre-registered criterion vs R2 alone (beats R2 beyond the sd sum at >= 3 of 5 budgets)
+
+- anchored alone, `per_window`: **R2 alone (baseline) WINS**
+- anchored alone, `per_track`: **INCONCLUSIVE**
+- R3 alone, `per_window`: **INCONCLUSIVE**
+- R3 alone, `per_track`: **INCONCLUSIVE**
+- PRIMARY: anchored × R2, `per_window` **(PRIMARY)**: **INCONCLUSIVE**
+- PRIMARY: anchored × R2, `per_track`: **INCONCLUSIVE**
+- S1: causal intent (running mean of anchored) × R2, `per_window`: **INCONCLUSIVE**
+- S1: causal intent (running mean of anchored) × R2, `per_track`: **INCONCLUSIVE**
+- S2: stacked on val (anchored, R2), `per_window`: **INCONCLUSIVE**
+- S2: stacked on val (anchored, R2), `per_track`: **INCONCLUSIVE**
+- S3: anchored × R3, `per_window`: **INCONCLUSIVE**
+- S3: anchored × R3, `per_track`: **INCONCLUSIVE**
```

## Uncommitted tracked diff

```text
diff --git a/outputs/runs/index.csv b/outputs/runs/index.csv
index 7777721..827fb6a 100644
--- a/outputs/runs/index.csv
+++ b/outputs/runs/index.csv
@@ -49,3 +49,33 @@ test_phase_schedule_smoke_thre0,test_phase,full,schedule,schedule,3,1,1.6572,0.6
 test_reload_best_between_phase0,test_reload,full,schedule,schedule,2,1,1.6611,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0\phase_1_p2\checkpoints\best.pth,691e06a
 test_fit_smoke_one_tiny_chunk0,test_fit,full,,train,1,1,1.7223,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0\checkpoints\best.pth,691e06a
 test_crosses_only_fit_writes_c0,test_crosses,full,,train,1,1,1.5028,0.6667,0.0,nan,nan,nan,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_crosses_only_fit_writes_c0,,691e06a
+test_phase_schedule_smoke_thre0,test_phase,full,schedule,schedule,3,1,1.6698,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_phase_schedule_smoke_thre0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_phase_schedule_smoke_thre0\phase_2_p3\checkpoints\best.pth,de35f65
+test_reload_best_between_phase0,test_reload,full,schedule,schedule,2,1,1.6368,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0\phase_1_p2\checkpoints\best.pth,de35f65
+test_fit_smoke_one_tiny_chunk0,test_fit,full,,train,1,1,1.7223,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0\checkpoints\best.pth,de35f65
+test_crosses_only_fit_writes_c0,test_crosses,full,,train,1,1,1.5028,0.6667,0.0,nan,nan,nan,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_crosses_only_fit_writes_c0,,de35f65
+test_phase_schedule_smoke_thre0,test_phase,full,schedule,schedule,3,1,1.6698,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_phase_schedule_smoke_thre0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_phase_schedule_smoke_thre0\phase_2_p3\checkpoints\best.pth,de35f65
+test_reload_best_between_phase0,test_reload,full,schedule,schedule,2,1,1.6368,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0\phase_1_p2\checkpoints\best.pth,de35f65
+test_fit_smoke_one_tiny_chunk0,test_fit,full,,train,1,1,1.7223,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0\checkpoints\best.pth,de35f65
+test_crosses_only_fit_writes_c0,test_crosses,full,,train,1,1,1.5028,0.6667,0.0,nan,nan,nan,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_crosses_only_fit_writes_c0,,de35f65
+test_fit_smoke_one_tiny_chunk0,test_fit,full,,train,1,1,1.7223,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-2\test_fit_smoke_one_tiny_chunk0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-2\test_fit_smoke_one_tiny_chunk0\checkpoints\best.pth,de35f65
+test_crosses_only_fit_writes_c0,test_crosses,full,,train,1,1,1.5028,0.6667,0.0,nan,nan,nan,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-2\test_crosses_only_fit_writes_c0,,de35f65
+test_fit_smoke_one_tiny_chunk0,test_fit,full,,train,1,1,1.7223,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-4\test_fit_smoke_one_tiny_chunk0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-4\test_fit_smoke_one_tiny_chunk0\checkpoints\best.pth,de35f65
+test_crosses_only_fit_writes_c0,test_crosses,full,,train,1,1,1.5028,0.6667,0.0,nan,nan,nan,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-4\test_crosses_only_fit_writes_c0,,de35f65
+test_phase_schedule_smoke_thre0,test_phase,full,schedule,schedule,3,1,1.6698,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_phase_schedule_smoke_thre0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_phase_schedule_smoke_thre0\phase_2_p3\checkpoints\best.pth,de35f65
+test_reload_best_between_phase0,test_reload,full,schedule,schedule,2,1,1.6368,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0\phase_1_p2\checkpoints\best.pth,de35f65
+test_fit_smoke_one_tiny_chunk0,test_fit,full,,train,1,1,1.7223,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0\checkpoints\best.pth,de35f65
+test_crosses_only_fit_writes_c0,test_crosses,full,,train,1,1,1.5028,0.6667,0.0,nan,nan,nan,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_crosses_only_fit_writes_c0,,de35f65
+test_fit_smoke_one_tiny_chunk0,test_fit,full,,train,1,1,1.7223,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-2\test_fit_smoke_one_tiny_chunk0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-2\test_fit_smoke_one_tiny_chunk0\checkpoints\best.pth,5aca9d7
+test_crosses_only_fit_writes_c0,test_crosses,full,,train,1,1,1.5028,0.6667,0.0,nan,nan,nan,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-2\test_crosses_only_fit_writes_c0,,5aca9d7
+test_phase_schedule_smoke_thre0,test_phase,full,schedule,schedule,3,1,1.6698,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-3\test_phase_schedule_smoke_thre0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-3\test_phase_schedule_smoke_thre0\phase_2_p3\checkpoints\best.pth,5aca9d7
+test_reload_best_between_phase0,test_reload,full,schedule,schedule,2,1,1.6368,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-3\test_reload_best_between_phase0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-3\test_reload_best_between_phase0\phase_1_p2\checkpoints\best.pth,5aca9d7
+test_fit_smoke_one_tiny_chunk0,test_fit,full,,train,1,1,1.7223,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-3\test_fit_smoke_one_tiny_chunk0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-3\test_fit_smoke_one_tiny_chunk0\checkpoints\best.pth,5aca9d7
+test_crosses_only_fit_writes_c0,test_crosses,full,,train,1,1,1.5028,0.6667,0.0,nan,nan,nan,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-3\test_crosses_only_fit_writes_c0,,5aca9d7
+test_phase_schedule_smoke_thre0,test_phase,full,schedule,schedule,3,1,1.6698,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_phase_schedule_smoke_thre0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_phase_schedule_smoke_thre0\phase_2_p3\checkpoints\best.pth,8513b28
+test_reload_best_between_phase0,test_reload,full,schedule,schedule,2,1,1.6368,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0\phase_1_p2\checkpoints\best.pth,8513b28
+test_fit_smoke_one_tiny_chunk0,test_fit,full,,train,1,1,1.7223,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0\checkpoints\best.pth,8513b28
+test_crosses_only_fit_writes_c0,test_crosses,full,,train,1,1,1.5028,0.6667,0.0,nan,nan,nan,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_crosses_only_fit_writes_c0,,8513b28
+test_phase_schedule_smoke_thre0,test_phase,full,schedule,schedule,3,1,1.6698,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_phase_schedule_smoke_thre0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_phase_schedule_smoke_thre0\phase_2_p3\checkpoints\best.pth,8c8be99
+test_reload_best_between_phase0,test_reload,full,schedule,schedule,2,1,1.6368,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_reload_best_between_phase0\phase_1_p2\checkpoints\best.pth,8c8be99
+test_fit_smoke_one_tiny_chunk0,test_fit,full,,train,1,1,1.7223,0.6667,1.0,0.2222,0.0,0.0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_fit_smoke_one_tiny_chunk0\checkpoints\best.pth,8c8be99
+test_crosses_only_fit_writes_c0,test_crosses,full,,train,1,1,1.5028,0.6667,0.0,nan,nan,nan,C:\Users\LENOVO\AppData\Local\Temp\pytest-of-LENOVO\pytest-0\test_crosses_only_fit_writes_c0,,8c8be99
diff --git a/paper/main.tex b/paper/main.tex
index 217738d..2c05161 100644
--- a/paper/main.tex
+++ b/paper/main.tex
@@ -68,55 +68,62 @@
 \newcommand{\Nnoncrossers}{495}
 \newcommand{\decisionsPerHour}{36\,000}
 
-% Window-level metrics, streaming test, val-tuned thresholds (Sec. V rule).
-% Read from outputs/runs/<run>/eval_log.csv, 2026-09-23.
-\newcommand{\baseAUCa}{0.777}   % R2s seed 42, 20260917_082845
-\newcommand{\baseFa}{0.200}     % R2s seed 42
-\newcommand{\baseAUCb}{0.748}   % R2s seed 43, 20260922_003330
-\newcommand{\baseFb}{0.259}     % R2s seed 43
-\newcommand{\baseAUCc}{0.746}   % R2s seed 44, 20260923_180101 (28 epochs, best ep 13)
-\newcommand{\baseFc}{0.181}     % R2s seed 44
-% Binary baseline three-seed summary, streaming test, val-tuned thresholds.
-\newcommand{\baseAUCmean}{0.757}
-\newcommand{\baseAUCsd}{0.017}
-\newcommand{\baseFmean}{0.213}
-\newcommand{\baseFsd}{0.041}
-\newcommand{\pureAUCa}{0.761}   % R3 seed 42, 20260918_105518
-\newcommand{\pureFa}{0.220}     % R3 seed 42
-\newcommand{\pureAUCb}{0.767}   % R3 seed 43, 20260921_100118
-\newcommand{\pureFb}{0.163}     % R3 seed 43
-\newcommand{\pureAUCc}{0.754}   % R3 seed 44, 20260923_014320 (best ep 5, spread 0.205)
-\newcommand{\pureFc}{0.192}     % R3 seed 44
-% R3 three-seed summary, streaming test, val-tuned thresholds.
-\newcommand{\pureAUCmean}{0.761}
-\newcommand{\pureAUCsd}{0.006}
-\newcommand{\pureFmean}{0.192}
-\newcommand{\pureFsd}{0.028}
-\newcommand{\auxAUC}{0.772}     % R1 seed 42, 20260916_075501
-\newcommand{\auxF}{0.225}       % R1 seed 42
-\newcommand{\hedgeAUC}{0.776}   % R4 seed 42, 20260919_111158
-\newcommand{\hedgeF}{0.185}     % R4 seed 42
-\newcommand{\censAUC}{0.755}    % R3C seed 42, 20260920_021440
-\newcommand{\censF}{0.204}      % R3C seed 42
-
-% Seed spread on the same arm, two seeds. This is the load-bearing number for
-% the argument that window metrics cannot separate the configurations.
-\newcommand{\baseFspread}{0.059}  % 0.259 - 0.200, R2s seeds 42/43
-\newcommand{\pureFspread}{0.057}  % 0.220 - 0.163, R3 seeds 42/43
-
-% Post-processing effect, R1 dump, k=15 within-track moving average.
-\newcommand{\smoothAUCgain}{0.024}
-\newcommand{\armAUCgap}{0.023}
-
-% Detection-curve values live in the tables and prose of Sec. VII directly, marked
-% as a block by \provtable rather than one macro per cell.
-% Provenance: scripts/report_detection_curve.py over
-%   outputs/diagnostics/r2s_s42/onset_test.npz#p_frame    (binary baseline)
-%   outputs/diagnostics/r3/onset_test.npz#p_readout       (pure hazard)
-%   outputs/diagnostics/r4/onset_test.npz#p_readout       (hedged)
-%   outputs/diagnostics/r3c/onset_test.npz#p_readout      (pure + censored)
-% streaming test split, seed 42, computed 2026-09-24. Head choice matters: an
-% onset arm's p_frame is degenerate (constant 0.5, never supervised).
+% Every value below comes from scripts/report_campaign.py over the 20 runs of the
+% pixel-free campaign (outputs/runs/*_c4_*), written to
+% outputs/diagnostics/v4_report/tables/ (results.json holds every per-seed value).
+% Window-level metrics: streaming test, val-tuned thresholds (Sec. V rule), from
+% each run's eval_log.csv.
+\newcommand{\baseAUCa}{0.773}   % R2 seed 42, c4_r2s_s42
+\newcommand{\baseFa}{0.234}     % R2 seed 42
+\newcommand{\baseAUCb}{0.792}   % R2 seed 43, c4_r2s_s43
+\newcommand{\baseFb}{0.244}     % R2 seed 43
+\newcommand{\baseAUCc}{0.783}   % R2 seed 44, c4_r2s_s44
+\newcommand{\baseFc}{0.250}     % R2 seed 44
+% Binary baseline three-seed summary.
+\newcommand{\baseAUCmean}{0.783}
+\newcommand{\baseAUCsd}{0.010}
+\newcommand{\baseFmean}{0.243}
+\newcommand{\baseFsd}{0.008}
+\newcommand{\pureAUCa}{0.802}   % R3 seed 42, c4_r3_s42
+\newcommand{\pureFa}{0.253}     % R3 seed 42
+\newcommand{\pureAUCb}{0.782}   % R3 seed 43, c4_r3_s43
+\newcommand{\pureFb}{0.273}     % R3 seed 43
+\newcommand{\pureAUCc}{0.794}   % R3 seed 44, c4_r3_s44
+\newcommand{\pureFc}{0.303}     % R3 seed 44
+% Pure hazard three-seed summary.
+\newcommand{\pureAUCmean}{0.793}
+\newcommand{\pureAUCsd}{0.010}
+\newcommand{\pureFmean}{0.277}
+\newcommand{\pureFsd}{0.025}
+% Other configurations, three-seed mean and sample sd.
+\newcommand{\auxAUC}{0.797}\newcommand{\auxAUCsd}{0.010}     % R1, c4_r1_s{42,43,44}
+\newcommand{\auxF}{0.255}\newcommand{\auxFsd}{0.032}
+\newcommand{\hedgeAUC}{0.789}\newcommand{\hedgeAUCsd}{0.008} % R4, c4_r4_s{42,43,44}
+\newcommand{\hedgeF}{0.245}\newcommand{\hedgeFsd}{0.059}
+\newcommand{\censAUC}{0.795}\newcommand{\censAUCsd}{0.009}   % R3C, c4_r3c_s{42,43,44}
+\newcommand{\censF}{0.271}\newcommand{\censFsd}{0.028}
+
+% Gaps between the binary and pure hazard three-seed means (pure hazard minus
+% binary). The F1 gap 0.0340 just exceeds the 0.0331 sum of the two sds; the AUC
+% gap 0.0103 stays inside its 0.0194.
+\newcommand{\FmeanGap}{0.034}
+\newcommand{\AUCmeanGap}{0.010}
+
+% Model size (Sec. V-A): build_model(resolved_config.yaml) gives 655,421 parameters
+% for the binary configuration and 658,517 with the hazard head.
+% Who/when split, anchored model, combination test (Sec. V-E): analyses.md and
+% outputs/diagnostics/combo_test/results.md (pre-registered in PREREG.md there).
+
+% Post-processing effect on the binary baseline, k=15 within-track moving average
+% (analyses.md): centered +0.011, causal (trailing) -0.004.
+\newcommand{\smoothAUCgain}{0.011}
+\newcommand{\smoothAUCcausal}{0.004}
+
+% Detection-curve values live in the tables and prose of Sec. VII directly.
+% Provenance: detection_per_window.md / detection_per_track.md / criterion.md in
+% outputs/diagnostics/v4_report/tables/, streaming test split, 3 seeds per
+% configuration. Head choice: p_readout for pure, hedged and censored; p_frame
+% for binary and auxiliary.
 
 \begin{document}
 
@@ -129,43 +136,24 @@ Protocol, Diagnosis, and Onset Timing Under Censoring}
 \maketitle
 
 \begin{abstract}
-Pedestrian crossing-intention models are usually trained and tested under an
-event-anchored protocol: each pedestrian track contributes one short observation
-window, placed a known one to two seconds before an annotated crossing event. A
-detector on a vehicle works differently. It watches continuously, and at every
-moment it must judge whether a crossing is about to begin. This paper defines
-both protocols precisely on the PIE dataset and measures what the difference
-between them costs. Anchored sampling gives a class ratio of about $2.5{:}1$
-because it takes roughly one window per track. Dense sliding-window sampling
-gives $33.9{:}1$, because that is how often a crossing is genuinely imminent
-when a camera watches without pause. The protocols also differ in what their
-negatives are made of. We find that $25.7\%$ of streaming training windows and
-$39.2\%$ of test windows show a pedestrian who does cross, only later than the
-horizon asks about, and the anchored protocol produces no such windows for
-either training or testing. A model trained under the anchored protocol falls
-from $0.873$ to $0.540$ AUC when tested on the stream, and re-tuning its
-decision threshold at the streaming base rate recovers almost none of that
-loss. What the model has lost is therefore its ability to rank pedestrians by
-risk, not the placement of its threshold. We argue that part of the difficulty
-comes from the label itself, which gives opposite answers to almost identical
-observations on either side of a fixed cutoff, and we propose treating the task
-as predicting \emph{when} a crossing will start, under right-censoring. A
-window whose future footage runs out is then a censored observation rather than
-a negative, and the model still reports the probability of a crossing within the
-original horizon, so it remains directly comparable with binary baselines. We
-also find that the window-level scores the field reports cannot rank these
-objectives: across three seeds each, the two differ on F1 by less than either
-one's own seed-to-seed spread, and a moving average over each track's scores
-moves AUC by more than the objectives do. We therefore evaluate on
-detection rate and lead time at matched false-alarm budgets. On a single seed
-the hazard formulation detects $3.1\times$ more crossing pedestrians at $205$
-alarms per hour. Across three seeds of each objective that separation
-disappears: the means differ by between $-2.3$ and $+7.6$ percentage points
-against standard deviations of $8$ to $22$, and a criterion fixed before the
-runs is met at none of the five budgets. We report the objective comparison as
-inconclusive. What the measurement does establish concerns the evaluation: at
-this base rate a single-run comparison, which is what the standard protocol
-invites, can report a threefold improvement that replication does not support.
+Standard benchmarks for pedestrian crossing prediction use an event-anchored
+protocol that places a few observation windows per pedestrian one to two seconds
+before an annotated crossing event. A detector on a vehicle instead watches
+continuously and must decide at every moment whether a crossing is about to
+begin. On the Pedestrian Intention Estimation (PIE) dataset, the anchored
+protocol yields about $2.5$ negative windows per positive and the streaming
+protocol $33.9$. In the streaming test set, $39\%$ of windows are negatives that
+show a pedestrian who does cross, only later than the one-second prediction
+horizon. The anchored protocol never produces such negatives. A model trained
+under the anchored protocol falls from $0.89$ to $0.52$ area under the ROC
+curve (AUC) on streaming data, and re-tuning its decision threshold recovers
+none of the loss. The model still ranks which pedestrians will cross ($0.80$
+AUC), but orders imminent crossings below distant ones. We then recast the task
+as predicting when a crossing will begin: windows whose footage ends early
+become censored observations instead of negatives, and the model still reports
+the probability of a crossing within the original horizon. With three training
+seeds per method, detection rate at fixed false-alarm rates does not separate
+this formulation from binary training.
 \end{abstract}
 
 \begin{IEEEkeywords}
@@ -183,19 +171,22 @@ about to step into it. Nothing tells it when to attend to a particular
 pedestrian, and most of the time nothing is about to happen.
 
 The protocol used to evaluate such systems asks a narrower question. Following
-the procedure introduced with PIE and JAAD~\cite{rasouli2019pie,rasouli2017jaad}
-and settled in the standard evaluation suite~\cite{kotseruba2021benchmark}, each
-pedestrian track contributes about one labeled observation window. Observation
-stops at the crossing event, and the window is placed so that its last frame
-falls a known one to two seconds before that event. The class ratio that results
-is mild, roughly $2.5{:}1$, and reported scores are high.
+the procedure introduced with the Pedestrian Intention Estimation (PIE) and
+Joint Attention in Autonomous Driving (JAAD)
+datasets~\cite{rasouli2019pie,rasouli2017jaad} and fixed in the standard
+evaluation suite~\cite{kotseruba2021benchmark}, each pedestrian track
+contributes a few labeled observation windows, all from one short interval.
+Observation stops at the crossing event, and the protocol places the windows so
+that their last frames fall one to two seconds before that event. The resulting class
+ratio is mild, about $2.5$ negatives per positive ($2.5{:}1$), and reported
+scores are high.
 
 The two settings differ in more than difficulty. Because the anchored protocol
-places its window relative to an event it already knows about, it never produces
-the windows that occur earlier in a crossing pedestrian's track. Those are the
-moments when a person who will eventually cross is still walking along the
-sidewalk. Under a short horizon such windows are negatives, they look almost
-exactly like the positives that follow them, and they are the situations in
+places its windows relative to an event it already knows about, it never
+produces the windows that occur earlier in a crossing pedestrian's track. Those
+are the moments when a person who will eventually cross is still walking along
+the sidewalk. Under a short horizon such windows are negatives, they look almost
+identical to the positives that follow them, and they are the situations in
 which a deployed detector must not raise a false alarm. Since the anchored
 protocol leaves them out of both training and testing, the standard pipeline
 neither teaches a model to handle them nor shows whether it can.
@@ -206,45 +197,47 @@ This paper measures that omission and what it costs.
   \centering
   \includegraphics[width=\textwidth]{fig1_protocols.pdf}
   \caption{The two sampling protocols applied to one pedestrian track. (a)~The
-  event-anchored protocol takes about one window per track, placed a known
+  event-anchored protocol takes a few windows per track, all ending a known
   interval before a known crossing onset; the rest of the track is never
-  sampled. (b)~Streaming sampling covers the whole track. Its negatives are
-  dominated by windows of the same pedestrian, in the same scene, shortly before
-  the onset. These are the hard temporal negatives that protocol (a) cannot
-  produce. Windows are drawn separated for legibility; in practice they overlap
+  sampled. (b)~Streaming sampling covers the whole track. Many of its negatives
+  are windows of the same pedestrian, in the same scene, before the onset: a
+  quarter of training windows and two fifths of test windows
+  (\cref{tab:composition}). These are the hard temporal negatives that
+  protocol (a) cannot produce. We draw the windows apart for legibility; in practice they overlap
   heavily (length $20$, stride $3$).}
   \label{fig:protocols}
 \end{figure*}
 
 We make five contributions.
 \begin{enumerate}
-  \item \textbf{The two protocols answer different questions.} We state both
-  precisely and show why their class ratios cannot be compared. One counts how
+  \item \textbf{The two protocols answer different questions.} We define both
+  and show that their class ratios measure different things. One counts how
   many pedestrians eventually cross; the other counts how often a crossing is
   about to begin (\cref{sec:protocols}).
   \item \textbf{The streaming negatives contain a class the benchmark cannot
-  produce.} We measure what they are made of, split by split. Two findings fall
-  out: the PIE training and test splits differ in exactly the respect this paper
+  produce.} We measure what they consist of, split by split. Two findings
+  follow: the PIE training and test splits differ in the respect this paper
   studies, and most of the seemingly difficult negatives are not confusable at
   all (\cref{sec:omits}).
   \item \textbf{The deployment gap is a ranking failure, not a threshold
   failure.} A cross-protocol matrix splits the anchored-to-streaming gap into the
   part threshold re-tuning recovers and the part it does not. We measure the
   first at approximately zero (\cref{sec:gap}).
-  \item \textbf{Onset timing under right-censoring repairs the label.} A masked
-  discrete-time hazard objective supervises only the bins whose outcome was
-  observed, and its horizon read-out stays numerically comparable with the
-  binary baselines, so the two remain measurable on one scale
-  (\cref{sec:method}).
+  \item \textbf{Onset timing under right-censoring repairs the label.} For each
+  future interval, the model predicts the probability that a crossing starts
+  there, given that none has started earlier (a discrete-time hazard). The loss
+  ignores intervals nobody observed, and the model still reports the probability
+  of a crossing within the original horizon, so it stays directly comparable
+  with binary baselines (\cref{sec:method}).
   \item \textbf{At this base rate the evaluation cannot resolve the objectives.}
-  Seed noise on window-level F1 (\baseFspread{} between two seeds of one
-  configuration) exceeds every gap between configurations, and a post-processing
-  moving average moves AUC by more than the configurations differ. We therefore
-  evaluate on detection rate and lead time at matched false-alarm budgets, the
-  accounting quickest change detection has used since the
-  1950s~\cite{page1954cusum}. Under a criterion fixed in advance and three seeds
-  per objective, that metric also fails to separate them, while a single seed
-  reports a threefold gap (\cref{sec:experiments}).
+  Across three seeds, the standard deviation of window-level F1 (\baseFsd{} and
+  \pureFsd) exceeds the \FmeanGap{} gap between the two objectives' means, and a
+  moving average over each track's scores moves AUC six times further than the
+  objectives differ. We therefore evaluate on detection rate and lead time at
+  matched false-alarm budgets, the accounting of quickest change
+  detection~\cite{page1954cusum}. Under a criterion fixed in advance, that
+  metric also fails to separate the objectives, although a single seed shows a
+  threefold gap (\cref{sec:experiments}).
 \end{enumerate}
 
 We do not claim streaming evaluation as a new idea; it is well established in
@@ -254,12 +247,6 @@ the crossing-intention literature has not adopted it, that this costs measurably
 and that a timing formulation repairs the underlying labeling problem instead of
 working around it.
 
-On a single seed that metric separates the objectives where window-level F1 does
-not, by $3.1\times$ at $205$ alarms per hour. Across two seeds of each it does
-not: both objectives vary more between their own seeds than they differ from
-each other. \cref{sec:detection} reports both facts, and we draw the
-methodological conclusion rather than the one about objectives.
-
 % ==========================================================================
 \section{Related Work}
 \label{sec:related}
@@ -267,16 +254,16 @@ methodological conclusion rather than the one about objectives.
 \subsection{Crossing Intention Estimation on PIE and JAAD}
 PIE~\cite{rasouli2019pie} and JAAD~\cite{rasouli2017jaad} are the standard
 datasets, and the protocol released with them~\cite{kotseruba2021benchmark} is
-the one most later work adopts: per-track binary labels, observation clipped at
-the crossing point, and a time-to-event interval of one to two seconds. Later
-work inherits it largely unchanged and measures improvements against it.
+the one most later work adopts, largely unchanged: per-track binary labels,
+observation clipped at the crossing point, and a time-to-event interval of one
+to two seconds.
 
 Several studies revisit that evaluation without changing which windows it
 produces. Azarmi \emph{et al.}~\cite{azarmi2024feature} split PIE into context
 sets and find a strong dependence on ego-vehicle speed; Rasouli and
-Kotseruba~\cite{rasouli2024diving} revise the task definitions and add classes
-of metric. Both work on the population the anchored protocol produces. This
-paper questions that population.
+Kotseruba~\cite{rasouli2024diving} revise the task definitions and add new
+types of metric. Both work on the population the anchored protocol produces.
+This paper questions that population.
 
 Yao \emph{et al.}~\cite{yao2021coupling} come closest. They report two sampling
 settings: an ``event-to-crossing'' setting that takes sequences from each track
@@ -284,9 +271,9 @@ in the one to two seconds before the crossing, and an ``original data'' setting
 that uses the whole track. They observe that the whole-track setting contains
 sequences long before the crossing event, where actions change little until the
 event approaches, and prefer the anchored setting for offering more action
-change. We differ in two ways. We label a window by whether an onset falls
-within a fixed horizon, not by whether the pedestrian eventually crosses, and
-those early windows are our main object of study.
+change. We differ in two ways. First, we label a window by whether an onset
+falls within a fixed horizon, not by whether the pedestrian eventually crosses.
+Second, the early windows they set aside are our main object of study.
 
 \subsection{Online Action Detection}
 Online action detection labels each frame of a stream from present and past
@@ -294,8 +281,8 @@ evidence alone~\cite{degeest2016oad}. Its onset-focused variant, action start
 detection~\cite{shou2018odas,gao2019startnet}, must signal a beginning as early
 as possible in untrimmed video where background frames dominate; recent work is
 mostly transformer-based~\cite{wang2021oadtr,xu2021lstr,zhao2022testra,wang2023mat}.
-Its evaluation machinery, including calibrated average precision and point-level
-average precision with a temporal tolerance, was built for this regime.
+The field built its evaluation machinery for this regime, including calibrated
+average precision and point-level average precision with a temporal tolerance.
 
 That literature also describes our failure mode. Shou \emph{et
 al.}~\cite{shou2018odas} observe that a start window shares scene and objects
@@ -307,7 +294,7 @@ motivation while evaluating on television, sports and ego-vehicle maneuver data.
 
 \subsection{Time-to-Event Modeling}
 Survival analysis models the time until an event when some observations end
-before the event is seen~\cite{kaplan1958nonparametric,cox1972regression}. The
+before the event~\cite{kaplan1958nonparametric,cox1972regression}. The
 discrete-time form we use in \cref{sec:method}, a per-interval hazard trained
 with a masked binary likelihood, is standard there and carries over to neural
 estimators~\cite{lee2018deephit}.
@@ -324,8 +311,9 @@ what makes them one.
 
 \textbf{Class imbalance.} Focal loss~\cite{lin2017focal} and class-balanced
 re-weighting~\cite{cui2019classbalanced} are the standard responses to a rare
-positive class. They move the decision boundary, while \cref{sec:gap} shows the
-ranking degrades under protocol shift, so we treat them as comparison baselines.
+positive class. They mainly move the decision boundary, whereas \cref{sec:gap}
+shows that protocol shift degrades the ranking itself. We therefore plan them as
+comparison baselines (\cref{sec:ablations}).
 
 % ==========================================================================
 \section{Problem Formulation}
@@ -354,14 +342,14 @@ The positive share now reflects how often a crossing is imminent per observed
 moment: on PIE, $2.9\%$ of windows, a ratio of $33.9{:}1$.
 
 \subsection{Why the Two Ratios Are Not Comparable}
-The anchored $2.5{:}1$ is a balance across tracks, since the protocol draws
-about one window from each; the streaming $33.9{:}1$ is a rate over time. They
-estimate different quantities, and moving between them is no difficulty dial.
-One objection is that streaming sampling is anchored sampling with a long
-time-to-event. A long-time-to-event anchored sample is still one window per
-track, placed a known interval from a known event. Streaming sampling uses no
-anchor, and its negatives are dominated by a category the anchored protocol
-never generates, which \cref{sec:omits} measures.
+The anchored $2.5{:}1$ is a balance across tracks, since the protocol draws the
+same few windows from each; the streaming $33.9{:}1$ is a rate over time. They
+estimate different quantities, so neither is a harder or easier version of the
+other. One objection is that streaming sampling is anchored sampling with a long
+time-to-event. A long-time-to-event anchored sample still draws a few windows
+per track, placed a known interval from a known event. Streaming sampling uses no
+anchor, and a large share of its negatives belongs to a category the anchored
+protocol never generates, as \cref{sec:omits} measures.
 
 \subsection{Four Kinds of Streaming Negative}
 \label{sec:taxonomy}
@@ -389,8 +377,9 @@ is a labeling error the binary formulation gives no way to avoid.
 We use PIE with the standard split assignment: sets 01, 02 and 06 for training,
 sets 04 and 05 for validation, and set 03 for testing. Streaming windows are
 $T = 20$ frames at stride $3$, with horizon $H = 32$ frames, about $1.07$\,s at
-$30$\,fps. We discard windows whose future is not fully observed instead of labeling them
-negative, since no crossing can be ruled out over an interval nobody watched.
+$30$\,fps. We discard windows whose future is not fully observed instead of
+labeling them negative, since an interval nobody watched cannot rule out a
+crossing.
 That leaves $\Ntrain$ training, $20\,490$ validation and $\Ntest$ test windows,
 with positive rates of $2.9$, $2.8$ and $3.1\%$. Every measurement in this
 section comes straight from the PIE annotation files and reproduces without
@@ -420,38 +409,37 @@ $8.9{:}1$ in training and $12.8{:}1$ in test.
   \end{tabular}
 \end{table}
 
-Two things in this table deserve attention. First, the splits are not built the
-same way. Test holds about half again as many hard temporal negatives as
-training, $39.2\%$ against $25.7\%$, and correspondingly fewer easy ones. A drop
+The table shows two things. First, the splits are not built the same way. Test
+holds about half again as many hard temporal negatives as training, $39.2\%$ against $25.7\%$, and correspondingly fewer easy ones. A drop
 from training to test on this data therefore cannot be read as a generalization
-gap alone, because the test split is harder in exactly the respect under study.
+gap alone, because the test split is harder in the respect under study.
 Validation is the easiest of the three at $15.2\%$, so thresholds chosen there
 are optimistic for test in a way a correctly run no-peeking protocol does not
 fix.
 
-Second, most of the seemingly difficult mass is not confusable. Grouping the
-hard temporal negatives by how far the crossing still is
-(\cref{fig:onsethist}), $68.7\%$ lie more than five seconds ahead. The genuinely
-confusable group is the windows whose onset falls in the two seconds just past
-the horizon, between $1.1$\,s and $3.2$\,s: $4{,}186$ training windows, $1.65$
-times the size of the positive set. That is the population a targeted method has
-to work on, and it is large enough to train against.
+Second, most of the seemingly difficult mass is not confusable. Grouped by time
+to onset (\cref{fig:onsethist}), $68.7\%$ of hard temporal negatives lie more
+than five seconds ahead. We call the windows whose onset falls in the $64$
+frames (about two seconds) just past the horizon, between $1.1$\,s and
+$3.2$\,s, the confusable band: $4{,}186$ training windows, $1.65$ times the size
+of the positive set. A targeted method has to work on this population, and it
+is large enough to train against.
 
 \begin{figure}[t]
   \centering
   \includegraphics[width=\columnwidth]{fig3_time_to_onset.pdf}
   \caption{Hard temporal negatives in the training split, grouped by how long
   remains until the crossing starts. Two thirds lie more than five seconds ahead
-  and are not confusable. The confusable band, the two shaded groups, is about
-  the size of the positive set.}
+  and are not confusable. The confusable band, the two shaded groups, is $1.65$
+  times the size of the positive set.}
   \label{fig:onsethist}
 \end{figure}
 
-\subsection{Why Not Simply Predict Further Ahead}
+\subsection{Why Not Predict Further Ahead}
 \label{sec:horizon}
 The obvious answer to a $33.9{:}1$ imbalance is to enlarge $H$. Relabeling the
-same windows at a range of horizons (\cref{tab:horizon}) shows this improves the
-nominal ratio considerably: the imbalance falls to $5.8{:}1$ at five seconds,
+same windows at a range of horizons (\cref{tab:horizon}) shows that this
+improves the nominal ratio: the imbalance falls to $5.8{:}1$ at five seconds,
 and the confusable band shrinks from $1.65$ to $0.26$ times the positive set. We
 keep $H = 32$ nonetheless, for three reasons.
 
@@ -476,16 +464,16 @@ keep $H = 32$ nonetheless, for three reasons.
 \textbf{A longer horizon deletes the windows that matter most.} It needs more
 observed future, so $26\%$ of windows fall away at five seconds and $44\%$ at
 ten. They fall away by track length, so short tracks go first, and short tracks
-are pedestrians who appear abruptly or are quickly occluded. The improved
-balance is bought by deleting the hardest part of the dataset.
+are pedestrians who appear abruptly or are quickly occluded. The better balance
+comes from deleting the hardest part of the dataset.
 
 \textbf{The confusion moves without disappearing.} The ratio improves because
-windows that were hard negatives at one second become positives at five. The
+windows that were hard temporal negatives at one second become positives at five. The
 model still cannot tell a window $4.9$\,s before onset from one $5.1$\,s before
 onset, and must now answer yes on the first, on evidence absent from the frame.
 False alarms become misses.
 
-\textbf{The horizon decides which phenomenon is detected.} At about one second
+\textbf{The horizon decides which phenomenon the model detects.} At about one second
 the visible cue is the body starting to move: weight shifting, a foot lifting,
 the torso turning. At five seconds no such cue exists, and what is visible is
 where the person is heading relative to the road. Two different problems need
@@ -500,25 +488,34 @@ far the anchored ratio reflects a sampling choice.
 \label{sec:gap}
 
 \subsection{Model and Training Setup}
-Every run uses one multimodal architecture: a hierarchical windowed-attention
-vision transformer over context crops, a pose and kinematics stream reducing
-$23$ whole-body keypoints~\cite{jin2020coco} to a $58$-dimensional per-frame
-vector, and a cross-attention module where the kinematic stream queries the
-visual features. Per-task heads share a model dimension of $128$, and the visual
-backbone is a frozen ImageNet-pretrained TinyViT-5M~\cite{wu2022tinyvit}. All
-runs share learning rate $10^{-4}$, weight decay $10^{-5}$, effective batch size
-$32$, a warmup--cosine schedule, and seed $42$.
+Every run uses one pose and kinematics model and reads no image pixels. Each
+frame becomes a $58$-dimensional vector: $49$ pose measurements derived from
+$23$ whole-body keypoints~\cite{jin2020coco} and $9$ channels of bounding-box position, size,
+their velocities and ego-vehicle speed. We standardize every channel with
+training-set statistics and clip it at $\pm 5$ standard deviations. Two
+temporal convolutions, a two-layer gated recurrent unit and multi-head
+self-attention encode the $20$-frame window at model dimension $128$, about
+$0.66$\,M parameters in all. A per-frame crossing head, pooled over frames by
+log-sum-exp, gives the binary output. All runs share learning rate $10^{-4}$,
+weight decay $10^{-5}$, batch size $32$ and a warmup-then-cosine schedule over
+at most $30$ epochs. A weighted sampler raises the share of crossing windows
+among training draws, and we store the training windows in a globally shuffled
+order so that every data chunk carries the global class mix. Training keeps
+the checkpoint with the highest validation AUC and stops after $15$ epochs
+without improvement. Every configuration runs with seeds $42$, $43$ and $44$.
 
 \subsection{The Cross-Protocol Matrix}
 We train each model under one protocol and evaluate it under both. Thresholds
 come from validation and apply unchanged to test; same-split sweeps serve
 diagnosis only. \cref{tab:matrix} gives the matrix for two models, one trained
-on all three annotation tasks and one on the crossing task alone.
+on the crossing task alone and one on all three annotation tasks.
 
 \begin{table}[t]
   \centering
   \caption{Cross-protocol results for the crossing task. Cells report
-  AUC $\cdot$ raw F1 $\rightarrow$ threshold-tuned F1.}
+  AUC $\cdot$ raw F1 $\rightarrow$ threshold-tuned F1. Model~B: mean over seeds
+  $42$, $43$ and $44$, sample standard deviation of AUC $\le 0.028$ in every
+  cell. Model~A: seed $42$ only.}
   \label{tab:matrix}
   \setlength{\tabcolsep}{3pt}
   \scriptsize
@@ -528,26 +525,28 @@ on all three annotation tasks and one on the crossing task alone.
     \cmidrule(lr){3-4}
     Model & train protocol & anchored & streaming \\
     \midrule
-    \multirow{2}{*}{A (3-task)}
-      & anchored  & 0.873 $\cdot$ 0.689 $\rightarrow$ 0.694 & \textbf{0.540 $\cdot$ 0.064 $\rightarrow$ 0.064} \\
-      & streaming & 0.514 $\cdot$ 0.186 $\rightarrow$ 0.322 & 0.742 $\cdot$ 0.211 $\rightarrow$ 0.190 \\
-    \midrule
     \multirow{2}{*}{B (crossing-only)}
-      & anchored  & 0.880 $\cdot$ 0.716 $\rightarrow$ 0.721 & \textbf{0.529 $\cdot$ 0.064 $\rightarrow$ 0.062} \\
-      & streaming & 0.679 $\cdot$ 0.258 $\rightarrow$ 0.428 & 0.784 $\cdot$ 0.249 $\rightarrow$ 0.225 \\
+      & anchored  & 0.889 $\cdot$ 0.730 $\rightarrow$ 0.729 & \textbf{0.516 $\cdot$ 0.063 $\rightarrow$ 0.062} \\
+      & streaming & 0.696 $\cdot$ 0.388 $\rightarrow$ 0.459 & 0.783 $\cdot$ 0.237 $\rightarrow$ 0.243 \\
+    \midrule
+    \multirow{2}{*}{A (3-task)}
+      & anchored  & 0.869 $\cdot$ 0.710 $\rightarrow$ 0.711 & \textbf{0.511 $\cdot$ 0.062 $\rightarrow$ 0.064} \\
+      & streaming & 0.711 $\cdot$ 0.397 $\rightarrow$ 0.510 & 0.785 $\cdot$ 0.225 $\rightarrow$ 0.265 \\
     \bottomrule
   \end{tabular}
 \end{table}
 
-Scores within a single protocol match published results, and the off-diagonal
-entries carry the finding. A model trained under the anchored protocol reaches
-$0.873$ AUC on the anchored test set and $0.540$ on the streaming one, close to
-chance. The crossing-only model collapses the same way, $0.880$ to $0.529$, so
-multi-task head interference does not explain it.
+The anchored-to-anchored scores fall in the range the standard benchmark
+reports~\cite{kotseruba2021benchmark}, and the off-diagonal entries carry the
+finding. A crossing-only model trained under the anchored protocol reaches
+$0.889 \pm 0.007$ AUC on the anchored test set over three seeds and
+$0.516 \pm 0.010$ on the streaming one, close to chance. The three-task model
+collapses the same way, $0.869$ to $0.511$, so multi-task head interference does
+not explain it.
 
 \subsection{Splitting the Deployment Gap}
 The gap splits into the part threshold re-tuning recovers and the part it does
-not. With every quantity in tuned F1 on the streaming test column:
+not. All three quantities are F1 scores on the streaming test set:
 \begin{align}
   \Gtot   &= \mathrm{F1}^{\text{tuned}}_{\text{str}\to\text{str}}
              - \mathrm{F1}^{\text{raw}}_{\text{anc}\to\text{str}}, \label{eq:gtot}\\
@@ -558,61 +557,107 @@ not. With every quantity in tuned F1 on the streaming test column:
 \end{align}
 so that $\Gtot = \Gprior + \Ghard$ by construction. $\Gprior$ is what
 re-choosing the operating point at the streaming base rate gains; $\Ghard$ is
-what remains, which we attribute to a hard-negative separation that anchored
-training never builds.
+what remains, which we attribute to hard temporal negatives that anchored
+training never sees.
 
-Both models give the same answer. For Model A, $\Gtot = +0.126$,
-$\Gprior = -0.001$ and $\Ghard = +0.126$; for Model B, $+0.161$, $-0.001$ and
-$+0.162$. Recalibration recovers essentially none of the gap, and essentially
-all of it is a failure to separate hard negatives.
+Both models give the same split. For Model B, averaged over three seeds,
+$\Gtot = +0.180 \pm 0.006$, $\Gprior = 0.000 \pm 0.001$ and
+$\Ghard = +0.180 \pm 0.006$; for Model A, $+0.203$, $+0.003$ and $+0.200$.
+Threshold re-tuning recovers none of the gap, and all of it falls in $\Ghard$.
 
 The AUC figures explain why. Re-tuning a threshold slides a decision boundary
-along an existing ranking, and when the ranking has fallen to $0.540$, barely
-better than random ordering, no threshold remains to find. The argument reaches
-past thresholds: any method whose main effect moves the operating point,
-including class re-weighting and focal
-loss~\cite{lin2017focal,cui2019classbalanced}, meets the same limit.
+along an existing ranking, and when the ranking has fallen to $0.516$, barely
+better than random ordering, no good threshold exists. The same argument
+predicts that any method whose main effect moves the operating point, including
+class re-weighting and focal loss~\cite{lin2017focal,cui2019classbalanced},
+meets the same limit; \cref{sec:ablations} plans that test.
+
+\subsection{Anchored Training Learns Who Crosses but Not When}
+\label{sec:whowhen}
+The streaming label asks two things at once: whether this pedestrian will cross,
+and whether the crossing starts within $H$. We score the two separately on the
+streaming test set (\cref{tab:whowhen}). The \emph{who} score averages each
+pedestrian's window scores up to the crossing and ranks crossers against
+pedestrians who never cross. The \emph{when} score keeps only windows whose
+crossing still lies ahead and ranks onsets within $H$ against later ones.
+
+\begin{table}[t]
+  \centering
+  \caption{The deployed question split into \emph{who} crosses and \emph{when}.
+  AUC on the streaming test split, mean $\pm$ standard deviation over three
+  seeds. $0.5$ is chance; below $0.5$ ranks in reverse. Hazard: the pure
+  onset-timing configuration of \cref{sec:method}.}
+  \label{tab:whowhen}
+  \setlength{\tabcolsep}{3pt}
+  \footnotesize
+  \begin{tabular}{l ccc}
+    \toprule
+    Trained on, objective & window & who & when \\
+    \midrule
+    anchored, binary  & $0.516 \pm 0.010$ & $\mathbf{0.799 \pm 0.010}$ & $\mathbf{0.414 \pm 0.012}$ \\
+    streaming, binary & $0.783 \pm 0.010$ & $0.766 \pm 0.034$ & $0.753 \pm 0.008$ \\
+    streaming, hazard & $0.793 \pm 0.010$ & $0.766 \pm 0.016$ & $0.766 \pm 0.010$ \\
+    \bottomrule
+  \end{tabular}
+\end{table}
+
+The anchored-trained model identifies crossers better than either streaming
+model, and it orders their windows backwards in time. Its \emph{when} AUC of
+$0.414$ means that, given a window before an imminent crossing and one before a
+later crossing, it scores the later one higher in $59\%$ of pairs. Its separability against never-crossers rises from $0.60$ for onsets
+within a second to $0.71$ for onsets beyond ten seconds, the reverse of every
+streaming-trained model (\cref{sec:discussion}). It therefore alarms early: at
+$205$ alarms per hour it catches $15.1\%$ of crossing pedestrians, about $21$
+seconds ahead on average. The anchored protocol labels tracks
+(\cref{sec:protocols}) and never shows the model a crossing pedestrian far from
+the onset, so nothing in its training separates an imminent crossing from a
+distant one.
+
+The \emph{who} signal does not transfer after training. We fixed the test in
+advance: multiply the anchored-trained model's score by the streaming binary
+baseline's, window by window, and apply the criterion of
+\cref{sec:evalprotocol} over three seeds, pairing runs of the same seed. The
+product detects $42.3\%$ of crossing pedestrians at $205$ alarms per hour and
+$23.1\%$ at $41$, against the baseline's $42.3\%$ and $22.9\%$
+(\cref{tab:seedspread}), and its AUC falls to $0.757$. A combination fitted on
+validation data reproduces the baseline. In these combinations the anchored
+model's \emph{who} signal adds no detection that the streaming model lacks, so
+the repair we pursue changes training (\cref{sec:method}).
 
 \subsection{Threats to Validity}
 \label{sec:threats}
-Six qualifications travel with these results and should stay attached wherever
-the numbers are quoted.
+Four qualifications apply to these results and to those in
+\cref{sec:experiments}.
 
-\textbf{Training set size is confounded with protocol.} The anchored training
+\textbf{Training set size varies with protocol.} The anchored training
 set holds about $4.9$k windows against the streaming set's $88$k. $\Ghard$
 therefore mixes training protocol with training set size, so we describe these
-results as consistent with a hard-negative gap, not as establishing one. The
+results as consistent with a gap caused by hard temporal negatives, not as
+establishing one. The
 matched-size control is one training run.
 \pending{matched-size streaming control, 4.9k windows}
 
 \textbf{F1 is unstable at this base rate.} With one positive in $34$, a handful
-of samples crossing the threshold moves F1 noticeably. The AUC drop, $0.88$ to
-$0.53$ in both models, is the more reliable measurement.
-
-\textbf{Model B has a configuration mismatch.} Its two halves used different
-sampler settings: the streaming half oversampled rare crossing windows and the
-anchored half did not. We therefore cite Model B only for the point that
-multi-task interference does not explain the gap, never for the size of $\Ghard$.
-
-\textbf{\cref{tab:matrix} is single-seed.} Each entry is one run at seed $42$.
-\cref{sec:experiments} reports the three-seed protocol applied to the
-configurations the paper's primary comparison rests on.
-
-\textbf{Checkpoint selection is a second source of run-to-run spread.}
-Validation F1 does not converge over training: it oscillates between $0.05$ and
-$0.25$ for twenty or more epochs without trend, so the selected epoch is the
-maximum of a noisy sequence and every run's validation best overshoots the test
-score its checkpoint delivers. Two runs with different objectives and the same
-seed also collapse on the same epochs, which points at the data order rather
-than the loss. We therefore treat epoch selection as part of the variance
-\cref{tab:seedspread} reports rather than as something the protocol removes, and
-this is a further reason to distrust single-run comparisons at this base rate.
-
-\textbf{We chose the metric after seeing the window-level tie.} We cannot undo
-that ordering. The detection curve is independently motivated by the deployment
-specification and by seven decades of quickest-change-detection
-practice~\cite{page1954cusum}, and \cref{sec:evalprotocol} pre-registers it for
-the seed replicates, but a reader should weigh it knowing the sequence.
+of samples crossing the threshold moves F1 noticeably. The AUC drop, from about
+$0.87$ to about $0.52$ in both models, is the more reliable measurement. Six of
+the $40$ tuned thresholds sit at the lower edge of the validation sweep
+($0.01$), including two that enter the seed-$42$ gap split, so a finer sweep
+could raise those tuned F1 values slightly.
+
+\textbf{The models overfit after an early peak.} In $19$ of the $20$ runs,
+validation AUC peaks within the first eight epochs and then declines while
+training loss keeps falling, to $0.03$ to $0.06$ for the binary baseline. We
+report the checkpoint at the validation peak. The horizon read-out of the onset
+configurations (\cref{sec:method}) also narrows after that epoch. The gap between its 5th and 95th percentiles on validation
+falls to near zero by the last epoch, while the selected checkpoints of the
+pure, hedged and censored configurations keep $0.13$ to $0.57$. The spread
+across three seeds includes this selection step.
+
+\textbf{We chose the metric after seeing the window-level results.} We cannot
+undo that order. The detection curve follows from the deployment requirement
+and from quickest change detection~\cite{page1954cusum}, and
+\cref{sec:evalprotocol} fixed it before the seed replicates, but readers should
+weigh the result knowing the sequence.
 
 % ==========================================================================
 \section{Onset Timing Under Right-Censoring}
@@ -621,16 +666,16 @@ the seed replicates, but a reader should weigh it knowing the sequence.
 \subsection{Where the Binary Label Goes Wrong}
 \cref{eq:binlabel} gets two situations wrong by construction.
 
-A window whose crossing begins at $H + 5$ frames is labeled negative and grouped
-with pedestrians who never cross, even though it looks almost identical to a
-window at $H - 5$ frames, which is labeled positive. This is less a hard
-learning problem than an ill-posed one, and it comes from where we placed the
-cutoff rather than from the data.
+It labels a window whose crossing begins at $H + 5$ frames negative and groups
+it with pedestrians who never cross, even though it looks almost identical to a
+window at $H - 5$ frames, which it labels positive. Only the position of the
+cutoff separates the two labels, which makes the task ill-posed near the cutoff.
 
-A window whose footage ends before $H$ can honestly take neither label. Calling
-it negative asserts an observation nobody made; discarding it, as
-\cref{sec:omits} does, throws away real information. The true statement, that no
-crossing happened before the footage ran out, cannot be written in one bit. Both
+A window whose footage ends less than $H$ frames after its last frame supports
+neither label. Calling it negative asserts an observation nobody made;
+discarding it, as \cref{sec:omits} does, throws away the frames that were
+observed. The true statement, that no
+crossing happened before the footage ran out, does not fit in one bit. Both
 situations are what a right-censored time-to-event dataset looks like when forced
 into binary classification.
 
@@ -643,8 +688,7 @@ For each streaming window we keep three quantities alongside the binary label:
   \item $c$, whether the track contains a crossing at any point.
 \end{itemize}
 In survival-analysis terms $o$ is the event time, $f$ the censoring time, and
-$c$ separates the two situations in which no crossing appears. The
-correspondence is exact.
+$c$ separates the two situations in which no crossing appears.
 
 \subsection{Discrete-Time Hazard}
 Divide the next $L$ frames into $K = L/w$ bins of width $w$. Let $h_k$ be the
@@ -656,8 +700,7 @@ $K_H = H/w$ bins then follows from the survival product:
   P(\text{onset} \le H) = 1 - \prod_{k=0}^{K_H - 1}\left(1 - h_k\right).
   \label{eq:readout}
 \end{equation}
-\cref{eq:readout} is what keeps the comparison with existing results alive,
-because it answers exactly the question \cref{eq:binlabel} asks. The timing
+\cref{eq:readout} answers the question \cref{eq:binlabel} asks, so the timing
 model and the binary baselines report the same quantity under the same metric.
 
 \subsection{Supervision When the Future Is Only Partly Seen}
@@ -669,8 +712,8 @@ shown in \cref{fig:cases}.
   every bin after $e$, since once a first crossing has begun later bins say
   nothing about a first crossing.
   \item \textbf{Onset beyond range} ($o \ge L$). We supervise all $K$ bins as
-  $0$. This is honest: the crossing was seen, and it fell outside the window's
-  look-ahead.
+  $0$. These zeros come from observation: the footage shows the crossing, and it
+  falls outside the look-ahead.
   \item \textbf{Censored} ($o < 0$, with some future seen). We supervise the
   bins covered by $f$ as $0$ and mask the rest. This is the case the binary
   label cannot express.
@@ -682,8 +725,8 @@ shown in \cref{fig:cases}.
 \begin{figure}[t]
   \centering
   \includegraphics[width=\columnwidth]{fig4_hazard_cases.pdf}
-  \caption{The four supervision cases. Shaded bins contribute to the loss and
-  dashed bins are masked, receiving no gradient; case~4 leaves the risk set
+  \caption{The four supervision cases. Shaded bins contribute to the loss; the
+  loss masks dashed bins, which receive no gradient; case~4 leaves the risk set
   altogether. Case~3 is the situation the binary label cannot express, and
   case~4 is the one it cannot get right.}
   \label{fig:cases}
@@ -696,29 +739,30 @@ The loss is a masked per-bin binary cross-entropy:
     \Big[\, y_k \log h_k + (1 - y_k)\log(1 - h_k) \,\Big].
   \label{eq:loss}
 \end{equation}
-The masking is the substantive part. A bin nobody observed must receive exactly
-zero gradient, not a small gradient toward zero, and we check this with
-automatic differentiation rather than by reading the code.
-
-The implementation enforces two constraints. First, $L > H$ strictly. A look-ahead no wider than the reported horizon would
-still hand a flat negative label to a crossing at $H + 5$, the defect the
-formulation exists to remove. Second, the read-out horizon must match
-the horizon used by the binary baselines, or \cref{eq:readout} quietly answers a
-different question under the same metric name. A test enforces the second by
+The mask carries the method. A bin nobody observed receives exactly zero
+gradient, not a small push toward zero, and a unit test checks this with
+automatic differentiation.
+
+The implementation enforces two constraints. First, $L > H$ strictly. A
+look-ahead no wider than the reported horizon would still give a flat negative
+label to a crossing at $H + 5$, the defect the formulation exists to remove.
+Second, the read-out horizon must match the horizon used by the binary
+baselines, or \cref{eq:readout} silently answers a different question under the
+same metric name. A test enforces the second by
 reproducing the data generator's own binary label from the timing read-out over
 a full track.
 
 \subsection{Choosing the Look-Ahead and Bin Width}
 The reported horizon stays at $H = 32$ frames (\cref{sec:horizon}). The
 look-ahead $L$ is a separate choice, sized against the composition measured in
-\cref{sec:omits}: $L = 96$ frames, or $3.2$\,s, covers about $92\%$ of the
-confusable band, against $47\%$ at $L = 60$. Windows whose onset falls inside
-$L$ carry a real ``crossing begins here'' signal; at $L = 96$ that applies to
-roughly $6{,}400$ training windows against $2{,}530$ at $L = H$, nearly tripling
-the windows carrying positive supervision in a task with a $2.9\%$ positive
-rate. Going wider buys little, since two thirds of the hard negative mass sits
-more than five seconds ahead. A bin width $w = 4$ gives $K = 24$ bins, keeps $8$
-inside the reported horizon, makes positives about four times denser per bin,
+\cref{sec:omits}. The confusable band ends $64$ frames past $H$, so
+$L = 96$ frames ($3.2$\,s) covers all of it, against $47\%$ at $L = 60$.
+Windows whose onset falls inside $L$ carry a ``crossing begins here'' signal; at
+$L = 96$ that applies to $6{,}716$ training windows against $2{,}530$ at
+$L = H$, $2.7$ times as many windows with positive supervision in a task with a
+$2.9\%$ positive rate. Going wider buys little, since two thirds of hard
+temporal negatives lie more than five seconds ahead. A bin width $w = 4$ gives
+$K = 24$ bins, keeps $8$ inside the reported horizon, makes positives about four times denser per bin,
 and costs $133$\,ms of timing resolution. Both settings affect training only.
 
 \subsection{Three Configurations of One Objective}
@@ -731,13 +775,13 @@ The \emph{hedged} configuration adds a direct read-out term, because under
 \cref{eq:loss} sums over the bins a window observed, so a fully observed window
 contributes about $17$ nats at initialization against $5.5$ for one carrying
 only the $32$ future frames generation guarantees. The term therefore starts far
-larger than a per-task cross-entropy and must be scaled down beside the binary
+larger than a per-task cross-entropy, so we scale it down beside the binary
 objective.
 
 \textbf{The failure mode to watch for.} With $K$ bins each bin sees a positive
 only about $2.9\%/K$ of the time, so the head can drive the loss down by
 predicting $h \approx 0$ everywhere, collapsing \cref{eq:readout} toward zero. A
-low mean hazard is correct and will not reveal this; the spread of the read-out
+low mean hazard is normal and will not reveal this; the spread of the read-out
 across validation windows will, since a narrow spike at the base rate means the
 head learned the average. Setting $w = 4$ is the mitigation applied, and $w = 8$
 halves $K$ again.
@@ -754,32 +798,34 @@ split produces one decision, which at stride $3$ and $30$\,fps gives
 \decisionsPerHour{} decisions per hour. A crossing pedestrian counts as detected
 when any window whose crossing still lies ahead scores above threshold, and the
 lead time is the largest such offset, the earliest warning the model would have
-raised. Alarms after the person has stepped out earn no credit and no penalty; a
-pedestrian who never crosses contributes a false alarm. The test split holds
-\Ncrossers{} crossing and \Nnoncrossers{} never-crossing pedestrians. Sweeping
-the threshold traces detection rate against alarm rate, read at five budgets:
-$1200$, $460$, $205$, $95$ and $41$ alarms per hour. Each budget fixes the
-lowest threshold whose realised alarm rate still fits, so a configuration cannot
-buy detections with alarms. The accounting comes from quickest change
-detection~\cite{page1954cusum}, which has measured detection delay against time
-between false alarms since the 1950s. We take its accounting and leave its
-detector: CUSUM accumulation measures $2$ to $10\times$ worse here, because a
-crossing approach is not the persistent distribution shift it assumes.
-
-\textbf{Two accounting rules, both reported.} Counting every alarming window and
-counting every nuisance pedestrian once give different answers that disagree at
-loose budgets. We report both and name the rule beside every number.
+raised. Alarms after the person has stepped out earn no credit and no penalty.
+Every window of a pedestrian who never crosses is a chance for a false alarm.
+The test split holds \Ncrossers{} crossing and \Nnoncrossers{} never-crossing
+pedestrians. Sweeping the threshold traces detection rate against alarm rate,
+read at five budgets: $1200$, $460$, $205$, $95$ and $41$ alarms per hour. Each
+budget fixes the lowest threshold whose realized alarm rate still fits, so a
+configuration cannot buy detections with alarms. The accounting comes from
+quickest change detection~\cite{page1954cusum}, which has measured detection
+delay against time between false alarms since the 1950s. We take its accounting
+and leave its cumulative-sum (CUSUM) detector, which assumes a persistent shift
+where a crossing approach is a brief change.
+
+\textbf{Two accounting rules, both reported.} \texttt{per\_window} accounting
+counts every alarming window. \texttt{per\_track} accounting counts each
+never-crossing pedestrian who triggers any alarm once, which matches what a
+driver experiences. Both divide by the same exposure time. They disagree at
+loose budgets, so we name the rule beside every number.
 
 \textbf{Window metrics appear alongside, never instead.} We give AUC and tuned
-F1 for every configuration, because \cref{sec:windowtie} needs them as evidence
-that the standard metrics cannot see what this objective changes.
+F1 for every configuration, because \cref{sec:windowtie} uses them to measure
+how much seed noise the standard metrics carry.
 
 \textbf{Pre-registration.} We fixed the budgets, accounting rules, metrics and
 success criterion before running the seed replicates: the pure hazard
 configuration's mean detection rate must exceed the binary baseline's by more
 than the sum of their standard deviations at three or more of the five budgets,
 and anything less we report as inconclusive. \cref{sec:threats} records that we
-chose this metric after observing the window-level tie.
+chose this metric after seeing the window-level results.
 
 \subsection{Baselines}
 \label{sec:baselines}
@@ -787,13 +833,15 @@ chose this metric after observing the window-level tie.
 single crossing head and the standard cross-entropy objective at $H = 32$,
 sharing the architecture, optimizer, sampler and schedule of every onset
 configuration and differing only in the loss. The paper rests on this
-comparison, so it runs on the same three seeds as the method arms.
+comparison, so the baseline runs on the same three seeds as the pure hazard
+configuration.
 
 \textbf{Onset configurations.} The three weightings of \cref{sec:method}
 (auxiliary, pure, hedged), plus a fourth adding the \seedconf{7\,470}
-right-censored training windows the generator otherwise discards. All four read
-out $P(\text{onset} \le 32)$ through \cref{eq:readout}, so every row answers the
-same question at the same horizon.
+right-censored training windows the generator otherwise discards. The pure,
+hedged and censored configurations report $P(\text{onset} \le 32)$ through
+\cref{eq:readout}; the auxiliary configuration reports its binary head. Every
+row therefore answers the same question at the same horizon.
 
 \textbf{What we do not compare against.} We retrain no published
 crossing-intention method. Those methods target the anchored protocol, and a
@@ -801,15 +849,13 @@ retrained version would measure how well we reimplemented them. The baseline
 above isolates the single variable this paper manipulates, the training
 objective.
 
-\subsection{Window Metrics Cannot Separate the Configurations}
+\subsection{Window Metrics Cannot Rank the Configurations With Confidence}
 \label{sec:windowtie}
 \begin{table}[t]
   \centering
   \caption{Window-level metrics on the streaming test split
-  ($n = \Ntest$), at validation-tuned thresholds. Blue daggers mark
-  single-seed values; red markers flag values we have not measured. The spread between two
-  seeds of the \emph{same} configuration exceeds every difference
-  \emph{between} configurations.}
+  ($n = \Ntest$), at validation-tuned thresholds. Every configuration runs
+  seeds $42$, $43$ and $44$; means carry the sample standard deviation.}
   \label{tab:onset}
   \begin{tabular}{l l cc}
     \toprule
@@ -827,149 +873,115 @@ objective.
       & 44 & \pureAUCc & \pureFc \\
       & \emph{mean} & \pureAUCmean\,$\pm$\,\pureAUCsd & \pureFmean\,$\pm$\,\pureFsd \\
     \midrule
-    auxiliary hazard   & 42 & \oneseed{\auxAUC}{aux AUC}   & \oneseed{\auxF}{aux F1} \\
-    hedged             & 42 & \oneseed{\hedgeAUC}{hedge AUC} & \oneseed{\hedgeF}{hedge F1} \\
-    pure $+$ censored  & 42 & \oneseed{\censAUC}{cens AUC} & \oneseed{\censF}{cens F1} \\
+    auxiliary hazard   & \emph{mean} & \auxAUC\,$\pm$\,\auxAUCsd & \auxF\,$\pm$\,\auxFsd \\
+    hedged             & \emph{mean} & \hedgeAUC\,$\pm$\,\hedgeAUCsd & \hedgeF\,$\pm$\,\hedgeFsd \\
+    pure $+$ censored  & \emph{mean} & \censAUC\,$\pm$\,\censAUCsd & \censF\,$\pm$\,\censFsd \\
     \bottomrule
   \end{tabular}
 \end{table}
 
 \cref{tab:onset} gives the window-level picture. Across the five configurations
-AUC spans \seedconf{0.746} to \seedconf{0.777} and tuned F1 spans
-\seedconf{0.163} to \seedconf{0.259}, and no configuration separates from the
-baseline by more than the measurement noise. Over three seeds the binary
-baseline gives a tuned F1 of \baseFmean\,$\pm$\,\baseFsd{} and the pure hazard
-configuration \pureFmean\,$\pm$\,\pureFsd. Each interval covers most of the
-range all five configurations occupy, and the two overlap almost completely.
-Seed choice moves F1 further than objective choice does.
-
-Three further measurements say the same thing. A moving average over each
-track's scores, a post-processing step that changes no model, moves AUC by
-\smoothAUCgain{}, which is larger than the \armAUCgap{} that separates the best
-and worst configurations. AUC ranks the configuration that detects the most
-crossings last. Average precision agrees with the detection ordering on three
-configurations and then inverts on the fourth.
-
-\textbf{Takeaway.} At a class ratio of $33.9{:}1$ seed noise and post-processing
-choices dominate the standard window-level scores, so they cannot rank these
-configurations. This is a property of the metric at this base
-rate, and it holds whichever configuration wins: a paper that reported only
-\cref{tab:onset} would conclude that the objective makes no difference.
-
-\subsection{The Objective Difference Does Not Survive Replication}
+and three seeds, AUC spans \seedconf{0.773} to \seedconf{0.808} and tuned F1
+spans \seedconf{0.178} to \seedconf{0.303}. On AUC, no configuration's mean
+differs from the baseline's by more than the sum of their standard deviations.
+On tuned F1, the pure hazard configuration averages \pureFmean\,$\pm$\,\pureFsd{}
+against the baseline's \baseFmean\,$\pm$\,\baseFsd, a \FmeanGap{} gap that
+exceeds the $0.033$ sum of the two deviations by $0.001$. The other three
+configurations stay inside that sum.
+
+A post-processing step moves AUC as far as the objectives differ. A centered
+moving average over each track's scores, which changes no model, raises the
+baseline's AUC by \smoothAUCgain, the size of the \AUCmeanGap{} gap between the
+binary and pure hazard means. A causal average, which a deployed detector could
+run, lowers it by \smoothAUCcausal.
+
+\textbf{Takeaway.} At a class ratio of $33.9{:}1$, the window-level scores move
+between configurations by about as much as a post-processing choice moves them.
+The one gap that clears the seed spread, pure hazard's F1, clears it by
+$0.001$, so these scores cannot rank the configurations with confidence.
+
+\subsection{The Objectives Tie on Detection Rate}
 \label{sec:detection}
 \begin{table}[t]
   \centering
-  \provtable{tab:detection, seed 42, illustrative only}
-  \caption{Detection rate at matched false-alarm budgets, \texttt{per\_window}
-  accounting, streaming test split, \emph{seed $42$ only}. Lead time in seconds
-  in parentheses. Reported here because it is what a single-seed study would
-  conclude; \cref{tab:seedspread} is the result.}
-  \label{tab:detection}
-  \setlength{\tabcolsep}{3.5pt}
+  \caption{\textbf{The paper's primary result.} Detection rate, mean $\pm$ sample
+  standard deviation over three seeds per objective, \texttt{per\_window}
+  accounting, streaming test split. Mean lead time in seconds in parentheses.
+  The two objectives differ by less than the run-to-run spread at every budget,
+  so no budget meets the pre-registered criterion.}
+  \label{tab:seedspread}
+  \setlength{\tabcolsep}{4pt}
   \small
-  \begin{tabular}{l cccc}
+  \begin{tabular}{l cc}
     \toprule
-    Alarms & binary & \textbf{pure} & hedged & pure $+$ \\
-    /hour  & baseline & \textbf{hazard} &      & censored \\
+    Alarms/hour & binary ($n{=}3$) & pure hazard ($n{=}3$) \\
     \midrule
-    $1200$ & 40.5 (4.5) & \textbf{66.8} (6.5) & 49.8 (4.8) & 65.4 (\textbf{8.1}) \\
-    $460$  & 21.5 (4.1) & \textbf{56.6} (4.4) & 28.8 (3.2) & 45.4 (4.6) \\
-    $205$  & 15.6 (3.4) & \textbf{48.8} (3.1) & 14.6 (2.7) & 31.2 (3.1) \\
-    $95$   & 10.7 (2.9) & \textbf{41.0} (1.8) &  9.3 (2.7) & 18.5 (2.4) \\
-    $41$   &  6.8 (1.7) & \textbf{34.1} (0.7) &  3.9 (1.3) & 10.7 (\textbf{2.5}) \\
+    $1200$ & $69.6 \pm 6.0$ (7.3) & $72.0 \pm 2.8$ (7.0) \\
+    $460$  & $55.0 \pm 6.2$ (5.8) & $59.2 \pm 4.9$ (4.7) \\
+    $205$  & $42.3 \pm 3.2$ (4.7) & $44.1 \pm 4.7$ (3.4) \\
+    $95$   & $32.0 \pm 4.2$ (3.8) & $31.4 \pm 5.9$ (3.0) \\
+    $41$   & $22.9 \pm 3.5$ (3.0) & $17.4 \pm 9.1$ (2.7) \\
     \bottomrule
   \end{tabular}
 \end{table}
 
-At seed $42$ the result looks decisive. \cref{tab:detection} shows the pure
-hazard configuration detecting more crossing pedestrians than the binary
-baseline at every budget, by $1.7\times$ at $1200$ alarms per hour, $3.1\times$
-at $205$ and $5.0\times$ at $41$, while window-level F1 separates the two by
-nothing at all (\cref{tab:onset}). Read on its own, that is a clean result: a
-reformulation whose benefit the standard metric cannot see.
-
-\textbf{Three seeds of each objective remove it.} \cref{tab:seedspread} gives
-the pre-registered comparison. At $205$ alarms per hour the baseline's three
-seeds detect $15.6\%$, $52.2\%$ and $29.3\%$, and the hazard configuration's
-detect $48.8\%$, $25.4\%$ and $30.7\%$. The two sets interleave. Pooled, the
-means differ by between $-2.3$ and $+7.6$ percentage points, against standard
-deviations of $8$ to $22$, and the hazard configuration is behind at the loosest
-budget. \textbf{The criterion fixed in \cref{sec:evalprotocol} is met at none of
-the five budgets, so we report the comparison as inconclusive.} The $3.1\times$
-of \cref{tab:detection} was the best hazard run measured against the worst
-baseline run.
-
-\textbf{Neither the accounting rule nor lead time rescues it.} Charging each
-nuisance pedestrian once however many windows it alarms on, the two objectives
-trade places: the baseline leads at $120$ alarms per hour ($69.9 \pm 8.9$ against
-$64.2 \pm 4.7$) and at $20$, the hazard configuration at $46$ and at $4$, and no
-budget separates them by more than the spread. Mean lead time favours the hazard
-configuration, but one seed carries it: at $41$ alarms per hour its three runs
-give $0.72$, $8.83$ and $1.23$ seconds against the baseline's $1.70$, $1.99$ and
-$1.76$.
-
-One asymmetry is worth recording without being claimed. The hazard
-configuration's seed-to-seed standard deviation is the smaller of the two at
-four of five budgets, and its means sit at or above the baseline's at four of
-five. Neither margin approaches the criterion, and standard deviations from
-three runs are crude, so this is a direction for more seeds rather than a
-finding.
+\cref{tab:seedspread} gives the pre-registered comparison. The two objectives'
+means lie within $0.7$ to $5.5$ percentage points of each other at every budget,
+against standard deviations of $2.8$ to $9.1$. The hazard configuration leads at
+the three loosest budgets and trails at the two tightest. \textbf{No budget meets
+the criterion fixed in \cref{sec:evalprotocol}, so we report the comparison as
+inconclusive.}
+
+\textbf{The seeds agree with each other.} At $205$ alarms per hour the baseline's
+three seeds detect $44.4\%$, $43.9\%$ and $38.5\%$ of crossing pedestrians, and
+the hazard configuration's detect $44.9\%$, $39.0\%$ and $48.3\%$. With spread
+of this size the criterion would register a gap of $8$ to $13$ points, and the
+two objectives never differ by more than $5.5$.
+
+\textbf{Neither the accounting rule nor lead time changes this.} Under
+\texttt{per\_track} accounting, both objectives detect every crossing pedestrian
+at $1200$ alarms per hour, since each never-crossing pedestrian can alarm at most
+once and the realized rate cannot pass about $478$ per hour. At the tighter budgets
+the hazard configuration leads by $0.5$ to $2.4$ points, inside the spread, for
+example $47.0 \pm 4.1$ against $44.6 \pm 4.6$ at $41$ alarms per hour. Mean lead
+time favors the binary baseline at every budget, by $0.3$ to $1.3$ seconds
+($4.7$ against $3.4$ seconds at $205$ alarms per hour).
 
 \begin{table}[t]
   \centering
-  \caption{\textbf{The paper's primary result.} Detection rate, mean $\pm$ sample
-  standard deviation over three seeds per objective, \texttt{per\_window}
-  accounting, streaming test split. Mean lead time in seconds in parentheses.
-  The two objectives differ by less than the run-to-run spread at every budget,
-  so the pre-registered criterion is met at none of them.}
-  \label{tab:seedspread}
-  \setlength{\tabcolsep}{4pt}
+  \caption{Detection rate of the other onset configurations, mean $\pm$ sample
+  standard deviation over three seeds, \texttt{per\_window} accounting,
+  streaming test split. None meets the pre-registered criterion against the
+  binary baseline of \cref{tab:seedspread}, under either accounting rule.}
+  \label{tab:detection}
+  \setlength{\tabcolsep}{3.5pt}
   \small
-  \begin{tabular}{l cc}
+  \begin{tabular}{l ccc}
     \toprule
-    Alarms/hour & binary ($n{=}3$) & pure hazard ($n{=}3$) \\
+    Alarms/hour & auxiliary & hedged & pure $+$ censored \\
     \midrule
-    $1200$ & $62.0 \pm 19.0$ (6.5) & $59.7 \pm 10.4$ (6.7) \\
-    $460$  & $45.2 \pm 22.0$ (5.0) & $45.9 \pm 11.3$ (5.2) \\
-    $205$  & $32.4 \pm 18.5$ (3.9) & $35.0 \pm 12.3$ (4.3) \\
-    $95$   & $21.3 \pm 16.2$ (2.9) & $25.4 \pm 13.6$ (3.8) \\
-    $41$   & $11.9 \pm\phantom{0}7.9$ (1.8) & $19.5 \pm 12.8$ (3.6) \\
+    $1200$ & $69.1 \pm 6.6$ & $69.1 \pm 12.1$ & $72.2 \pm 2.1$ \\
+    $460$  & $55.0 \pm 5.2$ & $56.4 \pm 11.3$ & $59.7 \pm 2.3$ \\
+    $205$  & $42.1 \pm 2.7$ & $46.7 \pm 11.4$ & $47.8 \pm 3.0$ \\
+    $95$   & $29.4 \pm 3.9$ & $37.4 \pm 9.8$  & $37.7 \pm 2.5$ \\
+    $41$   & $17.6 \pm 6.1$ & $23.4 \pm 11.7$ & $18.7 \pm 7.5$ \\
     \bottomrule
   \end{tabular}
 \end{table}
 
-\textbf{Two single-seed observations, offered as such.} The hedged
-configuration, which trains both objectives together, trails the baseline from
-$205$ alarms per hour downward and trails both pure configurations everywhere
-(\cref{sec:hedgeresult}). Adding the discarded right-censored windows costs
-detection rate against the pure configuration at every budget while lengthening
-lead time at the tightest, from $0.7$ to $2.5$ seconds. We ran one seed of each,
-and \cref{tab:seedspread} is the reason not to read either as a result.
-
-\textbf{Takeaway.} The detection curve separates the objectives cleanly at a
-fixed seed and not at all across three of them. Each objective varies more
-between its own seeds than the two differ from each other, on the metric chosen
-precisely because window-level scores could not tell them apart. The
-reformulation of \cref{sec:method} is therefore a sound way to state the task,
-and this evidence does not show that it detects crossings better. A single-seed
-study at this base rate, which is what the standard protocol invites, would have
-reported $3.1\times$.
-
-\subsection{The Multi-Task Variant Is the Weakest of the Four}
-\label{sec:hedgeresult}
-The hedged configuration, which trains the hazard objective and a fixed-horizon
-cross-entropy together, scores below both pure configurations and below the
-binary baseline on tuned F1 (\oneseed{\hedgeF}{hedge F1}
-against \baseFa{} and \pureFa{} at seed $42$). Its read-out also spreads widest
-across validation windows, which says the head is confident and wrong more
-often. The two objectives disagree about what a window whose future ran out
-should contribute, and training both at once resolves that disagreement in
-neither direction.
-
-\textbf{Takeaway.} This answers the natural question of whether the formulation
-reduces to multi-task learning. It does not: on the one seed we ran of it, the
-multi-task configuration is the weakest of the four.
+\textbf{The other configurations tie with the baseline as well}
+(\cref{tab:detection}). Adding the right-censored windows gives the highest mean
+at $205$ alarms per hour, $47.8 \pm 3.0$ against the baseline's
+$42.3 \pm 3.2$, a lead of $5.5$ points that falls short of the $6.2$ the
+criterion requires. The hedged configuration varies most between seeds, from
+$33.7\%$ to $55.1\%$ at $205$ alarms per hour.
+
+\textbf{Takeaway.} Three seeds of each objective, with seed-to-seed spread of a
+few points, show no detection difference between onset timing and binary
+training that clears the seed spread, at any budget or under either accounting
+rule. The binary baseline warns earlier on average. This evidence does not show
+that the reformulation of
+\cref{sec:method} detects crossings better than binary training.
 
 \subsection{Planned Ablations}
 \label{sec:ablations}
@@ -981,7 +993,8 @@ which separates our claim from a claim about class imbalance. We compare
 frame-pooling rules, which are configurable and cost nothing to implement.
 \todo[inline]{We have run none of these yet. If the deadline forces a choice, the
 focal and class-balanced comparison is the one a reviewer will ask for, because
-without it the improvement could be read as a reweighting effect.}
+\cref{sec:gap} predicts that reweighting cannot repair the ranking and this is
+the direct test.}
 
 % ==========================================================================
 \section{Discussion}
@@ -993,24 +1006,26 @@ turns abruptly into the road. Three seconds earlier the information sits outside
 the video, either because the decision is unmade or because it has not reached
 the body. No feature, architecture or loss can recover it.
 
-Some unknown share of the confusable band is therefore impossible, and claiming
-to handle hard negatives in general would overstate what is achievable. Three
-consequences follow. The goal becomes confidence where evidence exists and
-uncertainty where it does not, a matter of ranking and calibration, which
-matches the failure we measured: the ordering degraded while the operating point
-held. The binary label is a modeling choice, and \cref{sec:method} removes it.
+Some unknown share of the confusable band is therefore impossible to predict,
+and claiming to handle hard temporal negatives in general would overstate what
+is achievable. Two consequences follow. First, the goal becomes confidence where
+evidence exists and uncertainty where it does not. That is a matter of ranking
+and calibration, and it matches the failure we measured: the ranking degraded,
+and re-tuning the threshold recovered nothing. Second, the binary label is a
+modeling choice, and \cref{sec:method} replaces it.
 
 Where the limit sits is measurable, and we report it. Scoring windows at a given
 time to onset against windows of pedestrians who never cross, separability decays
-monotonically: AUC $0.78$ inside the first second, $0.69$ between two and three
-seconds, $0.63$ from five to ten, and $0.52$ to $0.54$ beyond ten seconds, which
-is chance. The decay is far larger than the seed spread, which stays under
-$0.04$ at every stratum except the last, and both objectives trace the same
-curve to within that spread. Two things follow. Crossing is predictable from
-this input for roughly ten seconds and no longer, which bounds what any method
-on PIE can achieve and supports keeping the horizon short (\cref{sec:horizon}).
-And a difference between training objectives is not where the remaining headroom
-lies. We have not found this measurement reported elsewhere in the
+monotonically. AUC is $0.81$ to $0.82$ inside the first second, $0.72$ to $0.73$
+between two and three seconds, $0.64$ to $0.66$ from five to ten, and $0.52$ to
+$0.54$ beyond ten seconds, which is chance. The decay is far larger than the
+seed spread, which stays under $0.03$ at every stratum, and both objectives
+trace the same curve to within $0.02$. The anchored-trained model shows the
+reverse profile (\cref{sec:whowhen}). Two things follow. For these models, crossing is
+predictable from this input for roughly ten seconds and no longer, which
+suggests a limit for any method using this input on PIE and supports keeping the
+horizon short (\cref{sec:horizon}). The choice of training objective also does
+not move this limit. We have not found this measurement reported elsewhere in the
 crossing-intention literature, and it costs one evaluation pass over an existing
 model.
 
@@ -1018,12 +1033,10 @@ model.
 Beyond the qualifications in \cref{sec:threats}, all results come from PIE,
 recorded in one city in daylight and clear weather; replication on JAAD would
 show whether the effect is specific to this dataset. The streaming training runs
-are also unstable, with validation loss spiking and recall snapping to $1.0$ on
-several epochs as the model briefly collapses into calling everything a
-crossing. Two configurations with different objectives collapse on identical
-epochs, which points at data ordering rather than the loss, and the
-streaming-trained entries therefore rest on a selected best epoch that may not
-be a settled optimum.
+are also unstable: on several epochs the model briefly calls every window a
+crossing, and recall jumps to $1.0$. As \cref{sec:threats} notes, this points
+at data ordering, and the streaming-trained entries rest on a selected best
+epoch that may not be a settled optimum.
 
 \subsection{What This Means for Evaluation Practice}
 If $\Ghard \approx \Gtot$ holds more widely, a model's standing under the
@@ -1035,11 +1048,12 @@ PIE and JAAD~\cite{rasouli2024diving} shows the adequacy of the standard
 evaluation is already an open concern; our results suggest the sampling
 protocol, and not only the metrics, deserves part of that attention.
 
-Our own comparison adds a second requirement. Three seeds of one objective span
-$15.6$ to $52.2$ percentage points of detection rate at a fixed alarm budget,
-wider than any gap we measured between objectives, so at this base rate a
-single-run result says little about the objective that produced it, whichever
-metric is reported. Streaming evaluation on PIE needs run-to-run spread reported
+Our own comparison adds a second requirement. At $205$ alarms per hour, three
+seeds of the binary objective detect between $38.5\%$ and $44.4\%$ of crossing
+pedestrians, a range as wide as the largest gap we measured between objectives.
+At this base rate a single-run result therefore says little about the objective
+that produced it, whichever metric a study reports. Streaming evaluation on PIE
+should report run-to-run spread
 as a matter of course, and a difference between methods needs to clear it first.
 
 % ==========================================================================
@@ -1052,7 +1066,7 @@ the anchored protocol leaves out, finding that a quarter of streaming training
 windows and two fifths of test windows show pedestrians who cross later, and we
 showed that a model trained under the anchored protocol loses almost all ranking
 ability on the streaming test set, with threshold re-tuning recovering
-essentially none of the loss. Since part of the difficulty lies in the labeling
+almost none of the loss. Since part of the difficulty lies in the labeling
 rather than in the model alone, we proposed treating the task as onset timing
 under right-censoring. A window whose future footage runs out is then censored
 rather than negative, a pedestrian who has already crossed leaves the risk set
@@ -1066,8 +1080,8 @@ binary training. One seed of the same comparison reports a threefold
 improvement. We take that gap between one run and three as the practical finding:
 at a $33.9{:}1$ base rate the evaluations this field relies on, window-level
 scores and detection curves alike, cannot support a claim about an objective from
-a single run. Reporting the streaming protocol without also reporting run-to-run
-spread will keep producing improvements that do not replicate.
+a single run. Streaming results reported without run-to-run spread can show
+improvements that do not replicate.
 
 \bibliographystyle{IEEEtran}
 \bibliography{refs}
```

## Staged diff

```text

```

