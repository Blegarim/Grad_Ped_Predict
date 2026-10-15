# Paper

"Who Crosses and When: What Event-Anchored Benchmarks Measure in Pedestrian Crossing Prediction", for AIAM 2026
(IET Conference Proceedings: 8 pages including references, UK English). `main.tex` is `article`-based and
replicates the IET sample layout.

## Build

```bash
cd paper/figures && ../../.venv/Scripts/python.exe make_tables.py && ../../.venv/Scripts/python.exe make_figures.py
cd .. && python iet_bib.py
max_print_line=1000 latexmk -pdf -bibtex- -interaction=nonstopmode -halt-on-error main.tex
./check_numbers.sh
```

- `make_tables.py` writes `generated/numbers.tex` and the table bodies `generated/tab_*.tex` from the result
  files (see Numbers). Rerun it whenever a result file changes. Never edit `generated/` by hand.
- `make_figures.py` writes the figure PDFs; the separability figure reads the same result files.
- `iet_bib.py` writes `refs_iet.tex` (the cited `refs.bib` entries in IET style; no IET `.bst` exists).
  Rerun it whenever `refs.bib` or the `\cite` keys change.
- `max_print_line=1000` keeps the result-status warnings on one line so `check_numbers.sh` can read them.
- A build is good when `latexmk` finishes, the PDF has exactly 8 pages, and `check_numbers.sh` reports zero
  unresolved references and zero overfull boxes. `latexmk -C` cleans.

## Numbers

A number in the text is `\V{key}` only if it appears more than once (in the text, or in the text and a table)
or is a derived number behind a central claim. A number reported once is typed in directly, so the source stays
readable. `make_tables.py` still generates every key with its value and its source file (one
`\defV{key}{value}% source` line per number), so a typed-in number is traced by searching for its value in
`generated/numbers.tex`, and must be retyped by hand if its result file changes. The keys come from:

- `outputs/diagnostics/tree_study_f64/results.json` (tree study; the as-registered `tree_study/` only for the
  pre-fix verdict) and `tree_study_f64/posthoc/bz32_vs_b.json` (post hoc contrast);
- `outputs/diagnostics/scene_audit/{results,features_summary}.json` and `posthoc/*.md`;
- `outputs/diagnostics/waiting_audit/results.json`;
- `outputs/diagnostics/v4_report/tables/results.json` (neural network) and `v5_report/*/curves.json` plus
  `outputs/runs/*_c5_*/eval_log.csv` (re-weighted losses);
- `paper/composition_report.json`, and the local test dumps for counts (pedestrians, anchored ratio);
- fixed settings (horizon, budgets, learner settings) in its `CONSTANTS` table, each with its source.

An unknown key stops the build, so a renamed or missing macro number cannot pass silently. Table bodies are generated
the same way and read with `\tabrows{name}` (the primitive `\input`, which works inside a tabular).

Draft-state markers, defined in the preamble:

| Macro | Meaning | Draft render |
|---|---|---|
| `\pending{tag}` | text or value not written yet | red `[tag]` |
| `\oneseed{v}{tag}` | measured on one seed, seed mean expected | blue value + † |

`\submissiontrue` hides the `\todo` notes and the marks, and turns any surviving `\pending` into a compile
error. `./check_numbers.sh --strict` exits nonzero while anything is unfilled.
