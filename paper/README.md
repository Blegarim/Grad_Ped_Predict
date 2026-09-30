# Paper

Streaming crossing-onset paper for AIAM 2026 (IET Conference Proceedings: 8 pages including references,
UK English). `main.tex` is `article`-based and replicates the IET sample layout.

## Build

```bash
cd paper/figures && ../../.venv/Scripts/python.exe make_figures.py
cd .. && python iet_bib.py
max_print_line=1000 latexmk -pdf -bibtex- -interaction=nonstopmode -halt-on-error main.tex
./check_numbers.sh
```

- `iet_bib.py` writes `refs_iet.tex` (the cited `refs.bib` entries in IET style; no IET `.bst` exists).
  Rerun it whenever `refs.bib` or the `\cite` keys change.
- `max_print_line=1000` keeps the result-status warnings on one line so `check_numbers.sh` can read them.
- A build is good when `latexmk` finishes and `check_numbers.sh` reports zero unresolved references and
  zero overfull boxes. `latexmk -C` cleans.

## Numbers

Every number in `main.tex` is in one of three states, defined in the preamble:

| Macro | Meaning | Draft render |
|---|---|---|
| `\seedconf{v}` | measured, confirmed across seeds | the value |
| `\oneseed{v}{tag}` | measured on one seed | blue value + † |
| `\pending{tag}` | not measured yet | red `[tag]` |

`\submissiontrue` hides the `\todo` notes and the marks, and turns any surviving `\pending` into a compile
error. `./check_numbers.sh --strict` exits nonzero while anything is unfilled.

Sources: figure values live in the `NUMBERS` dict in `figures/make_figures.py`; results come from
`outputs/runs/RESULTS_MATRIX.md` (update it first, then mirror into `main.tex`); window populations from
`CLAUDE.md` § Dataset Statistics; negative composition from `composition_report.json`
(`scripts/report_negative_composition.py`).
