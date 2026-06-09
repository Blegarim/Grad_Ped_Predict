# Golden-fixture regenerators

These scripts regenerate the committed fixtures in `tests/fixtures/golden/` from the **legacy
reference repo**, which was retired from `main` at the v1.0 cutover and now lives only in the
`legacy-archive` git tag.

To re-capture a fixture:

```bash
git checkout legacy-archive    # restores OLD/Undergrad_thesis_project/
python tests/_capture/capture_<module>_golden.py
git checkout main              # bring the regenerated fixture back
```

On `main` the committed fixtures are the source of truth — the per-module tests load them directly as
characterization references for the current `src/pedpredict` numerics. You only need these scripts if
you intend to deliberately re-baseline a fixture against the archived legacy behavior.
