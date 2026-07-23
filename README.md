# syn-vs-enc-simplified

## Usage

```
python main.py run-experiment --config config/main.yaml
python main.py run-experiment --config config/experiments/DRAC-fhe.yaml --n-bits 4
python main.py list-n-bits                                  # expand config/fhe.yaml's sweep -> fhe_n_bits.txt
python main.py run-single-internal-validation-bootstrap --config config/main.yaml --seed 42
python main.py generate-seeds --length 1000                 # -> internal_validation_bootstrap_seeds.txt
python main.py aggregate-internal-validation-bootstrap
```

To parallelize the FHE n_bits sweep, run `list-n-bits` then launch one
`run-experiment --n-bits N` process per value (locally, or as separate
cluster jobs) — each is independent and writes uniquely-named output files,
mirroring how internal validation bootstrap parallelizes across seeds via `run-single-internal-validation-bootstrap`.

## Creating visualizations

Run after experiments are complete:

```
python main.py create-all-visuals
```

This reads from `results/metrics/metrics/` and `results/resource_profiles/` and writes SVG figures to `results/figures/`. Only synth mode results at synth_scale=100 are included in the plots.

To use internal validation bootstrap data instead (see below), call `load_internal_validation_bootstrap()` directly from `src/visualization.py` and pass the resulting DataFrame to `generate_all_figures()`, or swap the loader inside that function.

## Bootstrap types

Two bootstrap approaches are implemented in `src/visualization.py`:

**Simple bootstrap** (`load_simple_bootstrap`, default) — the model is fit once on the full training set; the test set is resampled with replacement many times and evaluated on each resample. Each metrics JSON in `results/metrics/metrics/` stores the resulting distribution as a list of per-iteration values. Resource profiles are single scalar measurements. This is what `create-all-visuals` uses.

**Why resource metrics have no bootstrap distribution** — Bootstrap resamples prediction arrays, not inputs, so inference runs only once. Producing a resource distribution would require re-running inference for every resample (N× the inference cost), which is cost-prohibitive for resource-intensive models such as FHE. Prediction metrics are cheap to recompute from existing predictions; resource metrics are not.

**Internal validation bootstrap** (`load_internal_validation_bootstrap`) — the full training pipeline (including synthesis for synth modes) is re-run from scratch for each seed, each in its own subfolder under `results/internal_validation_bootstrap/`. Results must first be aggregated into a single JSON via `aggregate-internal-validation-bootstrap` before visualization. Each seed reflects a different resampled training set, so the distribution captures training-time variance rather than test-set variance.

## Significance testing

```
python main.py paired-bootstrap-tests --modes standard '*_100' --reference standard --side greater  # each generator vs Real
python main.py paired-bootstrap-tests --modes 'arf_*' --reference arf_100 --side less               # arf: larger scale vs 100%
python main.py paired-bootstrap-tests --modes 'fhe_*' --comparison adjacent --side less              # dose-response
```

Runs a **paired bootstrap test** on the differences between two modes' metric distributions within each (dataset, classifier) cell, with Holm-Bonferroni correction applied per cell. Writes `results/stats/paired_bootstrap__{metric}__{comparison}[__{test_type}][__{side}].csv` (add `--format both` for a companion markdown table). Any of the five metrics works via `--metric`; the default is `roc_auc`.

### Contrast design

`--comparison` sets which contrasts are built, and therefore the Holm family size. This is not cosmetic: correcting over the wrong family changes every adjusted p-value, and it cannot be fixed by filtering rows afterwards.

| design | contrasts | use for |
|---|---|---|
| `pairwise` (default) | C(k,2) | no mode is privileged |
| `reference` | k−1 | prespecified questions: each synthesis scale vs 100%, each generator vs Real, each bit width vs Real |
| `adjacent` | k−1 | dose-response along a sweep, e.g. fhe_2 vs fhe_4 vs fhe_6 |

Passing `--reference MODE` implies `--comparison reference`.

### Difference, non-inferiority, equivalence

`--test-type` selects which p-value is the primary Holm-corrected one; all three are always reported raw in `p_diff`, `p_noninf`, `p_equiv`.

- `difference` (default) tests against no difference. Answers "are these distinguishable".
- `noninferiority` tests H0: mode_a is worse than mode_b by at least `--margin`. A small p licenses "mode_a is non-inferior within the margin".
- `equivalence` is TOST against ±`--margin`, taking the larger of the two one-sided p-values.

`--side` sets the sidedness of the **difference** test. `two-sided` (default) tests H0 no difference. `greater` (H1: mode_a > mode_b) and `less` (H1: mode_a < mode_b) request the one-sided ASL in a prespecified direction, reported in `p_one_sided` and, for a one-sided run, used as the primary Holm-corrected p. It is exactly `p_diff / 2` in the hypothesized direction and the dual of a one-sided 90% CI. Direction is relative to `mode_a`, which the reference design pins to the reference: a vs-Real contrast (reference `standard`) uses `greater` to mean "the method is worse than Real". The noninferiority and equivalence tests are already one-sided/TOST and ignore `--side`. The study's prespecified analyses are one-sided (see below).

The default margin is 0.05 ROC-AUC, a deliberately lenient screening margin. **State the margin wherever these results are reported**, since the conclusion is meaningless without it, and consider a sensitivity check at 0.01 and 0.02. Note also that the reported CI is 95%, whereas TOST at α=0.05 formally corresponds to a 90% interval, so reading equivalence off the printed CI is conservative relative to `p_equiv`. Prefer the p-value column.

**Why the stored results are already paired.** Every mode (standard, synthetic, FHE) is evaluated on the same real held-out test set for a given dataset, and every one calls `run_bootstrap(..., seed=42)`, which draws its resample indices from a freshly seeded `np.random.default_rng(42)`. Since the number of test rows is a property of the dataset and not of the mode, replicate *i* of any two modes was computed on the identical resampled rows. Replicate-wise differences `d_i = A_i - B_i` are therefore genuine paired differences, and no re-running of inference is needed. `src/stats_tests.py` verifies this at load time rather than assuming it: a metric array whose length disagrees with the file's declared `n_bootstrap`, or a pair whose two arrays differ in length, is dropped with an error instead of silently producing an unpaired comparison.

**The p-value.** With `r = min(#{d>0}, #{d<0}) + #{d==0}` over B usable replicates, `p = min(1, 2(r+1)/(B+1))`. This is the achieved significance level read off the difference distribution, which is the dual of the percentile confidence interval, so `p < 0.05` holds exactly when the reported 95% CI of the difference excludes zero. That keeps the p-value and CI columns of a row consistent with each other and with the percentile convention already used in `aggregate-metrics-csv`. The `(r+1)/(B+1)` form floors p at about 0.002 rather than letting it reach a false zero when no replicate lands on the far side. For a one-sided run (`--side greater|less`), `p_one_sided = (wrong-side count + tie term + 1)/(B+1)`, exactly half of `p_diff` in the hypothesized direction and the dual of a one-sided 90% CI; it becomes the primary column while `p_diff` is still reported. The one-sided floor is about 0.001.

**Ties.** Metrics are rounded to 4 decimals before storage, and ROC-AUC is intrinsically discrete on test sets of 57 to 203 rows, so exact ties (`d_i == 0`) are common and are sometimes total. `--tie-rule` controls how they count:

| rule | `r` | note |
|---|---|---|
| `split` (default) | `min + ties/2` | standard sign-test treatment, unbiased about direction |
| `conservative` | `min + ties` | quantization can never manufacture significance, but see the ceiling below |
| `exclude` | `min`, with B reduced | highest power; a tie-heavy comparison then rests on few replicates |

The `conservative` rule has a structural ceiling that is easy to miss: significance at α=0.05 needs `r ≤ 24` at B=1000, so **any comparison with more than 24 ties can never be called significant regardless of effect size**. That is correct behaviour when ties mean genuinely identical predictions (high-bit FHE reproducing plaintext) but wrong when they are rounding artifacts, which is why `split` is the default.

`split` removes the hard cap but does not make tie-heavy cells easy to call: a cell with 485 ties and the other 515 replicates unanimously in one direction still lands near p ≈ 0.49. Only `exclude` reports such a cell as significant. Always read `n_ties` alongside the p-value, and treat a large `n_ties` as a signal that the two modes are near-identical rather than as a power problem to be flagged away.

Under every rule, two modes with identical predictions give p = 1.0.

**Choosing modes matters.** The Holm family is every contrast inside one (dataset, classifier) cell. Under the default pairwise design, selecting all 32 modes means 496 tests per cell, and against the ~0.002 p-value floor almost nothing can survive correction. That is a real property of the design, not a defect, which is why `--modes`, `--datasets` and `--models` accept both exact names and glob patterns, and why prespecified questions should use `--reference`: narrow the comparison to the question being asked.

### The prespecified analyses

All analyses use ROC-AUC and a **one-sided** difference test, since each contrast has a logical direction. Scope is the four manuscript datasets; `pregnancy_outcome` is present in `results/` but excluded from the paper (drop `--datasets` to include it). `--side` is read relative to `mode_a`, which the reference design pins to the reference, so a vs-Real contrast (reference `standard`) uses `--side greater` to mean "the method is worse than Real".

```
DATASETS="diabetes heart_disease maternal_health_risk pregnancy_outcome"

# A1  Primary, synthetic leg: each generator at scale 100 vs Real.
#     mode_a = standard, so "generator worse than Real" is standard > generator,
#     i.e. --side greater. Non-significant = valid replacement. 5 contrasts per cell.
python main.py paired-bootstrap-tests --modes standard '*_100' --reference standard \
  --side greater --datasets $DATASETS \
  --output results/stats/a1_synth_vs_real.csv

# A2  Primary, FHE leg (also RQ2 saturation): each bit width vs Real.
#     Same direction as A1: --side greater. Saturation = smallest n_bits that is
#     not significantly worse than Real, read per model. 6 contrasts per cell.
python main.py paired-bootstrap-tests --modes standard 'fhe_*' --reference standard \
  --side greater --datasets $DATASETS \
  --output results/stats/a2_fhe_vs_real.csv
# A2 secondary, dose-response: adjacent bit widths, "more bits better".
#     mode_a = the lower bit width, so mode_b > mode_a is --side less. 5 contrasts.
python main.py paired-bootstrap-tests --modes 'fhe_*' --comparison adjacent \
  --side less --datasets $DATASETS \
  --output results/stats/a2_fhe_adjacent.csv

# A3  RQ1, synthesis scale: 150/200/250/300 vs 100 within each generator.
#     mode_a = <gen>_100, so "larger scale better" is mode_b > mode_a, i.e.
#     --side less. 4 contrasts per cell, one run per generator.
for g in arf bayesian_network ctgan gaussian_copula nflow; do
  python main.py paired-bootstrap-tests --modes "${g}_*" --reference "${g}_100" \
    --side less --datasets $DATASETS \
    --output "results/stats/a3_scale__${g}.csv"
done
```

A1's `'*_100'` glob resolves to the five generators at synthesis scale 100 and excludes `standard` and the FHE modes, since neither carries a `_100` suffix; `standard` is added back explicitly as the reference. **RQ3** (where system-level costs concentrate) is descriptive resource profiling, not a significance test, and generator ranking ("which generator loses least") is read descriptively from the A1 mean gaps and their CIs rather than as a separate test family.

The each-vs-Real logic resolves the primary question directly: within a cell, a generator (A1) or FHE precision (A2) is a valid replacement for real data when it is **not** significantly worse than Real; where a synthetic option qualifies it is preferred as the cheaper, faster one-time cost (RQ3), and FHE is warranted only where the best synthetic option is significantly worse than Real. The earlier non-inferiority/equivalence margin and the direct synth-vs-FHE equivalence test were dropped on 2026-07-23; the one-sided difference test answers replacement without a margin to defend.

No standardized effect size (Cohen's d or similar) is reported. The 1000 values are bootstrap replicates of a statistic rather than independent observations, so their spread shrinks with test-set size and any sd-standardized quantity would describe `n_test` more than it describes the effect. The mean difference with its percentile CI, alongside both group means, is reported instead.

## TD
### visualizations
- mode combination legend
- log scale
- color consistency
- FHE-specific resource plots
- synthesis-specific fitting plots
- radar plots
- dynamic pareto?
- % synth_scale line graph

### synthesizers
- reuse synths
- % synth_scale (e.g. 100 = same as original)