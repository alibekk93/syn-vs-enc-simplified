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
python main.py create-visuals
```

This reads from `results/metrics/metrics/` and `results/resource_profiles/` and writes SVG figures to `results/figures/`. Only synth mode results at synth_scale=100 are included in the plots.

To use internal validation bootstrap data instead (see below), call `load_internal_validation_bootstrap()` directly from `src/visualization.py` and pass the resulting DataFrame to `generate_all_figures()`, or swap the loader inside that function.

## Bootstrap types

Two bootstrap approaches are implemented in `src/visualization.py`:

**Simple bootstrap** (`load_simple_bootstrap`, default) — the model is fit once on the full training set; the test set is resampled with replacement many times and evaluated on each resample. Each metrics JSON in `results/metrics/metrics/` stores the resulting distribution as a list of per-iteration values. Resource profiles are single scalar measurements. This is what `create-visuals` uses.

**Why resource metrics have no bootstrap distribution** — Bootstrap resamples prediction arrays, not inputs, so inference runs only once. Producing a resource distribution would require re-running inference for every resample (N× the inference cost), which is cost-prohibitive for resource-intensive models such as FHE. Prediction metrics are cheap to recompute from existing predictions; resource metrics are not.

**Internal validation bootstrap** (`load_internal_validation_bootstrap`) — the full training pipeline (including synthesis for synth modes) is re-run from scratch for each seed, each in its own subfolder under `results/internal_validation_bootstrap/`. Results must first be aggregated into a single JSON via `aggregate-internal-validation-bootstrap` before visualization. Each seed reflects a different resampled training set, so the distribution captures training-time variance rather than test-set variance.

## Significance testing

```
python main.py paired-bootstrap-tests --metric roc_auc --modes standard 'fhe_*'
python main.py paired-bootstrap-tests --metric roc_auc --modes standard 'fhe_*' '*_100' --format both
python main.py paired-bootstrap-tests --datasets diabetes --models xgboost --modes standard fhe_8
```

Runs a **paired bootstrap test** on the differences between two modes' metric distributions, all pairwise within each (dataset, classifier) cell, with Holm-Bonferroni correction applied per cell. Writes `results/stats/paired_bootstrap__{metric}.csv` (add `--format both` for a companion markdown table). Any of the five metrics works via `--metric`; the default is `roc_auc`.

**Why the stored results are already paired.** Every mode (standard, synthetic, FHE) is evaluated on the same real held-out test set for a given dataset, and every one calls `run_bootstrap(..., seed=42)`, which draws its resample indices from a freshly seeded `np.random.default_rng(42)`. Since the number of test rows is a property of the dataset and not of the mode, replicate *i* of any two modes was computed on the identical resampled rows. Replicate-wise differences `d_i = A_i - B_i` are therefore genuine paired differences, and no re-running of inference is needed. `src/stats_tests.py` verifies this at load time rather than assuming it: a metric array whose length disagrees with the file's declared `n_bootstrap`, or a pair whose two arrays differ in length, is dropped with an error instead of silently producing an unpaired comparison.

**The p-value.** With `r = min(#{d>0}, #{d<0}) + #{d==0}` over B usable replicates, `p = min(1, 2(r+1)/(B+1))`. This is the achieved significance level read off the difference distribution, which is the dual of the percentile confidence interval, so `p < 0.05` holds exactly when the reported 95% CI of the difference excludes zero. That keeps the p-value and CI columns of a row consistent with each other and with the percentile convention already used in `aggregate-metrics-csv`. The `(r+1)/(B+1)` form floors p at about 0.002 rather than letting it reach a false zero when no replicate lands on the far side.

**Ties.** Metrics are rounded to 4 decimals before storage, and ROC-AUC is intrinsically discrete on test sets of 57 to 203 rows, so exact ties (`d_i == 0`) are common. Ties are charged to the smaller of the two directional counts, which can only raise p. Quantization can therefore weaken a result but never manufacture significance. Two modes producing identical predictions fall out of the same formula as p = 1.0 with no special case. The `n_ties` column is reported so a null result driven by quantization is visible rather than hidden.

**Choosing modes matters.** The Holm family is every pairwise test inside one (dataset, classifier) cell. Selecting all 32 modes means 496 tests per cell, and against the ~0.002 p-value floor almost nothing can survive correction. That is a real property of the design, not a defect, which is why `--modes`, `--datasets` and `--models` accept both exact names and glob patterns: narrow the comparison to the question being asked.

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