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

## Abstract draft
The increasing use of machine learning in healthcare has amplified concerns around the protection of sensitive patient information, particularly protected health information (PHI). Privacy-preserving approaches such as synthetic data generation and secure computation have emerged as promising solutions, yet their relative trade-offs in predictive performance and computational efficiency remain insufficiently characterized. In this work, we present a unified experimental framework for systematically comparing these approaches on tabular healthcare datasets.

We focus on two primary privacy strategies: (1) synthetic data generation, where models are trained on artificially generated datasets that approximate the statistical properties of the original data, and (2) fully homomorphic encryption (FHE), where model inference is performed directly on encrypted data using the Concrete ML library. Our framework evaluates three downstream classifiers (logistic regression, random forest, and XGBoost) against five synthetic data generation methods (Gaussian copula, Bayesian network, CTGAN, adversarial random forests, and normalizing flows) and FHE-based inference, across four tabular healthcare datasets (breast cancer, diabetes, heart disease, and maternal health risk). Synthetic data methods are additionally evaluated at five oversampling scales (100-300% of the original training set size), and FHE is evaluated across a sweep of quantization precisions (2-12 bits).

To enable rigorous comparison, we evaluate model performance across real, synthetic, and encrypted settings using standard classification metrics with bootstrap-based uncertainty quantification. In addition to predictive accuracy, we perform detailed system-level profiling, capturing runtime, memory consumption, storage overhead, and FHE-specific characteristics such as circuit complexity.

Our results show that synthetic data fidelity, rather than the volume of generated data, is the primary driver of downstream utility: across all five generators, increasing the oversampling scale from 100% to 300% yielded negligible accuracy gains, while fidelity varied substantially by method. Adversarial random forests most closely preserved real-data performance (within roughly 1.3 accuracy points on average, and consistently the closest of any synthetic method on every dataset), followed by normalizing flows, while CTGAN underperformed on both predictive accuracy and computational cost, taking 26-40x longer to fit than the other generators while producing the least useful data. FHE-based inference matched real-data accuracy for logistic regression and random forest once quantization reached 6-8 bits, but this came at a cost of three to four orders of magnitude longer inference time and up to roughly 5x peak memory relative to standard inference, with both cost and circuit complexity scaling steeply with bit-width; XGBoost under FHE did not recover baseline accuracy at any tested bit-width. These findings indicate that adversarial random forests offer the most favorable utility-cost trade-off among the synthetic data methods evaluated, while FHE can deliver near-lossless accuracy for compatible model types at substantial and precisely quantifiable computational expense, concentrated most heavily in tree-ensemble models. This work provides a reproducible benchmark for evaluating privacy-preserving machine learning techniques in healthcare and offers practical guidance for selecting appropriate methods under real-world constraints.

