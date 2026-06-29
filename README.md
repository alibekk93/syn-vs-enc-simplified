# syn-vs-enc-simplified

## Usage

```
python main.py run-experiment --config config/main.yaml
python main.py run-experiment --config config/experiments/DRAC-fhe.yaml --n-bits 4
python main.py list-n-bits                                  # expand config/fhe.yaml's sweep -> fhe_n_bits.txt
python main.py run-single-bootstrap --config config/main.yaml --seed 42
python main.py generate-seeds --length 1000                 # -> bootstrap_seeds.txt
python main.py aggregate-bootstrap
```

To parallelize the FHE n_bits sweep, run `list-n-bits` then launch one
`run-experiment --n-bits N` process per value (locally, or as separate
cluster jobs) — each is independent and writes uniquely-named output files,
mirroring how bootstrap parallelizes across seeds via `run-single-bootstrap`.

## TD
### visualizations
- mode combination legend
- log scale
- color consistency
- FHE-specific resource plots
- synthesis-specific fitting plots
- radar plots
- dynamic pareto?
- % oversampling line graph

### synthesizers
- % oversampling (e.g. 100 is same as original)

### bugs
- check logs
- job for azure

### methodology
- bootstrap pre inference: identify a name and research past examples, draft justificationgit a
- resource metrics
- MIMIC dataset?
- multi-class classification?
- hyperparameter tuning?
- bootstrap sed fix end (31)
- mlp

## Abstract draft
The increasing use of machine learning in healthcare has amplified concerns around the protection of sensitive patient information, particularly protected health information (PHI). Privacy-preserving approaches such as synthetic data generation and secure computation have emerged as promising solutions, yet their relative trade-offs in predictive performance and computational efficiency remain insufficiently characterized. In this work, we present a unified experimental framework for systematically comparing these approaches on tabular healthcare datasets.

We focus on two primary privacy strategies: (1) synthetic data generation, where models are trained on artificially generated datasets that approximate the statistical properties of the original data, and (2) fully homomorphic encryption (FHE), where model inference is performed directly on encrypted data using the Concrete ML library. Our framework supports multiple downstream classifiers and is designed to be extensible, with additional synthetic data generation methods incorporated beyond baseline Gaussian copula models. Experiments are conducted exclusively in the tabular setting, reflecting the dominant structure of clinical datasets, and leveraging FHE-compatible models supported by Concrete ML.

To enable rigorous comparison, we evaluate model performance across real, synthetic, and encrypted settings using standard classification metrics. We further employ bootstrap-based statistical analysis to quantify uncertainty and assess the significance of observed performance differences across methods. In addition to predictive accuracy, we perform detailed system-level profiling, capturing runtime, memory consumption, storage overhead, and FHE-specific characteristics such as circuit complexity.

Our results highlight the trade-offs between data utility, privacy guarantees, and computational cost across approaches. Synthetic data methods provide a flexible and scalable alternative with reduced privacy risk but may incur degradation in predictive performance depending on data fidelity. FHE-based inference preserves data confidentiality without requiring data transformation but introduces significant computational overhead. This work provides a reproducible benchmark for evaluating privacy-preserving machine learning techniques in healthcare and offers practical guidance for selecting appropriate methods under real-world constraints.

