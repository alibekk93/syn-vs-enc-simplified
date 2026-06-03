# syn-vs-enc-simplified

## TD
- bootstrap: `results\bootstrap\`, `config\bootstrap.yaml`
- visualizations: color consistency; multi-panel figures; radar plots; dynamic pareto?

## Abstract draft
The increasing use of machine learning in healthcare has amplified concerns around the protection of sensitive patient information, particularly protected health information (PHI). Privacy-preserving approaches such as synthetic data generation and secure computation have emerged as promising solutions, yet their relative trade-offs in predictive performance and computational efficiency remain insufficiently characterized. In this work, we present a unified experimental framework for systematically comparing these approaches on tabular healthcare datasets.

We focus on two primary privacy strategies: (1) synthetic data generation, where models are trained on artificially generated datasets that approximate the statistical properties of the original data, and (2) fully homomorphic encryption (FHE), where model inference is performed directly on encrypted data using the Concrete ML library. Our framework supports multiple downstream classifiers and is designed to be extensible, with additional synthetic data generation methods incorporated beyond baseline Gaussian copula models. Experiments are conducted exclusively in the tabular setting, reflecting the dominant structure of clinical datasets, and leveraging FHE-compatible models supported by Concrete ML.

To enable rigorous comparison, we evaluate model performance across real, synthetic, and encrypted settings using standard classification metrics. We further employ bootstrap-based statistical analysis to quantify uncertainty and assess the significance of observed performance differences across methods. In addition to predictive accuracy, we perform detailed system-level profiling, capturing runtime, memory consumption, storage overhead, and FHE-specific characteristics such as circuit complexity.

Our results highlight the trade-offs between data utility, privacy guarantees, and computational cost across approaches. Synthetic data methods provide a flexible and scalable alternative with reduced privacy risk but may incur degradation in predictive performance depending on data fidelity. FHE-based inference preserves data confidentiality without requiring data transformation but introduces significant computational overhead. This work provides a reproducible benchmark for evaluating privacy-preserving machine learning techniques in healthcare and offers practical guidance for selecting appropriate methods under real-world constraints.

