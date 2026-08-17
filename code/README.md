# ACES-FL Source Code

This folder contains the reconstructed Python source code for the MSc ACES-FL implementation.

## Experiment folders

- `acesfl-baseline/` - FedAvg baseline using all 20 simulated clients.
- `acesfl-selection/` - ACES-FL resource-aware client-selection experiment (Selection V1).
- `acesfl-selection-v2/` - fairness-aware Selection V2.
- `acesfl-compression/` - adaptive FP32/FP16/INT8 model-delta compression.
- `acesfl-security/` - controlled model-poisoning experiment without the final defense.
- `acesfl-defense/` - poisoning experiment with robust norm, cosine-direction and recoverable-trust screening.
- `generate_thesis_results.py` - generates thesis tables/figures from archived experiment CSV outputs.

Each experiment preserves the package and script names shown in the uploaded ACES-FL implementation source-code PDF.

## Important reproducibility note

The source-code PDF contains the Python implementation files, but it does not contain the Flower `pyproject.toml`/run-configuration files or the raw UNSW-NB15 dataset. Those files have therefore **not been invented or reconstructed here**.

The repository's existing `data/README.md` should be used for dataset documentation. Run configuration can be added later from the original project environment or dissertation evidence.

## Validation

All Python files in this curated package were reconstructed from the PDF text and passed Python syntax compilation (`compile()` / AST parsing) after repair of PDF visual line wrapping.

This validation checks syntax only; full end-to-end execution requires the original Flower run configuration, dataset files and dependencies.
