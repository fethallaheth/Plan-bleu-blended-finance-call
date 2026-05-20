# Blended Finance Monte Carlo Simulation

This repository contains a Monte Carlo simulation toolkit for blended blue-bond governance analysis, with extensions for:

- France institutional stress testing under adverse scenarios
- Mediterranean country transferability readiness analysis

## Project Structure

- `blue_bond_simulation/`: Base comparative model (traditional governance vs blockchain-enabled governance)
- `france_stress_test/`: France institutional stress-test extension
- `transferability/`: Country transferability extension
- `run_all_extensions.py`: Runs both extension modules end-to-end

## Setup

```bash
pip install -r requirements.txt
```

## Run the Base Model

```bash
python blue_bond_simulation/main.py
```

## Run Extension Modules

```bash
python run_all_extensions.py
```

## Output Folders

Generated files are written to module-specific output directories:

- `blue_bond_simulation/outputs/`
- `france_stress_test/output/`
- `transferability/output/`

These generated outputs are ignored in Git by default for a cleaner source repository.

