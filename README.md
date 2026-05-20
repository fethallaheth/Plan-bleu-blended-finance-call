# Blended Finance Simulation

Minimal Monte Carlo models for blue-bond structure, France stress testing, and transferability screening.

## Modules

- `blue_bond_simulation/` base governance and mobilization model
- `france_stress_test/` France pathway and scenario stress pack
- `transferability/` cross-country readiness and transfer model

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python blue_bond_simulation/main.py
python france_stress_test/code/run_france_stress.py
python transferability/code/run_transferability.py
```

## Outputs

- `blue_bond_simulation/outputs/`
- `france_stress_test/output/`
- `transferability/output/`
