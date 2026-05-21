# Blended Finance Monte Carlo Suite

Monte Carlo studies for blended blue-bond governance, France stress testing, and cross-country transferability.

## Repository Layout

- `blue_bond_simulation/` base governance and mobilization model
- `france_stress_test/` France scenario and pathway stress pack
- `transferability/` regional readiness and transfer model

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m run_all
python -m blue_bond_simulation.code.run_blue_bond
python -m france_stress_test.code.run_france_stress
python -m transferability.code.run_transferability
```

Compatibility entrypoints remain available:

```bash
python blue_bond_simulation/main.py
python run_all.py
python run_all_extensions.py
```

## Generated Outputs

- `blue_bond_simulation/outputs/`
- `france_stress_test/output/data/`
- `france_stress_test/output/figures/`
- `france_stress_test/output/tables/`
- `transferability/output/data/`
- `transferability/output/figures/`
- `transferability/output/tables/`
