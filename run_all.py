from __future__ import annotations

from blue_bond_simulation.code.run_blue_bond import main as run_blue_bond
from france_stress_test.code.run_france_stress import run as run_france_stress
from transferability.code.run_transferability import run as run_transferability


def main() -> None:
    run_blue_bond()
    run_france_stress()
    run_transferability()
    print("All simulation modules completed.")


if __name__ == "__main__":
    main()
