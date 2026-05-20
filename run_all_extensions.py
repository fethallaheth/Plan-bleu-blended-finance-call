from __future__ import annotations

from france_stress_test.code.run_france_stress import run as run_france_stress
from transferability.code.run_transferability import run as run_transferability


def main() -> None:
    run_france_stress()
    run_transferability()
    print("All extension modules completed.")


if __name__ == "__main__":
    main()

