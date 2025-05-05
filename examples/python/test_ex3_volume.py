#!/usr/bin/env python3

import subprocess
import itertools


def run_test(dim, mesh_degree, sol_degree, num_qpts):
    """Run ex3-volume.py with given parameters."""
    cmd = [
        "python", "ex3-volume.py",
        "-d", str(dim),
        "-m", str(mesh_degree),
        "-p", str(sol_degree),
        "-q", str(num_qpts)
    ]
    print("\n" + "=" * 80)
    print(f"Testing: dim={dim}, mesh_degree={mesh_degree}, sol_degree={sol_degree}, num_qpts={num_qpts}")
    print("=" * 80)
    subprocess.run(cmd)


def main():
    # Test parameters
    dimensions = [1, 2, 3]
    mesh_degrees = [2, 3, 4]
    sol_degrees = [2, 3, 4]
    num_qpts = [4, 6, 8]

    # Run tests for all combinations
    for dim, m_deg, s_deg, qpts in itertools.product(dimensions, mesh_degrees, sol_degrees, num_qpts):
        # Ensure number of quadrature points is sufficient
        if qpts > max(m_deg, s_deg):
            run_test(dim, m_deg, s_deg, qpts)


if __name__ == "__main__":
    main()
