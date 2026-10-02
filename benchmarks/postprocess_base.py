#!/usr/bin/env python3

# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

import fileinput
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd


def read_logs(files: Optional[Union[str, Sequence[str]]] = None) -> pd.DataFrame:
    """
    Read CEED benchmark log files or stdin and parse records into a pandas DataFrame.

    Args:
        files: Optional filename or sequence of filenames. Reads from sys.argv/stdin if None.
    """
    data_default: Dict[str, Any] = {
        "executable": "unknown",
        "backend": "unknown",
        "backend_memtype": "unknown",
        "hostname": "unknown",
        "test": "unknown",
        "num_procs": 0,
        "num_procs_node": 0,
        "degree": 0,
        "quadrature_pts": 0,
        "code": "libCEED",
    }
    data = data_default.copy()
    runs: List[Dict[str, Any]] = []

    for line in fileinput.input(files):
        # Legacy header with total MPI tasks
        if "Running the tests using a total of" in line:
            data = data_default.copy()
            data["num_procs"] = int(line.split("a total of ", 1)[1].split(None, 1)[0])
        elif "Reading test file" in line:
            data_default["executable"] = line.split(": ")[1].rsplit("/", 1)[1].rsplit(".", 1)[0]
            data["executable"] = data_default["executable"]
        elif "tasks per node" in line:
            data["num_procs_node"] = int(line.split(" tasks per", 1)[0].rsplit(None, 1)[1])
        elif "CEED Benchmark Problem" in line:
            data = data.copy()
            runs.append(data)
            data["test"] = line.split()[-2] + " " + line.split("-- ")[1].strip()
            data["bp"] = data["test"].rsplit()[-1]
            data["case"] = (
                "scalar"
                if any(f"Problem {k}" in line for k in (1, 3, 5))
                else "vector"
            )
        elif "Hostname" in line:
            data["hostname"] = line.split(":", 1)[1].strip()
        elif "Total ranks" in line:
            data["num_procs"] = int(line.split(":", 1)[1].strip())
        elif "Ranks per compute node" in line:
            data["num_procs_node"] = int(line.split(":", 1)[1].strip())
        elif "libCEED Backend MemType" in line:
            data["backend_memtype"] = line.split(":", 1)[1].strip()
        elif "libCEED Backend" in line:
            data["backend"] = line.split(":", 1)[1].strip()
        elif "Solution Order (P)" in line:
            data["degree"] = int(line.split(":", 1)[1]) - 1
        elif "Quadrature Order (Q)" in line:
            data["quadrature_pts"] = int(line.split(":", 1)[1])
        elif "Global nodes" in line:
            data["num_unknowns"] = int(line.split(":", 1)[1])
            if data.get("case") == "vector":
                data["num_unknowns"] *= 3
        elif "Global DOFs" in line:
            data["num_unknowns"] = int(line.split(":", 1)[1])
        elif "Local Elements" in line:
            data["num_elem"] = int(line.split(":", 1)[1].split()[0]) * data["num_procs"]
        elif "DoF per node" in line:
            data["dof_per_node"] = int(line.split(":", 1)[1])
        elif "Total KSP Iterations" in line:
            data["ksp_its"] = int(line.split(":", 1)[1].split()[0])
        elif "CG Solve Time" in line:
            ksp_its = data.get("ksp_its", 1)
            data["time_per_it"] = float(line.split(":", 1)[1].split()[0]) / (ksp_its if ksp_its > 0 else 1)
        elif "DoFs/Sec in CG" in line or "DOFs/Sec in CG" in line:
            data["cg_iteration_dps"] = 1e6 * float(line.split(":", 1)[1].split()[0])

    return pd.DataFrame(runs)


if __name__ == "__main__":
    from rich.console import Console
    console = Console()
    df = read_logs()
    console.print(f"[bold cyan]Parsed {len(df)} total runs.[/bold cyan]")
    if not df.empty:
        console.print(df.head())
