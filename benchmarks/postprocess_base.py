#!/usr/bin/env python3
# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
# All Rights reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

import pandas as pd
import fileinput
import pprint

# Read all input files specified on the command line, or stdin and parse
# the content, storing it as a pandas dataframe


def read_logs(files=None):
    """Read all input files and return pandas DataFrame"""
    data_default = dict(
        file='unknown',
        backend='unknown',
        backend_memtype='unknown',
        hostname='unknown',
        test='unknown',
        num_procs=0,
        num_procs_node=0,
        degree=0,
        quadrature_pts=0,
        code='libCEED',
    )
    data = data_default.copy()

    runs = []
    for line in fileinput.input(files):
        # Legacy header contains number of MPI tasks
        if 'Running the tests using a total of' in line:
            data = data_default.copy()
            data['num_procs'] = int(
                line.split(
                    'a total of ',
                    1)[1].split(
                    None,
                    1)[0])
        # MPI tasks per node
        elif 'tasks per node' in line:
            data['num_procs_node'] = int(
                line.split(
                    ' tasks per',
                    1)[0].rsplit(
                    None,
                    1)[1])
        # New Benchmark Problem
        elif "CEED Benchmark Problem" in line:
            # Starting a new block
            data = data.copy()
            runs.append(data)
            data['file'] = fileinput.filename()
            data['test'] = line.split()[-2] + " " + line.split('-- ')[1]
            data['bp'] = data['test'].rsplit()[-1]
            data['case'] = 'scalar' if (('Problem 1' in line) or ('Problem 3' in line)
                                        or ('Problem 5' in line)) else 'vector'
        elif "Hostname" in line:
            data['hostname'] = line.split(':')[1].strip()
        elif "Total ranks" in line:
            data['num_procs'] = int(line.split(':')[1].strip())
        elif "Ranks per compute node" in line:
            data['num_procs_node'] = int(line.split(':')[1].strip())
        # Backend
        elif 'libCEED Backend MemType' in line:
            data['backend_memtype'] = line.split(':')[1].strip()
        elif 'libCEED Backend' in line:
            data['backend'] = line.split(':')[1].strip()
        # P
        elif 'Basis Nodes' in line:
            data['degree'] = int(line.split(':')[1]) - 1
        # Q
        elif 'Quadrature Points' in line:
            data['quadrature_pts'] = int(line.split(':')[1])
        # Total DOFs
        elif 'Global nodes' in line:
            data['num_unknowns'] = int(line.split(':')[1])
            if data['case'] == 'vector':
                data['num_unknowns'] *= 3
        elif 'Global DOFs' in line:  # Legacy
            data['num_unknowns'] = int(line.split(':')[1])
        # Number of elements
        elif 'Local Elements' in line:
            data['num_elem'] = int(
                line.split(':')[1].split()[0]) * data['num_procs']
        elif 'DoF per node' in line:
            data['dof_per_node'] = int(line.split(':')[1])
        # CG Solve Time
        elif 'Total KSP Iterations' in line:
            data['ksp_its'] = int(line.split(':')[1].split()[0])
        elif 'CG Solve Time' in line:
            data['time_per_it'] = float(
                line.split(':')[1].split()[0]) / data['ksp_its']
        # CG DOFs/Sec
        elif 'DoFs/Sec in CG' in line or 'DOFs/Sec in CG' in line:
            data['cg_iteration_dps'] = 1e6 * \
                float(line.split(':')[1].split()[0])
        # End of output

    return pd.DataFrame(runs)


if __name__ == "__main__":
    runs = read_logs()
    print(runs)         # Print summary (about 10 lines)
    print('First entry:\n', runs.iloc[0])
    print('Last entry:\n', runs.iloc[-1])
