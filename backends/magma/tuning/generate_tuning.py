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

import argparse
import os
import io
import re
import subprocess
import pandas as pd
import time

script_dir = os.path.dirname(os.path.realpath(__file__))


def build(nb, build_cmd):
    with open(f"{script_dir}/../ceed-magma-gemm-selector.cpp", 'r') as f:
        data = f.read()
        data = re.sub(
            '.*(#define CEED_AUTOTUNE_RTC_NB).*',
            r'\1' + f" {nb}",
            data)
    with open(f"{script_dir}/../ceed-magma-gemm-selector.cpp", 'w') as f:
        f.write(data)
    subprocess.run(build_cmd, cwd=f"{script_dir}/../../..")
    subprocess.run(["make", "tuning"], cwd=f"{script_dir}")


def benchmark(backend):
    data = subprocess.run(["./tuning", f"{backend}"], capture_output=True)
    return pd.read_csv(io.StringIO(data.stdout.decode('utf-8')), header=None,
                       delim_whitespace=True, names=['P', 'N', 'Q', 'Q_COMP', 'TRANS', 'MFLOPS'])


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser("MAGMA RTC autotuning")
    parser.add_argument(
        "-arch",
        help="Device architecture name for tuning data",
        required=True)
    parser.add_argument(
        "-max-nb",
        help="Maximum block size NB to consider for autotuning",
        default=32,
        type=int)
    parser.add_argument(
        "-ceed",
        help="Ceed resource specifier",
        default="/cpu/self")
    parser.add_argument(
        "-build-cmd",
        help="Command used to build libCEED from the source root directory",
        default="make")
    args = parser.parse_args()

    for nb in range(1, args.max_nb + 1):
        # Rebuild the code for the given value of NB
        build(nb, args.build_cmd)

        # Run the benchmarks
        start = time.perf_counter()
        data_nb = benchmark(args.ceed)
        print(
            f"Finished benchmarks for NB = {nb}, backend = {args.ceed} ({time.perf_counter() - start} s)")

        # Save the data for the highest performing NB
        if nb == 1:
            data = pd.DataFrame(data_nb)
            data['NB'] = nb
        else:
            idx = data_nb['MFLOPS'] > data['MFLOPS']
            data.loc[idx, 'NB'] = nb
            data.loc[idx, 'MFLOPS'] = data_nb.loc[idx, 'MFLOPS']

    # Print the results
    with open(f"{script_dir}/{args.arch}_rtc.h", 'w') as f:
        f.write(
            "////////////////////////////////////////////////////////////////////////////////\n")
        f.write(f"// auto-generated from data on {args.arch}\n\n")

        rows = data.loc[data['TRANS'] == 1].to_string(header=False, index=False, columns=[
                                                      'P', 'N', 'Q', 'Q_COMP', 'NB']).split('\n')
        f.write(
            "////////////////////////////////////////////////////////////////////////////////\n")
        f.write(
            f"std::vector<std::array<int, RECORD_LENGTH_RTC> > drtc_t_{args.arch}" +
            " = {\n")
        count = 0
        for row in rows:
            f.write("    {" + re.sub(r'(\s+)', r',\1', row) +
                    ("},\n" if count < len(rows) - 1 else "}\n"))
            count += 1
        f.write("};\n\n")

        rows = data.loc[data['TRANS'] == 0].to_string(header=False, index=False, columns=[
                                                      'P', 'N', 'Q', 'Q_COMP', 'NB']).split('\n')
        f.write(
            "////////////////////////////////////////////////////////////////////////////////\n")
        f.write(
            f"std::vector<std::array<int, RECORD_LENGTH_RTC> > drtc_n_{args.arch}" +
            " = {\n")
        count = 0
        for row in rows:
            f.write("    {" + re.sub(r'(\s+)', r',\1', row) +
                    ("},\n" if count < len(rows) - 1 else "}\n"))
            count += 1
        f.write("};\n")
