#!/usr/bin/env python3

# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

import argparse
import os
import glob
import re
import shutil
import subprocess
import pandas as pd
import time

script_dir = os.path.dirname(os.path.realpath(__file__))


def benchmark(nb, build_cmd, backend, log):
    # Build for new NB
    ceed_magma_h = f"{script_dir}/../ceed-magma.h"
    shutil.copyfile(ceed_magma_h, ceed_magma_h + ".backup")
    with open(ceed_magma_h, "r") as f:
        data = f.read()
        data = re.sub(
            r".*(#define ceed_magma_queue_sync\(\.\.\.\)).*",
            r"\1 " +
            ("hipDeviceSynchronize()" if "hip" in backend else "cudaDeviceSynchronize()"),
            data)
    with open(ceed_magma_h, "w") as f:
        f.write(data)

    ceed_magma_gemm_selector_cpp = f"{script_dir}/../ceed-magma-gemm-selector.cpp"
    shutil.copyfile(
        ceed_magma_gemm_selector_cpp,
        ceed_magma_gemm_selector_cpp +
        ".backup")
    with open(ceed_magma_gemm_selector_cpp, "r") as f:
        data = f.read()
        data = re.sub(
            ".*(#define CEED_AUTOTUNE_RTC_NB).*",
            r"\1 " + f"{nb}",
            data)
    with open(ceed_magma_gemm_selector_cpp, "w") as f:
        f.write(data)

    subprocess.run(build_cmd, cwd=f"{script_dir}/../../..")
    subprocess.run(["make", "tuning", "OPT=-O0"], cwd=f"{script_dir}")
    shutil.move(ceed_magma_h + ".backup", ceed_magma_h)
    shutil.move(ceed_magma_gemm_selector_cpp +
                ".backup", ceed_magma_gemm_selector_cpp)

    # Run the benchmark
    with open(log, "w") as f:
        process = subprocess.run(
            [f"{script_dir}/tuning", f"{backend}"], stdout=f, stderr=f)
    csv = pd.read_csv(
        log,
        header=None,
        delim_whitespace=True,
        names=[
            "P",
            "Q",
            "N",
            "Q_COMP",
            "TRANS",
            "MFLOPS"])
    return csv


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
        "-build-cmd",
        help="Command used to build libCEED from the source root directory",
        default="make")
    parser.add_argument(
        "-ceed",
        help="Ceed resource specifier",
        default="/cpu/self")
    args = parser.parse_args()

    nb = 1
    while nb <= args.max_nb:
        # Run the benchmarks
        start = time.perf_counter()
        data_nb = benchmark(nb, args.build_cmd, args.ceed,
                            f"{script_dir}/output-nb-{nb}.txt")
        print(
            f"Finished benchmarks for NB = {nb}, backend = {args.ceed} ({time.perf_counter() - start} s)")

        # Save the data for the highest performing NB
        if nb == 1:
            data = pd.DataFrame(data_nb)
            data["NB"] = nb
        else:
            idx = data_nb["MFLOPS"] > 1.05 * data["MFLOPS"]
            data.loc[idx, "NB"] = nb
            data.loc[idx, "MFLOPS"] = data_nb.loc[idx, "MFLOPS"]

        # Speed up the search by considering only some values on NB
        if nb < 2:
            nb *= 2
        elif nb < 8:
            nb += 2
        else:
            nb += 4

    # Print the results
    with open(f"{script_dir}/{args.arch}_rtc.h", "w") as f:
        f.write(
            "////////////////////////////////////////////////////////////////////////////////\n")
        f.write(f"// auto-generated from data on {args.arch}\n\n")

        rows = data.loc[data["TRANS"] == 1].to_string(header=False, index=False, justify="right", columns=[
                                                      "P", "Q", "N", "Q_COMP", "NB"]).split("\n")
        f.write(
            "////////////////////////////////////////////////////////////////////////////////\n")
        f.write(
            f"std::vector<std::array<int, RECORD_LENGTH_RTC> > drtc_t_{args.arch}" +
            " = {\n")
        count = 0
        for row in rows:
            f.write("    {" + re.sub(r"([0-9])(\s+)", r"\1,\2", row) +
                    ("},\n" if count < len(rows) - 1 else "}\n"))
            count += 1
        f.write("};\n\n")

        rows = data.loc[data["TRANS"] == 0].to_string(header=False, index=False, justify="right", columns=[
                                                      "P", "Q", "N", "Q_COMP", "NB"]).split("\n")
        f.write(
            "////////////////////////////////////////////////////////////////////////////////\n")
        f.write(
            f"std::vector<std::array<int, RECORD_LENGTH_RTC> > drtc_n_{args.arch}" +
            " = {\n")
        count = 0
        for row in rows:
            f.write("    {" + re.sub(r"([0-9])(\s+)", r"\1,\2", row) +
                    ("},\n" if count < len(rows) - 1 else "}\n"))
            count += 1
        f.write("};\n")
