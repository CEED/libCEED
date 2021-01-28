#!/bin/bash

# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
# All Rights reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

declare -A run_flags
    run_flags[problem]=euler_vortex
    run_flags[degree]=2
    run_flags[dm_plex_box_faces]=20,20,1
    run_flags[P_inlet]=1
    run_flags[T_inlet]=1
    run_flags[euler_test]=none
    run_flags[lx]=1e3
    run_flags[ly]=1e3
    run_flags[lz]=1
    run_flags[ts_max_time]=.02
    run_flags[ts_rk_type]=5bs
    run_flags[ts_rtol]=1e-10
    run_flags[ts_atol]=1e-10

declare -A test_flags
    test_flags[degree_start]=1
    test_flags[degree_stride]=1
    test_flags[degree_end]=3
    test_flags[res_start]=6
    test_flags[res_stride]=2
    test_flags[res_end]=10

echo ",mesh_res,degree,rel_error" > conv_test_result.csv
i=0
for ((d=${test_flags[degree_start]}; d<=${test_flags[degree_end]}; d+=${test_flags[degree_stride]})); do
    run_flags[degree]=$d
    for ((res=${test_flags[res_start]}; res<=${test_flags[res_end]}; res+=${test_flags[res_stride]})); do
        run_flags[dm_plex_box_faces]=$res,$res,1
        args=''
        for arg in "${!run_flags[@]}"; do
            if ! [[ -z ${run_flags[$arg]} ]]; then
                args="$args -$arg ${run_flags[$arg]}"
            fi
        done
        ./navierstokes $args | grep "Relative Error:" | awk -v i="$i" -v res="$res" -v d="$d" '{ print i","res","d","$3}' >> conv_test_result.csv
        i=$((i+1))
    done
done

# Compare the output CSV file with the reference file
diff conv_test_result.csv tests-output/fluids_navierstokes_etv.csv
