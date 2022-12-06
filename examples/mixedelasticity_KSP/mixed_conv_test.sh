#!/bin/bash

# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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

# You can run convergence test by:
#./mixed_conv_test.sh -u 2 -p 1

# where u, p are polynomial orders of displacement and pressure fields
# Reading arguments with getopts options
while getopts u:p: flag
do
    case "${flag}" in
        u) order_u=${OPTARG};;
        p) order_p=${OPTARG};;
    esac
done

echo "Running convergence test for mixed-linear elasticity with polynomial order u:${order_u}, p:${order_p}";
declare -A run_flags

    run_flags[dm_plex_dim]=2
    run_flags[dm_plex_box_faces]=4,4
    run_flags[dm_plex_simplex]=0
    run_flags[u_order]=$order_u
    run_flags[p_order]=$order_p
    run_flags[problem]=mixed-linear-2d
    run_flags[ksp_max_it]=1000
    run_flags[q_extra]=1
    run_flags[pc_type]=svd

declare -A test_flags
    test_flags[res_start]=4
    test_flags[res_stride]=2
    test_flags[res_end]=10

file_name=conv_test_result.csv

echo "run,mesh_res,error_u,error_p" > $file_name

i=0

for ((res=${test_flags[res_start]}; res<=${test_flags[res_end]}; res+=${test_flags[res_stride]})); do

    run_flags[dm_plex_box_faces]=$res,$res

    args=''
    for arg in "${!run_flags[@]}"; do
        if ! [[ -z ${run_flags[$arg]} ]]; then
            args="$args -$arg ${run_flags[$arg]}"
        fi
    done
    ./main $args | grep "L2 error" | awk -v i="$i" -v res="$res" '{ printf "%d,%d,%e,%e\n", i, res, $8, $9}' >> $file_name
    i=$((i+1))
done

python mixed_conv_rate.py -f conv_test_result.csv