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

# After make the problem, you can run convergence test by: ./conv_test.sh -d 2 (or -d 3)
# Reading arguments with getopts options
while getopts d: flag
do
    case "${flag}" in
        d) dim=${OPTARG};;
    esac
done
echo "Running convergence test in ${dim}D for Projection problem in H(div) space";

declare -A run_flags
    #run_flags[pc_type]=svd
    run_flags[ceed]=/cpu/self/ref/serial
    if [[ $dim -eq 2 ]];
    then
        run_flags[problem]=mass2d
        run_flags[dm_plex_dim]=$dim
        run_flags[dm_plex_box_faces]=2,2
        run_flags[dm_plex_box_lower]=0,0
        run_flags[dm_plex_box_upper]=1,1
    else
        run_flags[problem]=mass3d
        run_flags[dm_plex_dim]=$dim
        run_flags[dm_plex_box_faces]=2,2,2
    fi

declare -A test_flags
    test_flags[res_start]=4
    test_flags[res_stride]=2
    test_flags[res_end]=12

file_name=conv_test_result.csv

echo "run,mesh_res,error_u" > $file_name

i=0

for ((res=${test_flags[res_start]}; res<=${test_flags[res_end]}; res+=${test_flags[res_stride]})); do
    if [[ $dim -eq 2 ]]; then
        run_flags[dm_plex_box_faces]=$res,$res
    else
        run_flags[dm_plex_box_faces]=$res,$res,$res
    fi
    args=''
    for arg in "${!run_flags[@]}"; do
        if ! [[ -z ${run_flags[$arg]} ]]; then
            args="$args -$arg ${run_flags[$arg]}"
        fi
    done
    ./main $args | grep "L2 error of u" | awk -v i="$i" -v res="$res" '{ printf "%d,%d,%e\n", i, res, $6}' >> $file_name
    i=$((i+1))
done

python conv_rate.py -f conv_test_result.csv