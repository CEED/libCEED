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

make -B

declare -A run_flags
    run_flags[problem]=euler_vortex  # Options: "euler_vortex" and "advection2d"
    run_flags[degree]=2
    run_flags[dm_plex_box_faces]=20,20,1
    run_flags[ts_adapt_dt_max]=.01
    run_flags[P_inlet]=1
    run_flags[T_inlet]=1
    run_flags[euler_test]=none
    run_flags[lx]=900
    run_flags[ly]=900
    run_flags[lz]=1
    run_flags[ts_max_time]=.1
    #run_flags[ts_max_steps]=20   # turn off time stepping

# Remove previous test results
if ! [[ -z ./verification-output/${run_flags[problem]}/*.log ]]; then
    rm -R ./verification-output/${run_flags[problem]}/*.log
fi

declare -A test_flags
    test_flags[degree_start]=1
    test_flags[degree_end]=4
    test_flags[res_start]=10
    test_flags[res_stride]=20
    test_flags[res_end]=90

alpha=.5


for ((d=${test_flags[degree_start]}; d<=${test_flags[degree_end]}; d++)); do
    run_flags[degree]=$d
    for ((res=${test_flags[res_start]}; res<=${test_flags[res_end]}; res+=${test_flags[res_stride]})); do
        run_flags[dm_plex_box_faces]=$res,$res,1
        beta=$((($res)*($d)))
        run_flags[ts_adapt_dt_max]=$(echo -e scale=5 "\n" $alpha \/ $beta |bc -l)
        args=''
        for arg in "${!run_flags[@]}"; do
            if ! [[ -z ${run_flags[$arg]} ]]; then
                args="$args -$arg ${run_flags[$arg]}"
            fi
        done
        echo $args  &>> ./verification-output/${run_flags[problem]}/${run_flags[degree]}_${res}.log
        mpiexec.hydra -n 4 ./navierstokes -ts_adapt_monitor $args  &>> ./verification-output/${run_flags[problem]}/${run_flags[degree]}_${res}.log
    done
done

# Remove ns files
rm ns-*

# Plot
# python3 convergence_plot.py verification-output/euler_vortex/*.log
python3 convergence_plot.py verification-output/${run_flags[problem]}/*.log