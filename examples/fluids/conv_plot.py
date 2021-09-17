#!/usr/bin/env python3

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

import pandas as pd
import argparse
from pylab import *
from matplotlib import use


def plot():
    # Define argparse for the input variables
    parser = argparse.ArgumentParser(description='Get input arguments')
    parser.add_argument('-f',
                        dest='conv_result_file',
                        type=str,
                        required=True,
                        help='Path to the CSV file')
    args = parser.parse_args()
    conv_result_file = args.conv_result_file

    # Load the data
    runs = pd.read_csv(conv_result_file)
    colors = ['orange', 'red', 'navy', 'green', 'magenta',
              'gray', 'blue', 'purple', 'pink', 'black']
    res = 'mesh_res'
    fig, ax = plt.subplots()

    i = 0
    for group in runs.groupby('degree'):
        data = group[1]
        data = data.sort_values('rel_error')
        p = data['degree'].values[0]
        h = 1/data[res]
        E = data['rel_error']
        H =  amin(E) * (h/amin(h))**p
        ax.loglog(h, E, 'o', color=colors[i])
        ax.loglog(h, H, '--', color=colors[i], label='O(h$^' + str(p) + '$)')
        i = i + 1

    ax.legend(loc='best')
    ax.set_xlabel('h')
    ax.set_ylabel('Relative Error')
    ax.set_title('Convergence by h Refinement')
    plt.savefig('conv_plt_h.png')


if __name__ == "__main__":
    plot()
