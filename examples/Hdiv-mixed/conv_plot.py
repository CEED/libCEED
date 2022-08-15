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

# After ./conv_test.sh you can plot using
# python conv_plot.py -f conv_test_result.csv

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
    data = pd.read_csv(conv_result_file)
    fig, ax = plt.subplots()

    data = data.sort_values('run')

    E_u = data['error_u']
    E_p = data['error_p']
    #E_hdiv = data['error_hdiv']
    h = 1/data['mesh_res']
    N = data['mesh_res']
    H1 =  amin(E_p)* (h/amin(h)) # H = C h^1
    H2 =  amin(E_u)* (h/amin(h))**2  # H = C h^2

    ax.loglog(h, E_p, 'o', color='blue', label='Pressure')
    ax.loglog(h, E_u, 'o', color='black', label = 'Velocity')
    #ax.loglog(h, E_hdiv, '*', color='red', label = 'Velocity in H(div)')
    ax.loglog(h, H1, '--', color='blue', label='O(h)')
    ax.loglog(h, H2, '--', color='black', label='O(h$^2$)')

    ax.legend(loc='upper left')
    ax.set_xlabel('h')
    ax.set_ylabel('L2 Error')
    ax.set_title('Convergence by h Refinement')
    #xlim(.06, .3)
    fig.tight_layout()
    plt.savefig('convrate_mixed.png', bbox_inches='tight')

    conv_u = []
    conv_p = []
    conv_u.append(0)
    conv_p.append(0)
    for i in range(1,len(E_u)):
        conv_u.append(log10(E_u[i]/E_u[i-1])/log10(h[i]/h[i-1]))
        conv_p.append(log10(E_p[i]/E_p[i-1])/log10(h[i]/h[i-1]))

    result = {'Number of element':N, 'order of u':conv_u, 'order of p':conv_p}
    df = pd.DataFrame(result)
    print(df)
    


if __name__ == "__main__":
    plot()
