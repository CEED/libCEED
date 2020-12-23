import fileinput
import numpy as np
import pandas as pd
from pylab import *
from matplotlib import use


def read_logs(files=None):
    """Read all input files and return pandas DataFrame"""
    data_default = dict(
        problem='unknown',
        num_procs=0,
        mesh_res=0,
        degree=0,
        quadrature_pts=0,
        num_unknowns=0,
        max_error=0
    )
    data = data_default.copy()
    runs = []
    for line in fileinput.input(files):
        # Number of MPI tasks
        if 'rank(s)' in line:
            data = data_default.copy()
            data['num_procs'] = int(line.split(': ', 1)[1])
        # New Problem
        elif "Problem Name" in line:
            # Starting a new block
            data = data.copy()
            runs.append(data)
            data['problem'] = line.split(': ')[1].strip()
        # Mesh resolution
        elif "Box Faces" in line:
            res = line.split(': ')[1]
            data['mesh_res'] = int(line.split(',')[1])
        # P
        elif 'Basis Nodes' in line:
            data['degree'] = int(line.split(': ')[1]) - 1
        # Q
        elif 'Quadrature Points' in line:
            data['quadrature_pts'] = int(line.split(': ')[1])
        # Total DOFs
        elif 'Global DoFs' in line:
            data['num_unknowns'] = int(line.split(': ')[1])
        # Max Error
        elif 'Max Error' in line:
            data['max_error'] = float(line.split(': ')[1])
        # End of output
    return pd.DataFrame(runs)


### Plotting
def plot():
    # Load the data
    runs = read_logs()
    colors = ['orange', 'red', 'navy', 'green', 'magenta',
              'teal', 'blue', 'purple', 'pink', 'cyan']
    xaxis = 'mesh_res'
    yaxis = 'max_error'
    fig, ax = plt.subplots()
    i = 0
    for group in runs.groupby('degree'):
        data = group[1]
        data = data.sort_values('max_error')
        x = data[xaxis]
        y = data[yaxis]
        ax.loglog(x, y, 'o', color=colors[i])
        log_x = np.log10(x)
        log_y = np.log10(y)
        m, b = np.polyfit(log_x, log_y, 1)
        lbl = data['degree'].values[0]
        ax.loglog(x, 10**b * x**m, '-', color=colors[i], label='p=' + str(lbl))
        i = i + 1

    ax.legend(loc='best')
    ax.set_xlabel('h')
    ax.set_ylabel('Max Error')
    ax.set_title('Convergence by h Refinement')
    xlim(1, 100)
    fig.tight_layout()
    plt.savefig('h_ref_conv_test.png', bbox_inches='tight')


if __name__ == "__main__":
    runs = read_logs()
    plot()
    print(runs)
