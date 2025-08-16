# libCEED: Code for Efficient Extensible Discretization

libCEED is a lightweight library for expressing and manipulating operators that arise in high-order element-based discretization of partial differential equations.
libCEED's representations are much for efficient than assembled sparse matrices, and can achieve very high performance on modern CPU and GPU hardware.
This approach is applicable to a broad range of linear and nonlinear problems, and includes facilities for preconditioning.
libCEED is meant to be easy to incorporate into existing libraries and applications, and to build new tools on top of.

libCEED has been developed as part of the DOE Exascale Computing Project
co-design Center for Efficient Exascale Discretizations (CEED).

## Install

To install libCEED for Python, run

    pip install libceed

or in a clone of the repository via `pip install .`

## Examples and Tutorials

For examples and short tutorials see the folder `examples/tutorials`. It
contains some [Jupyter](https://jupyter.org/) notebooks using Python and C.
Jupyter can be installed locally so that users can edit and interact with these
notebook.

`tutorial-0`-`tutorial-5` illustrate libCEED for Python, each one focusing on one
class of objects.

`tutorial-6` shows a standalone libCEED C example.
