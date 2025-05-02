from setuptools import setup, Extension
from sys import platform
import os

# Get CEED directory
ceed_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Include directories
include_dirs = [os.path.join(ceed_dir, "include")]

# Library directories
library_dirs = [os.path.join(ceed_dir, "lib")]

# Source files
sources = ["qfunctions.c"]

# Compiler arguments
extra_compile_args = []
if platform == "linux" or platform == "linux2" or platform == "darwin":
    extra_compile_args = ["-O3", "-march=native", "-std=c99"]

# Define the extension module
qfunctions = Extension("libceed_qfunctions",
                      sources=sources,
                      include_dirs=include_dirs,
                      library_dirs=library_dirs,
                      libraries=["ceed"],
                      extra_compile_args=extra_compile_args)

# Setup
setup(name="libceed_qfunctions",
      ext_modules=[qfunctions])
