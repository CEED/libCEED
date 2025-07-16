## libCEED Python Examples

These examples are written using libCEED's Python interface.

### Tutorials

These Jupyter notebooks explore the concepts of the libCEED API, including how to install the Python interface and the usage of each API object, with interactive examples.

### Basic Examples

The basic libCEED C examples in the folder `/examples/ceed` are also available as Python examples.

To build the QFunctions into a shared library that the Python examples use, run

```bash
make setup
```

To execute the examples, run:

```
python ex1_volume.py
```

A full list of command-line arguments are shown by adding the command-line argument "--help".
