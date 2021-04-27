# libCEED Python Tests

These files provide libCEED for Python tests. Full examples of finite element
operators can be found in the file `test-5-operator.py`.

## Testing

To run the tests, first build the user QFunctions file by running

  python setup-qfunctions.py build

Then, to run the test suite use the command

  pytest test-*.py --ceed /cpu/self/ref/serial

## Building QFunctions

To build user defined QFunctions, modify `libceed-qfunctions.c` to include
the apropriate QFunction single source file and run
`python setup-qfunctions.py build`. The files `test-4-qfunction.py` and
`test-5-operator.py` both contain the example function `load_qfs_so()` for
loading the user defined QFunctions so the QFunction pointers can be passed to
libCEED.
