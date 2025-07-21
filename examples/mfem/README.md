## libCEED MFEM Examples

These examples show to write libCEED operators (BP1 and BP3) within the open-source finite element library [MFEM](https://www.mfem.org/).

First compile MFEM and libCEED individually. After that, compile the MFEM example:

```bash
export MFEM_DIR=/path/to/mfem
make
```

To run the executable, write:

```
./bp[1, 3]
```

Optional command-line arguments are shown by adding the command-line argument "--help".
