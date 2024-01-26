An example how to write libCEED operators (BP1-BP6) within the open-source
finite element library [deal.II](https://www.dealii.org/). As reference,
operators are presented that use the native matrix-free infrastructure.

First compile deal.II and libCEED individually. After that, compile the deal.II example:

```bash
mkdir build
cd build
cmake ../ -DDEAL_II_DIR=~/path/to/dealii -DCEED_DIR=~/path/to/libceed
make
```

To run the executable, write:

```
./bps
```

Optional command-line arguments are shown by adding the command-line argument "--help".
