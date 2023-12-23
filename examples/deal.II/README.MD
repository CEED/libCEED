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

Run the executable to create an input file:

```
./bp > input.json
```

Modify the input file and run the executable:
```
mpirun -np 8 ./bp input.json
```

