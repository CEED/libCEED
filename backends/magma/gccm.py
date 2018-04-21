# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
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

################################################################################
#  -- MAGMA (version 2.0) --
#     Univ. of Tennessee, Knoxville
#     @date
#
# File gccm.py
#
# This is a MAGMA DSL compiler.
# Given a file, e.g., 'src.c' that contains MAGMA DSL constructs, executing
#    python gccm.py src.c
# will generate the equivalent C code in src_tmp.c that calls CUDA kernels,
# generated in file src_tmp.cu. The original src.c file remains unchanged.
#
################################################################################
import sys

def main():
    # The source file is the 1st argument to the script
    if len(sys.argv) != 2:
        print('usage: %s <src.c>' % sys.argv[0])
        sys.exit(1)

    # read the input file in prog
    file = open(sys.argv[1], 'r')
    prog = file.read()
    file.close()

    fname  = sys.argv[1]
    fname  = fname[:fname.find(".")]
    cfile  = open(fname + "_tmp.c" , "w")
    cufile = open(fname + "_cuda.cu", "w")

    print('%s %s --> %s %s' % (sys.argv[0],sys.argv[1], fname+"_tmp.c", fname+"_cuda.cu"))

    fname  = fname.replace("-", "_")
    fname  = fname[fname.rfind("/")+1:]

    cufile.write("#include <ceed-impl.h>                       \n\n")
    cufile.write("#include \"atomics.cuh\"                       \n")
    cufile.write("#include \"magma_check_cudaerror.h\"         \n\n")
    cufile.write("#define  MAX_TB_XDIM      2147483647           \n")
    cufile.write("#define  MAX_THREADS_PTB  1024                 \n")
    cufile.write("#define  OUR_THREADS_PTB  512                \n\n")

    # Search for the magma_template keyword in the input file
    for kernel in range(prog.count("magma_template")):
        i1 = prog.index("magma_template")
        cfile.write(prog[:i1])
        prog = prog[i1+14:]

        # templates are of the form:
        #    magma_template<<e=0:r->nelem, d=0:ncomp, i=0:r->elemsize>>
        #      (const CeedScalar *uu, CeedScalar *vv)
        #         { code_for_single_thread; }

        # text between << >> , i.e, bound = "e=0:r->nelem, d=0:ncomp, i=0:r->elemsize"
        bounds   = prog[prog.find("<<")+2:prog.find(">>")]

        # text/argd betweer (), i.e., args = "const CeedScalar *uu, CeedScalar *vv"
        args     = prog[prog.find("(")+1 : prog.find(")")]

        # text between {}, i.e., template = "code_for_single_thread;"
        i1, i2   = find_matching(prog,'{','}')
        template = prog[i1+1:i2]

        prog     = prog[i2:]

        # boundvars = "e, d, i" and boundargs = "0, r->nelem, 0, ncomp, 0, r->elemsize"
        boundvars, boundargs  = find_argsbound( bounds )

        # add arguments from kernel, i.e. listars += "uu, vv"
        boundargs +=  "," + find_argslist( args )

        # Here parts[0] = e , parts[1] = d, parts[2] = i
        parts = boundvars.split(",");
        numparts = len(parts)

        # Replace DSL call with the real generated call in the tmp.c file
        kname = "magma_template" + "_" + str(kernel) + "_" + fname
        kcall = kname + "(" + boundargs + ");"
        declaration  = "void                                            \n"
        declaration +=  kname+"(                                        \n"
        for i in range(numparts):
            declaration += "    int "+parts[i]+"begin, int "+parts[i]+"end,\n"
        declaration += "    " + args + ");                              \n"
        cfile.write(declaration)
        cfile.write(kcall)

        # Set how to initialize parts
        partsinit = ["blockIdx.x", "blockIdx.y", "threadIdx.x"]
        if (numparts == 1): partsinit[0] = partsinit[2]
        if (numparts == 2): partsinit[2] = partsinit[2]

        # Prepare the magma_template kernel and write it in the .cu file ###
        kernel  = "__global__ void                                      \n"
        kernel +=  kname+"_kernel(                                      \n"
        for i in range(numparts):
            kernel += "    int "+parts[i]+"begin, int "+parts[i]+"end,  \n"
        kernel += "    " + args+ ")                                     \n"
        kernel += "{                                                    \n"
        if (numparts == 1):
            # preparing the kernel for template1 (one loop)  
            kernel += "   int "+parts[0]+" = " +  \
                parts[0]+"begin + blockIdx.x*blockDim.x+threadIdx.x;\n"
            kernel += "   if ("+parts[0]+" < "+parts[0]+"end)       \n"
            kernel += "   {                                         \n"
            kernel += template  +                                  "\n"
            kernel += "}  \n\n"
        else:
            # preparing the kernel for template2 (two loops) and template3 (3 loops)
            for i in range(numparts):
                kernel += "   int "+parts[i]+" = "+ \
                    parts[i]+"begin + "+partsinit[i]+";\n"
            kernel += template +             "                    \n\n"

        # the interface function calling the kernel ########################
        kernel += "extern \"C\" void                                    \n"
        kernel +=  kname+"(                                             \n"
        for i in range(numparts):
            kernel += "    int "+parts[i]+"begin, int "+parts[i]+"end,  \n"
        kernel += "    " + args + ")\n"
        kernel += "{                \n"
        if (numparts == 1):
            # preparing the function for template1 (one loop)
            kernel += "   int siz = "+parts[0]+"end-"+parts[0]+"begin;             \n"  
            kernel += "   int THR_PTB = siz>MAX_THREADS_PTB? OUR_THREADS_PTB : siz;\n"
            kernel += "   int nTB = magma_ceildiv(siz, THR_PTB);                 \n\n"
            kernel += "   for(int r=0; r<nTB; r+=MAX_TB_XDIM)                      \n"
            kernel += "   {                                                        \n"
            kernel += "      int  this_TB = min(MAX_TB_XDIM, nTB-r);               \n"
            kernel += "      dim3 grid(this_TB, 1, 1);                             \n"
            kernel += "      dim3 threads(THR_PTB);                              \n\n"
            kernel += "      " + kname + "_kernel<<<grid,threads,0>>>(             \n"
            kernel += "          " + parts[0]+"begin+r*THR_PTB, "+parts[0]+"end,   \n"
            kernel += "          " + find_argslist( args ) + ");                   \n"
            kernel += "      magma_check_cudaerror();                              \n"
            kernel += "   }                                                        \n"
        elif (numparts == 2):
            kernel += "   dim3 grid(   "+parts[0]+"end -"+parts[0]+"begin);  \n"
            kernel += "   dim3 threads("+parts[1]+"end-" +parts[1]+"begin);\n\n"
            kernel += "   " + kname + "_kernel<<<grid,threads,0>>>(         \n"
            for i in range(numparts):
                kernel += "    " + parts[i]+"begin, "+parts[i]+"end,        \n"
            kernel += "   " + find_argslist( args ) + ");                   \n"
        else:
            kernel += "   dim3 grid(   "+parts[0]+"end -"+parts[0]+"begin,"
            kernel +=                    parts[1]+"end -"+parts[1]+"begin); \n"
            kernel += "   dim3 threads("+parts[2]+"end-" +parts[2]+"begin);\n\n"
            kernel += "   " + kname + "_kernel<<<grid,threads,0>>>(         \n"
            for i in range(numparts):
                kernel += "    " + parts[i]+"begin, "+parts[i]+"end,        \n"
            kernel += "   " + find_argslist( args ) + ");                   \n"
        kernel += "}                                                  \n\n"

        # write the current magma templates kernel
        cufile.write(kernel)
        # print(kernel)

    # write until the end the rest of the original file
    cfile.write(prog)

    # close the files
    cfile.close()
    cufile.close()

################################################################################
# Auxiliary fnctions
################################################################################

# Given a code in 'line' this returns the indices for the sub-code starting
# from first "left" parenthesis/string to the closing "right" parenthesis
def find_matching(line, left, right):
    i3 = line.find(left)
    num= 1
    i4 = len(line)
    for i in range(i3+1,i4):
        if line[i] == right:
            num -= 1
        elif line[i] == left:
            num += 1
        if num == 0:
            return i3, i+1
    return i3, i4

# Given a declaration list of args separated by ",", this function
# parses the list and returns a list of just their names
def find_argslist( args ):
    args += ")"
    arglist = ""
    numargs = args.count(",")+1
    for i in range(numargs):
        i1 = args.find(",")
        if i1 == -1:
            i1 = args.find(")")
        for j in range(i1):
            if args[i1-j] in {' ', '*', '&'}:
                arglist = arglist + "," + args[i1-j+1:i1]
                args = args[i1+1:]
                break

    return arglist[1:]

# Given bounds list of the form "e=0:r->nelem, d=0:ncomp, ..." this returns
# the list of parameters "e, d, ..." and list of their start and end arguments,
# i.e., "0, r->nelem, 0, ncomp, ..."
def find_argsbound( bounds ):
    bounds   += ","
    args      = ""
    argsbound = ""
    numargs = bounds.count("=")
    for i in range(numargs):
        if (i != 0 ):
            argsbound += ","
            args      += ","
        i1 = bounds.find("=")
        args = args + bounds[:i1]
        argsbound = argsbound + bounds[i1+1 : bounds.find(":")] + ","
        argsbound = argsbound + bounds[bounds.find(":")+1 : bounds.find(",")]
        bounds = bounds[bounds.find(",")+1:]

    return args, argsbound

################################################################################

if __name__ == '__main__':
   main()
