#!/bin/bash

ulimit -c 0 # Do not dump core

output=$(mktemp $1.XXXX)

backends=(${BACKENDS:?Variable must be set, e.g., \"/cpu/self/ref /cpu/self/opt\"})
printf "1..$[3*${#backends[@]}]\n";

# for examples/ceed ex*, grep the code to fetch arguments from a TESTARGS line
if [ ${1::6} == "petsc-" ]; then
    args=$(grep -F //TESTARGS examples/petsc/${1:6}.c* | cut -d\  -f2- )
elif [ ${1::5} == "mfem-" ]; then
    args=$(grep -F //TESTARGS examples/mfem/${1:5}.c* | cut -d\  -f2- )
elif [ ${1::2} == "ex" ]; then
    args=$(grep -F //TESTARGS examples/ceed/$1.c | cut -d\  -f2- )
else
    args='{ceed_resource}'
fi

for ((i=0;i<${#backends[@]}; ++i)); do
    i0=$((3*$i+1)) # return code
    i1=$(($i0+1))  # stdout
    i2=$(($i0+2))  # stderr
    backend=${backends[$i]}

    # Run in subshell
    (build/$1 ${args/\{ceed_resource\}/$backend} || false) > ${output}.out 2> ${output}.err
    status=$?
    # grep to skip test if backend cannot handle resource
    if grep -F -q -e 'backend cannot use resource' \
            -e 'OCCA backend failed' ${output}.err; then
        printf "ok $i0 # SKIP $1 $backend\n"
        printf "ok $i1 # SKIP $1 $backend stdout\n"
        printf "ok $i2 # SKIP $1 $backend stderr\n"
        continue
    fi

    if [ $status -eq 0 ]; then
        printf "ok $i0 $1 $backend\n"
    else
        printf "not ok $i0 $1 $backend\n"
    fi

    # stdout
    if [ -f output/$1.out ]; then
        if diff -u output/$1.out ${output}.out > ${output}.diff; then
            printf "ok $i1 $1 $backend stdout\n"
        else
            printf "not ok $i1 $1 $backend stdout\n"
            while read line; do
                printf "# ${line}\n"
            done < ${output}.diff
        fi
    elif [ -s ${output}.out ]; then
        printf "not ok $i1 $1 $backend stdout\n"
        while read line; do
            printf "# + ${line}\n"
        done < ${output}.out
    else
        printf "ok $i1 $1 $backend stdout\n"
    fi
    # stderr
    if [ -s ${output}.err ]; then
        printf "not ok $i2 $1 $backend stderr\n"
        while read line; do
            printf "# +${line}\n"
        done < ${output}.err
    else
        printf "ok $i2 $1 $backend stderr\n"
    fi
done
rm -f ${output} ${output}.out ${output}.diff ${output}.err
