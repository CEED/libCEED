#!/bin/bash

ulimit -c 0 # Do not dump core

# Make CeedError exit nonzero without using signals/abort()
export CEED_ERROR_HANDLER=exit

output=$(mktemp $1.XXXX)
backends=(${BACKENDS:?Variable must be set, e.g., \"/cpu/self/ref /cpu/self/blocked\"})
target="$1"
# Only the unit tests (txxx) link with libceed_test (where /cpu/self/tmpl is
# defined), so filter those backends out for everything else.  Note that this is
# only relevant for the prove target; the test and junit targets are managed in
# the makefile and set BACKENDS appropriately.
if [ "t" != "${target::1}" ]; then
    for idx in ${!backends[@]}; do
        test /cpu/self/tmpl = ${backends[$idx]::14} && unset backends[$idx]
    done
fi
printf "1..$[3*${#backends[@]}]\n";

# for examples/ceed petsc*, mfem*, or ex* grep the code to fetch arguments from a TESTARGS line
if [ ${1::6} == "petsc-" ]; then
    args=$(grep -F //TESTARGS examples/petsc/${1:6}.c* | cut -d\  -f2- )
elif [ ${1::5} == "mfem-" ]; then
    args=$(grep -F //TESTARGS examples/mfem/${1:5}.c* | cut -d\  -f2- )
elif [ ${1::4} == "nek-" ]; then
    args=$(grep -F "C TESTARGS" examples/nek/bps/${1:4}.usr* | cut -d\  -f3- )
elif [ ${1::2} == "ex" ]; then
    args=$(grep -F //TESTARGS examples/ceed/$1.c | cut -d\  -f2- )
else
    args='{ceed_resource}'
fi

tmpfiles="${output} ${output}.out ${output}.diff ${output}.err SESSION.NAME"
trap 'rm -f ${tmpfiles}' EXIT

for ((i=0;i<${#backends[@]}; ++i)); do
    i0=$((3*$i+1)) # return code
    i1=$(($i0+1))  # stdout
    i2=$(($i0+2))  # stderr
    backend=${backends[$i]}

    # Run in subshell
    (build/$1 ${args/\{ceed_resource\}/$backend} || false) > ${output}.out 2> ${output}.err
    status=$?

    # grep to skip test if backend cannot handle resource
    if grep -F -q -e 'OCCA backend failed to use' ${output}.err; then
        printf "ok $i0 # SKIP - occa mode not supported $1 $backend\n"
        printf "ok $i1 # SKIP - occa mode not supported $1 $backend stdout\n"
        printf "ok $i2 # SKIP - occa mode not supported $1 $backend stderr\n"
        continue
    fi

    # grep to skip test if backend chooses to whitelist test
    if grep -F -q -e 'Backend does not implement' \
            ${output}.err ; then
        printf "ok $i0 # SKIP - not implemented $1 $backend\n"
        printf "ok $i1 # SKIP - not implemented $1 $backend stdout\n"
        printf "ok $i2 # SKIP - not implemented $1 $backend stderr\n"
        continue
    fi

    # grep to pass test t103, t104, t105, t106, t107 on error
    if grep -F -q -e 'access' ${output}.err \
            && [[ "$1" = "t103"* || "$1" = "t104"* || "$1" = "t105"* || "$1" = "t106"* || "$1" = "t107"* ]] ; then
        printf "ok $i0 PASS - expected failure $1 $backend\n"
        printf "ok $i1 PASS - expected failure $1 $backend stdout\n"
        printf "ok $i2 PASS - expected failure $1 $backend stderr\n"
        continue
    fi

    # grep to pass test t308 on error
    if grep -F -q -e 'vectors incompatible' ${output}.err \
            && [[ "$1" = "t308"* ]] ; then
        printf "ok $i0 PASS - expected failure $1 $backend\n"
        printf "ok $i1 PASS - expected failure $1 $backend stdout\n"
        printf "ok $i2 PASS - expected failure $1 $backend stderr\n"
        continue
    fi

    # grep to skip test if Device memory is not supported
    if grep -F -q -e 'Can only provide to HOST memory' \
            ${output}.err ; then
        printf "ok $i0 # SKIP - not supported $1 $backend\n"
        printf "ok $i1 # SKIP - not supported $1 $backend stdout\n"
        printf "ok $i2 # SKIP - not supported $1 $backend stderr\n"
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
