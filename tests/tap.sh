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

# for examples/ceed petsc*, mfem*, or ex* grep the code to fetch arguments from a TESTARGS line
declare -a allargs
if [ ${1::6} == "petsc-" ]; then
    allargs=$(grep -F //TESTARGS examples/petsc/${1:6}.c* | cut -d\  -f2- )
elif [ ${1::5} == "mfem-" ]; then
    allargs=$(grep -F //TESTARGS examples/mfem/${1:5}.c* | cut -d\  -f2- )
elif [ ${1::4} == "nek-" ]; then
    allargs=$(grep -F "C TESTARGS" examples/nek/bps/${1:4}.usr* | cut -d\  -f3- )
elif [ ${1::2} == "ex" ]; then
    # get all test configurations
    numconfig=$(grep -F //TESTARGS examples/ceed/$1.c* | wc -l)
    for ((i=0;i<${numconfig};++i)); do
      allargs+=("$(awk -v i="$i" '/\/\/TESTARGS/,/\n/{j++}j==i+1{print; exit}' examples/ceed/$1.c | cut -d\  -f2- )")
    done
else
    allargs='{ceed_resource}'
fi

printf "1..$[3*${#backends[@]}*${#allargs[@]}]\n";

tmpfiles="${output} ${output}.out ${output}.diff ${output}.err SESSION.NAME"
trap 'rm -f ${tmpfiles}' EXIT

# test configurations loop
for ((j=0;j<${#allargs[@]};++j)); do
args=${allargs[$j]}
printf "# TESTARGS: $args\n"

# backends loop
for ((i=0;i<${#backends[@]};++i)); do
    i0=$((3*$i+1+j*3*${#backends[@]})) # return code
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

    # grep to pass test t11* on error
    if grep -F -q -e 'access' ${output}.err \
            && [[ "$1" = "t11"* ]] ; then
        printf "ok $i0 PASS - expected failure $1 $backend\n"
        printf "ok $i1 PASS - expected failure $1 $backend stdout\n"
        printf "ok $i2 PASS - expected failure $1 $backend stderr\n"
        continue
    fi

    # grep to pass test t303 on error
    if grep -F -q -e 'vectors incompatible' ${output}.err \
            && [[ "$1" = "t303"* ]] ; then
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

    # grep to skip tests t41*, ex1, and ex2 for OCCA
    #  This exception will be removed with the OCCA backend overhaul
    if grep -F -q -e 'OklPath' ${output}.err \
            && [[ "$1" = "t41"* || "$1" = "ex"* ]] ; then
        printf "ok $i0 # SKIP - gallery not supported $1 $backend\n"
        printf "ok $i1 # SKIP - gallery not supported $1 $backend stdout\n"
        printf "ok $i2 # SKIP - gallery not supported $1 $backend stderr\n"
        continue
    fi

    # grep to skip multigrid test for OCCA
    #  This exception will be removed with the OCCA backend overhaul
    if [[ "$backend" = *"occa" ]] \
            && [[ "$1" = "petsc-multigrid" ]] ; then
        printf "ok $i0 # SKIP - QFunction reuse not supported by $backend\n"
        printf "ok $i1 # SKIP - QFunction reuse not supported by $backend stdout\n"
        printf "ok $i2 # SKIP - QFunction reuse not supported by $backend stderr\n"
        continue
    fi

    if [ $status -eq 0 ]; then
        printf "ok $i0 $1 $backend\n"
    else
        printf "not ok $i0 $1 $backend\n"
    fi

    # stdout
    if [ -f tests/output/$1.out ]; then
        if diff -u tests/output/$1.out ${output}.out > ${output}.diff; then
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
done # backend loop
done # test configuration loop
