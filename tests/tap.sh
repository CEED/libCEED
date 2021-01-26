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
declare -a names
if [ ${1::6} == "petsc-" ]; then
    allargs=$(grep -F //TESTARGS examples/petsc/${1:6}.c* | cut -d\  -f2- )
elif [ ${1::5} == "mfem-" ]; then
    allargs=$(grep -F //TESTARGS examples/mfem/${1:5}.c* | cut -d\  -f2- )
elif [ ${1::4} == "nek-" ]; then
    # get all test configurations
    numconfig=$(grep -F C_TESTARGS examples/nek/bps/${1:4}.usr* | wc -l)
    for ((i=0;i<${numconfig};++i)); do
      allargs+=("$(awk -v i="$i" '/C_TESTARGS/,/\n/{j++}j==i+1{print; exit}' examples/nek/bps/${1:4}.usr* | cut -d\  -f2- )")
    done
elif [ ${1::7} == "fluids-" ]; then
    numconfig=$(grep -F //TESTARGS examples/fluids/${1:7}.c* | wc -l)
    for ((i=0;i<${numconfig};++i)); do
      # get test name
      names+=("$(awk -v i="$i" '/\/\/TESTARGS/,/\n/{j++}j==i+1{print substr($1,18,length($1)-19)}' examples/fluids/${1:7}.c)")
      # get all test configurations
      allargs+=("$(awk -v i="$i" '/\/\/TESTARGS/,/\n/{j++}j==i+1{print; exit}' examples/fluids/${1:7}.c | cut -d\  -f2- )")
    done
elif [ ${1::7} == "solids-" ]; then
    allargs=$(grep -F //TESTARGS examples/solids/${1:7}.c* | cut -d\  -f2- )
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
printf "# Test Name: ${names[$j]}\n"
printf "# TESTARGS: $args\n"

# backends loop
for ((i=0;i<${#backends[@]};++i)); do
    i0=$((3*$i+1+j*3*${#backends[@]})) # return code
    i1=$(($i0+1))  # stdout
    i2=$(($i0+2))  # stderr
    backend=${backends[$i]}

    # Fluids and Solids QFunctions use VLA; not currently supported in OCCA
    if [[ "$backend" = *occa* && \
            ( "$1" = fluids-* || "$1" = solids-* || "$1" = t507* ) ]]; then
        printf "ok $i0 # SKIP - no support for VLA with $backend\n"
        printf "ok $i1 # SKIP - no support for VLA with $backend stdout\n"
        printf "ok $i2 # SKIP - no support for VLA with $backend stderr\n"
        continue;
    fi

    # Nek5000 integration not currently supported in OCCA
    if [[ "$backend" = *occa* && \
            ( "$1" = nek-* ) ]]; then
        printf "ok $i0 # SKIP - no support for Nek5000 with $backend\n"
        printf "ok $i1 # SKIP - no support for Nek5000 with $backend stdout\n"
        printf "ok $i2 # SKIP - no support for Nek5000 with $backend stderr\n"
        continue;
    fi

    # Navier-Stokes test problem too large for most CUDA backends
    if [[ "$backend" = *gpu* && "$backend" != /gpu/cuda/gen && \
            ( "$1" = fluids-* ) ]]; then
        printf "ok $i0 # SKIP - test problem too large for $backend\n"
        printf "ok $i1 # SKIP - test problem too large for $backend stdout\n"
        printf "ok $i2 # SKIP - test problem too large for $backend stderr\n"
        continue;
    fi

    # Run in subshell
    (build/$1 ${args/\{ceed_resource\}/$backend} || false) > ${output}.out 2> ${output}.err
    status=$?

    # grep to skip test if backend chooses to whitelist test
    if grep -F -q -e 'Backend does not implement' \
            ${output}.err ; then
        printf "ok $i0 # SKIP - not implemented $1 $backend\n"
        printf "ok $i1 # SKIP - not implemented $1 $backend stdout\n"
        printf "ok $i2 # SKIP - not implemented $1 $backend stderr\n"
        continue
    fi

    # grep to pass test t215 on error
    if grep -F -q -e 'access' ${output}.err \
            && [[ "$1" = "t215"* ]] ; then
        printf "ok $i0 PASS - expected failure $1 $backend\n"
        printf "ok $i1 PASS - expected failure $1 $backend stdout\n"
        printf "ok $i2 PASS - expected failure $1 $backend stderr\n"
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

    # grep to skip t506 for MAGMA, range of basis kernels limited for now
    if [[ "$backend" = *magma* ]] \
            && [[ "$1" = t506* ]] ; then
        printf "ok $i0 # SKIP - backend basis kernel not available $1 $backend\n"
        printf "ok $i1 # SKIP - backend basis kernel not available $1 $backend stdout\n"
        printf "ok $i2 # SKIP - backend basis kernel not available $1 $backend stderr\n"
        continue
    fi

    # grep to skip t318 for cuda/ref and MAGMA, Q is too large for these backends
    if [[ "$backend" = *magma* || "$backend" = *cuda/ref ]] \
            && [[ "$1" = t318* ]] ; then
        printf "ok $i0 # SKIP - backend basis kernel not available $1 $backend\n"
        printf "ok $i1 # SKIP - backend basis kernel not available $1 $backend stdout\n"
        printf "ok $i2 # SKIP - backend basis kernel not available $1 $backend stderr\n"
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
    elif [[ "$1" == t003* ]]; then
    # For t003, the output will vary widely; only checking stderr
        printf "ok $i1 $1 $backend stdout\n"
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
