#!/bin/bash

output=$(mktemp $1.XXXX)

printf "1..9\n"

backends=(/cpu/self /cpu/occa /gpu/occa)

for ((i = 0; i < ${#backends[@]}; ++i)); do
    i0=$((3*$i+1)) # return code
    i1=$(($i0+1))  # stdout
    i2=$(($i0+2))  # stderr 
    backend=${backends[$i-1]}

    if build/$1 $backend > ${output}.out 2> ${output}.err ; then
        printf "ok $i0 $1 $backend\n"
    else
        printf "not ok $i0 $1 $backend\n"
    fi
    # stdout
    if [ -f output/$1.out ]; then
        if diff -u output/$1.out ${output}.out; then
            printf "ok $i1 $1 $backend stdout\n"
        else
            printf "not ok $i1 $1 $backend stdout\n" 
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
            printf "# + ${line}\n"
        done < ${output}.err
    else
        printf "ok $i2 $1 $backend stderr\n"
    fi
done
rm -f ${output} ${output}.out ${output}.err; 
