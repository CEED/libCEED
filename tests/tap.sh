#!/bin/bash

output=$(mktemp $1.XXXX)

function clean() { rm -f ${output} ${output}.out ${output}.err; }
function error() { clean; exit 1; }

for backend in /cpu/self /cpu/occa /gpu/occa
do
    echo -e \\tbuild/$@ $backend
    printf "\t1..3\n"
    if build/"$@" $backend > ${output}.out 2> ${output}.err ; then
    #if build/"$@" $backend 2> ${output}.err ; then
        printf "\t\tok 1 $@\n"
    else
        printf "\t\tnot ok 1 $@\n"
        error
    fi

    # continue before diff'ing outputs
    #continue

    if [ -f output/$1.out ]; then
        if diff -u output/$1.out ${output}.out; then
            printf "\t\tok 2 $1 stdout\n"
        else
            printf "\t\tnot ok 2 $1 stdout\n" 
            error
        fi
    elif [ -s ${output}.out ]; then
        printf "\t\tnot ok 2 $1 stdout\n"
        while read line; do
            printf "# + ${line}\n"
        done < ${output}.out
        error
    else
        printf "\t\tok 2 $1 stdout\n"
    fi
    if [ -s ${output}.err ]; then
        printf "\t\tnot ok 3 $1 stderr\n"
        while read line; do
            printf "# + ${line}\n"
        done < ${output}.err
        error
    else
        printf "\t\tok 3 $1 stderr\n"
    fi
done
clean
