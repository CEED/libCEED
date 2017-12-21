#!/bin/bash

output=$(mktemp $1.XXXX)

function clean() { rm -f ${output} ${output}.out ${output}.err; }
function error() { clean; exit 1; }
function quit0() { clean; exit 0; }

printf "1..3\n"
for backend in /cpu/self /cpu/occa
do
    echo build/$@ $backend
    if build/"$@" $backend > ${output}.out 2> ${output}.err ; then
        printf "ok 1 $@\n"
    else
        printf "not ok 1 $@\n"
        error
    fi
    
    if [ -f output/$1.out ]; then
        if diff -u output/$1.out ${output}.out; then
            printf "ok 2 $1 stdout\n"
        else
            printf "not ok 2 $1 stdout\n" 
            error
        fi
    elif [ -s ${output}.out ]; then
        printf "not ok 2 $1 stdout\n"
        while read line; do
            printf "# + ${line}\n"
        done < ${output}.out
        error
    else
        printf "ok 2 $1 stdout\n"
    fi
    if [ -s ${output}.err ]; then
        printf "not ok 3 $1 stderr\n"
        while read line; do
            printf "# + ${line}\n"
        done < ${output}.err
        error
    else
        printf "ok 3 $1 stderr\n"
    fi
done
clean
