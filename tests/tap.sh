#!/bin/bash

output=$(mktemp $1.XXXX)

printf "1..3\n"
if build/"$@" > ${output}.out 2> ${output}.err ; then
    printf "ok 1 $@\n"
else
    printf "not ok 1 $@\n"
fi
if [ -f output/$1.out ]; then
    if diff -u output/$1.out ${output}.out; then
        printf "ok 2 $1 stdout\n"
    else
        printf "not ok 2 $1 stdout\n"
    fi
elif [ -s ${output}.out ]; then
    printf "not ok 2 $1 stdout\n"
    while read line; do
        printf "# + ${line}\n"
    done < ${output}.out
else
    printf "ok 2 $1 stdout\n"
fi
if [ -s ${output}.err ]; then
    printf "not ok 3 $1 stderr\n"
    while read line; do
        printf "# + ${line}\n"
    done < ${output}.err
else
    printf "ok 3 $1 stderr\n"
fi
rm ${output} ${output}.out ${output}.err
