#!/bin/bash

output=$(mktemp $1.XXXX)

if ./"$@" > ${output}; then
    printf "ok $@\n"
else
    printf "not ok $@\n"
fi
if [ -f output/$1.out ]; then
    if diff -u output/$1.out ${output}; then
        printf "ok diff $1\n"
    else
        printf "not ok diff $1\n"
    fi
else
    while read line; do
        printf "+ ${line}\n"
    done < ${output}
fi
rm ${output}
