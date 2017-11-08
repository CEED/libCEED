#!/bin/bash

if ./"$@"; then
    printf "ok $@\n"
else
    printf "not ok $@\n"
fi
