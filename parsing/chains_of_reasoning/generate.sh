#!/usr/bin/env bash

cd /home/scimpian/msc-project/
for entry in /cluster/project2/mr/scimpian/chains_of_reasoning_data/*
do
    echo "entry $entry"
    regex="/cluster/project2/mr/scimpian/chains_of_reasoning_data/(.+)"

    if [[ $entry =~ $regex ]]
    then
        rel="${BASH_REMATCH[1]}"
        echo "$rel"
        cmd="PYTHONPATH=./ python -u ./parsing/chains_of_reasoning/chains_of_reasoning_parser.py --relation ${rel}"
        eval $cmd
    fi
done
