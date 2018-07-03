#!/usr/bin/env bash
cd /home/scimpian/msc-project/
regex="emb_dim=([0-9]+)_l2=([0-9]+.[0-9]+)"

for entry in ./textual_chains_of_reasoning_models/*
do
    echo "$entry"
    if [[ $entry =~ $regex ]]
    then
        dim="${BASH_REMATCH[1]}"
        l2="${BASH_REMATCH[2]}"
        PYTHONPATH=./ python -u ./path_rnn_v2/experiments/textual_chains_of_reasoning_eval.py \
            "${entry}/model" \
            --emb_dim=${dim} \
            --l2=${l2}
    fi
done