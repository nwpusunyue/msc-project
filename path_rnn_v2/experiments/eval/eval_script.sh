#!/usr/bin/env bash
cd /home/scimpian/msc-project/
basedir=$1
subdir=$2
eval_file_path=$3

echo "$basedir"
echo "$subdir"

if [ "$subdir" = "baseline" ]
then
    regex="emb_dim=([0-9]+)_l2=([0-9]+.[0-9]+)"
    for entry in ./textual_chains_of_reasoning_models/baseline/*
    do
        echo "Entry $entry"
        if [[ $entry =~ $regex ]]
        then
            dim="${BASH_REMATCH[1]}"
            l2="${BASH_REMATCH[2]}"
            PYTHONPATH=./ python -u ./path_rnn_v2/experiments/eval/textual_chains_of_reasoning_eval.py \
                "${entry}/model" \
                --emb_dim=${dim} \
                --l2=${l2}
        fi
    done
elif [ "$subdir" = "baseline_neighb" ]
then
    regex="emb_dim=([0-9]+)_l2=([0-9]+\.[0-9]+)_drop=([0-9]+\.[0-9]+|None)_neighb_dim=([0-9]+)_smoothing=([0-9]+\.[0-9]+|None)"
    for entry in ./textual_chains_of_reasoning_models/baseline_neighb/*
    do
        echo "Entry $entry"
        if [[ $entry =~ $regex ]]
        then
            dim="${BASH_REMATCH[1]}"
            l2="${BASH_REMATCH[2]}"
            drop="${BASH_REMATCH[3]}"
            neighb_dim="${BASH_REMATCH[4]}"
            smoothing="${BASH_REMATCH[5]}"
            echo "${dim} ${l2} ${drop} ${neighb_dim} ${smoothing}"

            PYTHONPATH=./ python -u ./path_rnn_v2/experiments/eval/textual_chains_of_reasoning_neighb_eval.py \
                "${entry}/model" \
                --emb_dim=${dim} \
                --l2=${l2} \
                --neighb_dim=${neighb_dim} \
                --label_smoothing=${smoothing}
        fi
    done
elif [ "$subdir" = "baseline_distance" ]
then
    regex="emb_dim=([0-9]+)_l2=([0-9]+\.[0-9]+)_drop=([0-9]+\.[0-9]+|None)"
    for entry in ./textual_chains_of_reasoning_models/baseline_distance/*
    do
        echo "Entry $entry"
        if [[ $entry =~ $regex ]]
        then
            dim="${BASH_REMATCH[1]}"
            l2="${BASH_REMATCH[2]}"
            drop="${BASH_REMATCH[3]}"
            echo "${dim} ${l2} ${drop}"

            PYTHONPATH=./ python -u ./path_rnn_v2/experiments/eval/distance_chains_of_reasoning_eval.py \
                "${entry}/model" \
                --emb_dim=${dim} \
                --l2=${l2}
        fi
    done
    elif [ "$subdir" = "baseline_distance" ]
then
    regex="emb_dim=([0-9]+)_l2=([0-9]+\.[0-9]+)_drop=([0-9]+\.[0-9]+|None)"
    for entry in ./textual_chains_of_reasoning_models/baseline_distance/*
    do
        echo "Entry $entry"
        if [[ $entry =~ $regex ]]
        then
            dim="${BASH_REMATCH[1]}"
            l2="${BASH_REMATCH[2]}"
            drop="${BASH_REMATCH[3]}"
            echo "${dim} ${l2} ${drop}"

            PYTHONPATH=./ python -u ./path_rnn_v2/experiments/eval/distance_chains_of_reasoning_eval.py \
                "${entry}/model" \
                --emb_dim=${dim} \
                --l2=${l2}
        fi
    done
elif [ "$subdir" = "baseline_truncated_relation" ] || [ "$subdir" = "lstm_truncated_relation" ] || [ "$subdir" = "attention_truncated_relation" ]
then
    regex="emb_dim=([0-9]+)_l2=([0-9]+\.[0-9]+)_drop=([0-9]+\.[0-9]+|None)"
    for entry in ./textual_chains_of_reasoning_models/${subdir}/*
    do
        echo "Entry $entry"
        if [[ $entry =~ $regex ]]
        then
            dim="${BASH_REMATCH[1]}"
            l2="${BASH_REMATCH[2]}"
            drop="${BASH_REMATCH[3]}"
            echo "${dim} ${l2} ${drop}"

            PYTHONPATH=./ python -u ./path_rnn_v2/experiments/${basedir}/${subdir}.py \
                --emb_dim=${dim} \
                --l2=${l2} \
                --testing \
                --model_path="${entry}/model" \
                --eval_file_path=${eval_file_path}
        fi
    done
fi