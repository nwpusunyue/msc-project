#!/usr/bin/env bash
#!/usr/bin/env bash
cd /home/scimpian/msc-project/
basedir=$1
subdir=$2
entry=$3
root_dir=$4
eval_file_path=$5
pred_save_path=$6

regex="emb_dim=([0-9]+)_l2=([0-9]+\.[0-9]+)_drop=([0-9]+\.[0-9]+|None)_(.+)"
tokenizer_regex="tokenizer=([a-zA-Z]+)_"
neighb_regex="neighb_dim=([0-9]+)_"
masked_regex="masked=(True|False)"
paths_regex="paths=(all|shortest)_"
embd_regex="embd=(medline|medhop|random)_"
src_target_regex="srctarget=(True|False)_"

entry="${root_dir}/textual_chains_of_reasoning_models/${subdir}/${entry}"

echo "Entry $entry"
if [ -d "${entry}" ]; then
    if [[ $entry =~ $regex ]]
    then
        l2="${BASH_REMATCH[2]}"
        drop="${BASH_REMATCH[3]}"

        rest="${BASH_REMATCH[4]}"

        dim="${BASH_REMATCH[1]}"
        l2="${BASH_REMATCH[2]}"
        drop="${BASH_REMATCH[3]}"

        rest="${BASH_REMATCH[4]}"

        if [[ ${rest} =~ $tokenizer_regex ]]
        then
            tokenizer="${BASH_REMATCH[1]}"
        else
            tokenizer=punkt
        fi

        if [[ ${rest} =~ $neighb_regex ]]
        then
            neighb_dim=${BASH_REMATCH[1]}
        else
            neighb_dim="none"
        fi

        if [[ ${rest} =~ $masked_regex ]]
        then
            masked="${BASH_REMATCH[1]}"
        else
            masked="none"
        fi

        if [[ ${rest} =~ $paths_regex ]]
        then
            paths="${BASH_REMATCH[1]}"
        else
            paths="shortest"
        fi

        if [[ ${rest} =~ $embd_regex ]];
        then
            word_embd="${BASH_REMATCH[1]}"
        else
            if [ "${tokenizer}" = "genia" ];
            then
             word_embd="medline"
            else
              word_embd="medhop"
            fi
        fi

        if [ "$paths" = "all" ];
        then
            eval_batch_size=2
            limit=500
        else
            eval_batch_size=40
            limit=100
        fi

        if [ "$word_embd" = "medhop" ];
        then
            word_embd_path="medhop_word2vec_punkt_v2"
        elif [ "$word_embd" = "medline" ];
        then
            word_embd_path="medline_word2vec"
        else
            word_embd_path="random"
        fi

        if [[ ${rest} =~ $src_target_regex ]]
        then
            src_target="${BASH_REMATCH[1]}"
        else
            src_target="none"
        fi

        echo "$dim: ${dim} l2: ${l2} drop: ${drop} tok: ${tokenizer} embd: ${word_embd_path} neighb: ${neighb_dim} masked: ${masked} paths: ${paths} eval_size: ${eval_batch_size}"

        cmd="PYTHONPATH=./ python -u ./path_rnn_v2/experiments/${basedir}/${subdir}.py \
                    --emb_dim=${dim} \
                    --l2=${l2} \
                    --tokenizer=${tokenizer} \
                    --testing \
                    --word_embd_path=${word_embd_path} \
                    --model_path=\"${entry}/model\" \
                    --eval_file_path=${eval_file_path} \
                    --eval_batch_size=${eval_batch_size} \
                    --limit=${limit} \
                    --paths_selection=${paths} \
                    --train_pred_file_path=${pred_save_path}/train_${subdir}_emb_dim=${dim}_l2=${l2}_tok=${tokenizer}_embd=${word_embd}_lim=${limit}_paths=${paths}_masked=${masked}_srctarget=${src_target} \
                    --dev_pred_file_path=${pred_save_path}/dev_${subdir}_emb_dim=${dim}_l2=${l2}_tok=${tokenizer}_embd=${word_embd}_lim=${limit}_paths=${paths}_masked=${masked}_srctarget=${src_target} \
                    --test_pred_file_path=${pred_save_path}/test_${subdir}_emb_dim=${dim}_l2=${l2}_tok=${tokenizer}_embd=${word_embd}_lim=${limit}_paths=${paths}_masked=${masked}_srctarget=${src_target}"

        if [ "$neighb_dim" != "none" ];
        then
            cmd="$cmd \
                 --neighb_dim=$neighb_dim"
        fi

        if [ "$masked" = "True" ]
        then
            cmd="$cmd \
                 --masked"
        fi

        if [ "$src_target" = "True" ]
        then
            cmd="$cmd \
                 --source_target_only"
        fi

        echo $cmd

        eval $cmd
    fi
else
    echo Dir not found
fi
