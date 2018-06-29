#!/usr/bin/env bash
for lim in 500
do
  for tok in punkt treebank
  do
    for type in train dev train_mini
    do
        echo "${type} ${lim} ${tok}"
        PYTHONPATH=../ python -u \
        ../parsing/medhop_parser_pipeline.py \
        "../qangaroo_v1.1/medhop/${type}.json" \
        "../data/sentwise=F_cutoff=4_limit=${lim}_method=all_tokenizer=${tok}_medhop_${type}.json" \
        "../data/${type}_doc_store_${tok}.pickle" \
        --limit=${lim} \
        --cutoff=4 \
        --entities_path='./entities.txt' \
        --path_search_method=shortest \
        --tokenizer=${tok}
    done
  done
done
