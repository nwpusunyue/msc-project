#!/usr/bin/env bash
ls -1t ./textual_chains_of_reasoning_logs/acc_* | head -1 | xargs echo
ls -1td ./textual_chains_of_reasoning_logs/run_*/ | head -1 | xargs echo
ls -1td ./textual_chains_of_reasoning_models/run_*/ | head -1 | xargs echo

ls -1t ./textual_chains_of_reasoning_logs/acc_* | head -1  | xargs rm -rf
ls -1td ./textual_chains_of_reasoning_logs/run_*/ | head -1 | xargs rm -rf
ls -1td ./textual_chains_of_reasoning_models/run_*/ | head -1 | xargs rm -rf
