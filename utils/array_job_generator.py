#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os

import sys


def cartesian_product(dicts):
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c):
    path = '/home/scimpian/msc-project'
    command = 'PYTHONPATH=. anaconda-python3-gpu -u ' \
              '{}/path_rnn_v2/experiments/textual_chains_of_reasoning_eval.py ' \
              '--emb_dim={} ' \
              '--l2={} ' \
              '--dropout={}'.format(path, c['emb_dim'], c['l2'], c['dropout'])

    return command


def to_logfile(c, path):
    outfile = '{}/textual_chains_of_reasoning_medhop.{}.log'.format(path, summary(c))
    return outfile


def main(_):
    hyperparameters_space = dict(
        emb_dim=[100, 200, 300],
        l2=[0.0, 0.001],
        dropout=[0.0, 0.1]
    )

    configurations = cartesian_product(hyperparameters_space)

    path = '/home/scimpian/msc-project/qsub_logs'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/scimpian/'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    configurations = list(configurations)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        command_line = '{} >> {} 2>&1'.format(to_cmd(cfg), logfile)
        command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)
    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
#$ -l tmem=2G
#$ -l h_rt=1:00:00
#$ -P gpu
#$ -l gpu=1

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/scimpian/msc-project/
export PYTHONPATH=.

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}\n'.format(job_id, command_line))


if __name__ == '__main__':
    main(sys.argv[1:])
