#!/bin/sh

#------ qsub option --------#
#PBS -q small-g
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -W group_list=gq42
#PBS -j oe

#------- Program execution -------#
uv run src/finetune_trainer.py
