#!/bin/sh

#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -W group_list=gq42
#PBS -j oe

#------- Program execution -------#
cd ${PBS_O_WORKDIR}
uv run ./src/finetune_trainer.py
