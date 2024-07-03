#!/bin/bash
#SBATCH -J SW_DS_TEST
#SBATCH -p first
#SBATCH -N 1
#SBATCH -n 96
#SBATCH --ntasks-per-node=96
#SBATCH -o DS-3.o%j
#SBATCH -e DS-3.e%j
ulimit -s unlimited
make

