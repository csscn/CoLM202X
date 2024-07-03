#!/bin/bash
#SBATCH -J Ds_test
#SBATCH -p second
#SBATCH -N 1
#SBATCH -n 96
#SBATCH --ntasks-per-node=96
#SBATCH -o DS-3.o%j
#SBATCH -e DS-3.e%j
ulimit -s unlimited
source /home/zhanghan/.cresm
#mpirun -n 64 ./mksrfdata.x Heihe_1km_ERA5_Ds.nml
#mpirun -n 16 ./mkinidata.x Heihe_1km_ERA5_Ds.nml
mpirun -n 65 ./colm.x  Heihe_1km_ERA5_Ds.nml : -n 13 python mpirecv.py 

