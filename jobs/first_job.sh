#!/bin/bash
#SBATCH -t 0:20:00
#SBATCH -N 1 -c 24

module load 2020
module load Python/3.8.2-GCCcore-9.3.0

cp -r $HOME/FinalAssignment $TMPDIR

cd $TMPDIR/FinalAssignment
python myscript.py input.dat

#mkdir -p SHOME/run3/results
#cp result.dat run3.log $HOME/run3/results