#!/bin/bash
#SBATCH --job-name="firsttest"
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=00:01:00
#SBATCH --partition=thin

module load 2020
module load Python/3.8.2-GCCcore-9.3.0

cp -r $HOME/FinalAssignment $TMPDIR

cd $TMPDIR/FinalAssignment
python myscript.py input.dat

#mkdir -p SHOME/run3/results
#cp result.dat run3.log $HOME/run3/results