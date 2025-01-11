#!/bin/sh
#SBATCH -J inference_
#SBATCH -p normal.q
#SBATCH --nice=1000
#SBATCH --nodelist star028
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -o %x.out
#SBATCH -e %x.err


# Hidden dimension
python -u ongkoo_score.py -m us -d 2025-01-11  > ongkoo_us.log
