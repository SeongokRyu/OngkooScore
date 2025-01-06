#!/bin/sh
#SBATCH -J inf_gin_0.25_max_10.0
#SBATCH -p normal.q
#SBATCH --nice=1000
#SBATCH --nodelist star029
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -o %x.out
#SBATCH -e %x.err


# Hidden dimension
python -u ongkoo_score.py -m us -d 2025-01-04  > ongkoo_us.log
