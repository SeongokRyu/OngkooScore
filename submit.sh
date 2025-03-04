#!/bin/
#SBATCH -J inf
#SBATCH -p gpu-large.q
#SBATCH --nice=1000
#SBATCH -N 1
#SBATCH -n 96
#SBATCH -o %x.out
#SBATCH -e %x.err


# Hidden dimension
python -u ongkoo_score.py -m us -d 2025-02-12  > ongkoo_us.log
