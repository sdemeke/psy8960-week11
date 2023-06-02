#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=127
#SBATCH --mem=20gb
#SBATCH -t 00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=demek004@umn.edu
#SBATCH -p amdsmall
cd ~/psy8960-week11
module load R/4.2.2-openblas
Rscript R/week11-cluster.R
