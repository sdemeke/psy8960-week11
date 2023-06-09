#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --mem=244gb
#SBATCH -t 00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=demek004@umn.edu
#SBATCH -p amdsmall
cd ~/psy8960-week11
module load R/4.2.2-openblas
Rscript R/week11-cluster.R
