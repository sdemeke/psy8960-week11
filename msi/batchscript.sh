#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --mem=10gb
#SBATCH -t 00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=demek004@umn.edu
#SBATCH -p amdsmall
cd ~/psy8960-week11/R
module load R/4.2.2-openblas
Rscript cluster_test.R
