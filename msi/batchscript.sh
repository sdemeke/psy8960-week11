#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH -t 00:00:10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=demek004@umn.edu
#SBATCH -p amdsmall
cd ~/psy8960-week11/R
module load R/4.2.2-openblas
Rscript week11-cluster.R
