#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=120gb
#SBATCH -t 00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=demek004@umn.edu
#SBATCH -p amdsmall
cd ~/psy8960-week11
module load R/4.3.0-openblas
Rscript R/week11-cluster.R
