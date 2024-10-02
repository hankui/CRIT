#!/bin/bash
#SBATCH --job-name=lc_run_gpu # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem 507904M # in unit MB 507904M=496G; is the max of the innovator 
#SBATCH --gres=gpu:2   # Number of GPUs per node (MAX is 2 for SDSU tested on Hank on Apr 19 2021)
#SBATCH --ntasks-per-node=40 # CPUs per node (MAX=40 for CPU nodes and 80 for GPU)
#SBATCH --output=out-%j-%N.log # Standard output (log file)
#SBATCH --partition=gpu # Partition/Queue
#SBATCH --time=2-00:00:00 # Maximum walltime 192=8-days
#SBATCH --mail-type=ALL #see https://slurm.schedmd.com/sbatch.html for all options.
#SBATCH --mail-user=hankui.zhang@sdstate.edu

# cd /mmfs1/home/jacks.local/hankui.zhang/mycode.research/LCMAP/LCMAP_run/CRIT_paper
# sbatch my.transformer.gpu.innovator.map.slurm
# squeue -l | grep hankui


# nvidia-smi


# module load rasterio/1.3.9
module load rasterio/tensorflow # changed on Mar 5, 2024 

date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;

python Pro_load_model_run_tile_v2_5.py > classification_output.txt 

wait 
date_end=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
time_diff=`echo "scale=2;($date_end-$date_start+0.01)*1.0/3600.0"|bc`;
date
echo "$time_diff hours used";
