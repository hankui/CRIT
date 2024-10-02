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
# sbatch my.transformer.gpu.innovator.CRIT.slurm
# squeue -l | grep hankui


# nvidia-smi


# module load rasterio/1.3.9
module load rasterio/tensorflow # changed on Mar 5, 2024 
# 
# conda init bash 
date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
which python
version=4_1 # fixed test 
date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
# which python

rm -rf ./tmp/tb_log/*


SLEEP=60

MAXGPU=2
version=9_31 # daily model - daily with sensor + more sensors + x and y locations
version=9_21 # daily model - daily with sensor + more sensors + x and y locations
version=11_69 # daily model - daily with sensor + more sensors + x and y locations
# conda init bash 
date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;

gpui=0
layer=4
year=1985
year=""
# for learning_rate in 0.001 0.0001;
# for learning_rate in 0.001 0.01;
for learning_rate in 0.001;
do
# for L2 in 1e-4 1e-3;
# for L2 in 1e-4 1e-5;
for L2 in 1e-4;
do
# for epoch in 100;
for epoch in 70;
# for epoch in 10;
do
# for year in 1985 2000;
# for year in 1985 2000 2018 " ";
for year in " ";
# for year in 2000 2018 " ";
do	
for BATCH in 512;
do
# for DROP in 0 0.1 0.2 0.3 0.4 0.5;
for DROP in 0.1;
do
	# learning_rate=0.0001; method=1; epoch=70; L2=0; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 
	# learning_rate=0.0001; method=2; L2=1e-5; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 
	method=2; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 
	echo "python Pro_2d1d_CNN_v${version}.py ${learning_rate} ${epoch} ${method} ${L2} ${BATCH} 0 ${year} "
	# python  Pro_lcmap_2d1d_CNN_v${version}.py ${learning_rate} ${epoch} ${method} ${L2} ${BATCH} 0 ${year} > B${BATCH}.rate${learning_rate}.e${epoch}.L${L2}.v${version}.y${year} & 
	python  Pro_lcmap_CRIT_v${version}.py ${DROP} ${epoch} ${method} ${learning_rate} ${L2} ${BATCH} ${gpui} ${year} > inno.DROP${DROP}.rate${learning_rate}.b${BATCH}.e${epoch}.L${L2}.v${version}.y${year} & 
	gpui=$((${gpui}+1))
	if [ ${gpui} -ge ${MAXGPU} ]; then 
		wait 
		gpui=0
		echo "gpui=0"
	fi
done 
done 
done 
done 
done 
done

wait 
date_end=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
time_diff=`echo "scale=2;($date_end-$date_start+0.01)*1.0/3600.0"|bc`;
date
echo "$time_diff hours used";


version=7_8 # daily model - 16-day with no sensor
version=7_9 # daily model - 16-day with sensor
version=8_0 # daily model - daily with sensor
version=8_1 # daily model - daily with sensor
version=8_2 # daily model - daily with sensor + more parameters to try
version=8_3 # daily model - daily with sensor
version=8_5 # daily model - daily with sensor + more sensors
version=8_7 # daily model - daily with sensor + more sensors + x and y locations
version=8_77 # daily model - daily with sensor + more sensors + x and y locations
version=8_8 # daily model - daily with sensor + more sensors + x and y locations
version=8_9 # daily model - daily with sensor + more sensors + x and y locations

