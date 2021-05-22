#!/bin/bash -l
# Batch script to run a GPU job on Myriad under SGE.

# 0. Force bash as the executing shell.
#$ -S /bin/bash

#2. Request half hour of wallclock time (format hours:minutes:second).
#$ -l h_rt=25:00:0

#3. Request 8 gigabyte of RAM (must be an integer)
#$ -l mem=16G

#4. Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=15G

# Request 16 cores.
#$ -pe smp 16

# set up the job array.
#$ -t 1-10
#5. Set the name of the job.
#$ -N O2_analysis

#6. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucesxc0/Scratch/output/crowded_spectator_analysis/O2/

# Parse parameter file to get variavles.
number=$SGE_TASK_ID
paramfile=/home/ucesxc0/Scratch/output/crowded_spectator_analysis/O2/params.txt
index="`sed -n ${number}p $paramfile | awk '{print $1}'`"
variable1="`sed -n ${number}p $paramfile | awk '{print $2}'`"

#8. activate the virtualenv
conda activate tf-2
#9.  load the cuda module
module unload compilers
module load compilers/gnu/4.9.2
#module load cuda/10.1.243/gnu-4.9.2
#module load cudnn/7.6.5.32/cuda-10.1
#10. Run job
./main.py -c person -i $variable1

conda deactivate

