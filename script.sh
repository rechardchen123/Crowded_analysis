#!/bin/bash -l
# Batch script to run a GPU job on Myriad under SGE.

# 0. Force bash as the executing shell.
#$ -S /bin/bash

#2. Request half hour of wallclock time (format hours:minutes:second).
#$ -l h_rt=10:00:0

#3. Request 8 gigabyte of RAM (must be an integer)
#$ -l mem=16G

#4. Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=15G

# Request 16 cores.
#$ -pe smp 8

#5. Set the name of the job.
#$ -N FA_H-205

#6. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucesxc0/Scratch/output/wembley_stadium_analysis/Wembley_stadium_spectator_behavior_analysis_FA_H-205/



#8. activate the virtualenv
conda activate tf-2
#9.  load the cuda module
module unload compilers
module load compilers/gnu/4.9.2
#module load cuda/10.1.243/gnu-4.9.2
#module load cudnn/7.6.5.32/cuda-10.1
#10. Run job
./main.py -c person -i ./video/FA_Semi_finals/H-205.mp4

conda deactivate

