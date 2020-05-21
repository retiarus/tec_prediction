#!/bin/bash
#SBATCH -J testTensorFlow
#SBATCH --partition=gpuq
#SBATCH -N 1
#SBATCH --ntasks-per-node=30

# If modules are needed by the script, then source modules environment:
. /etc/profile.d/modules.sh
. $HOME/anaconda/bin/activate
conda activate dscience
echo $PATH
module add cuda80/toolkit/8.0.61
# Work directory
workdir="$SLURM_SUBMIT_DIR"

# Full path to application + application name
application="$(which python)"

# Run options for the application
options="$workdir/main.py --seq_length_min=7200 --step_min=30 --window_train=144 --window_predict=96 --batch_size=50 --epochs=100 --pytorch True --data scin  --source=/home/pedro/resized_type_1/"

###############################################################
### You should not have to change anything below this line ####
###############################################################
# change the working directory (default is home directory)
cd $workdir
echo Running on host $(hostname)
echo Time is $(date)
echo Directory is $(pwd)
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo $SLURM_JOB_NODELIST

command="$application $options"

# Run the executable
ulimit -s unlimited
export OMP_NUM_THREADS=20
echo Running $command
time $command
