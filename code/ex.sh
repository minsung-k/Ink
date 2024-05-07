#!/bin/sh
#SBATCH --job-name=train_w3_half           # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=minsung.kang@ufl.edu     # Where to send mail	

#SBATCH --nodes=1                   # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=3           # Number of CPU cores per task
#SBATCH --mem=9gb                   # Job memory request
#SBATCH --time=200:00:00              # Time limit hrs:min:sec
#SBATCH --output=/blue/xxian/minsung.kang/Ink/logs/%j.log

pwd; hostname; date

module purge
module load tensorflow
python ex.py 3 5 50 # data_num weight epochs 
date

# 4/17
# fixed but shuffle order
# weight = 0

# 4/14

# run more shuffled case,

# weight = 0.25, 0.5, 0.75, 1.25, 1.5 

# data_num  Shuffled 1 cases
# weight = 0.5, 1.5, 0.25, 0.75, 1.25
# epoch = 100
# total 5 = 5 cases

# data_num  0 ~ 11 12 cases
# weight = 1.5
# epoch = 100

# total 12 * 1 = 12 cases


# 4/12 
# run the new code with model_0412.py module. change dataset

# data_num 0 ~ 3 4 cases
# weight = 0,1
# epoch = 100

# total 4 * 2 = 8 cases

# If it works, do weight with 3,5,7,9, ...

# ---------------------------------------------

# 4/11 do the previous task again

# more diverse case
# data_num 4 ~ 11 8 cases
# weight = 0,1,3
# epoch 100

# total 8 * 3 = 24 cases 

# Random shuffle
# weight = 0,1,3

# If it works, do weight with 5,7,9, ...

# ---------------------------------------------

# 4/10 data_num epoch weight

# more diverse case
# data_num 4 ~ 11 8 cases
# weight = 0,1,3
# epoch 100

# total 8 * 3 = 24 cases 

# Random shuffle
# weight = 0,1,3

# ---------------------------------------------

# 4/8 data_num epoch weight

# 0(16), 1(27), 2(38), 3(45) 
# weight = 0,1,3,5,7,9,11,13,15
# epoch 100

# total 4 * 9 = 36 cases

# ---------------------------------------------

# 4/5 epoch weight

# 16, 27, 38, 45 total 4 * 6 = 24 cases
# 100 0
# 100 1
# 100 5
# 100 10
# 100 20
# 100 50

# 20 0
# 20 1
# 20 2
# 20 5
# 20 10
# 20 20

# ---------------------------------------------

# 4/2 epoch weight

# 100 0
# 100 1
# 100 5
# 100 0.5
