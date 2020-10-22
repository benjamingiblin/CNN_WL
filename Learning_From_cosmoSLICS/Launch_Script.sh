#!/bin/bash
#SBATCH --partition=GPU
#SBATCH -n 1
#SBATCH -t 7-00:00          #### Time before job is ended automatically = 7days. 
#SBATCH --job-name=CNN_Nlayers
#SBATCH --requeue
#SBATCH --mail-type=ALL
#SBATCH --mem=15000                    #### The max amount of memory you expect your job to use at any one time.
                                       ### Probably leave this alone unless the job crashes due to memory issues.

nclayers=$1
python CNN_cosmoSLICS_PyTorch.py $nclayers
