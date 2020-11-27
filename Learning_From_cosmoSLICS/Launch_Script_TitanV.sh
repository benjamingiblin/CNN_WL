#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:TitanV:1 ####request GPU
#SBATCH -n 1
#SBATCH -t 7-00:00          #### Time before job is ended automatically = 7days. 
#SBATCH --job-name=CNN_Nlayers
#SBATCH --requeue
#SBATCH --mail-type=ALL
#SBATCH --mem=32G                    #### The max amount of memory you expect your job to use at any one time.
                                       ### Probably leave this alone unless the job crashes due to memory issues.

nclayers=$1
conv1_filter=$2
RUN=$3
python CNN_cosmoSLICS_PyTorch.py $nclayers $conv1_filter $RUN
