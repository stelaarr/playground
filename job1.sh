#!/bin/bash
#SBATCH --job-name=job1
#SBATCH --output=job1_output.log
#SBATCH --error=job1_error.log
#SBATCH --gpus 1                   # Request 1 GPU
#SBATCH -t 1-00:00:00

# Load any necessary modules (e.g., Anaconda if needed)
module load cuda/12.1

# Activate your Conda environment
source activate gals

# Launch Jupyter Notebook without opening a browser
jupyter nbconvert --to notebook --execute try_jupy.ipynb --output executed_try_jupy.ipynb
