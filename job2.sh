#!/bin/bash
#SBATCH --job-name=clip_mnist     # Job name
#SBATCH --output=clip_mnist.out   # Output file
#SBATCH --error=clip_mnist.err    # Error log
#SBATCH --gpus 1                           # Request 1 GPU
#SBATCH -t 1-00:00:00                      # Time limit (1 day)

# Load necessary modules (assuming you are using Mambaforge)
module load Mambaforge/23.3.1-1-hpc1-bdist

# Activate your conda environment
source activate gals  # Use your environment name

# Check if the Python file exists before running
python /proj/sciml/users/x_stear/playground/try2.py
