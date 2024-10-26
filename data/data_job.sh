#!/bin/bash
#SBATCH --job-name=data_job     # Name of the job
#SBATCH --output=convert_mnist.out    # Output file
#SBATCH --error=convert_mnist.err     # Error file
#SBATCH --gpus 1 
#SBATCH --time=01:00:00    

# Load any necessary modules (if needed)
module load python/3.8  # Adjust this based on your Python version

# Activate your environment if using one
source activate gals

# Run the Python script
python /proj/sciml/users/x_stear/playground/data/convert_mnist.py

