#!/bin/bash
#SBATCH --job-name=convert          # Job name
#SBATCH --output=$TMP_DIR/mgg/convert_output_%j.txt     # Standard output and error log
#SBATCH --error=$TMP_DIR/mgg/convert_error_%j.txt       # Error log
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --gres=gpu:8                   # Request one GPU
#SBATCH --mem=1500G                      # Memory per node
#SBATCH --time=01:00:00                # Time limit hrs:min:sec
#SBATCH --partition=a100               # Partition to run the job on

srun python dataset_to_t10n.py
