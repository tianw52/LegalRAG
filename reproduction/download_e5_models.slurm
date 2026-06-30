#!/bin/bash
#SBATCH --job-name=dl-e5-models
#SBATCH --account=def-jieliang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# NOTE: This script is kept for reference only.
# E5 models are now pre-downloaded on the LOGIN NODE before job submission.
# Run  python3 reproduction/preflight_check.py  to verify everything is cached.
#
# DO NOT use this script to download models on compute nodes — compute nodes
# have no internet access on this cluster.
echo "This download script is now a no-op."
echo "E5 models must be pre-downloaded on the login node."
echo "Run: python3 reproduction/preflight_check.py"
