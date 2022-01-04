#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --mincpus=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G

#SBATCH --time=4-00:00:00
#SBATCH --output=rm_m7_r3.txt
#SBATCH --mail-user=ansh.puvvada@students.iiit.ac.in
#SBATCH --mail-type=ALL


eval "$(conda shell.bash hook)"
conda activate venv

python -m app.auto_trainer rm_m7_r3
