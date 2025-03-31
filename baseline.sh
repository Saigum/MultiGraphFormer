#!/bin/bash
#SBATCH --job-name=example_job       # Job name
#SBATCH --nodes=1                    # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task (adjust if needed)
#SBATCH --cpus-per-task=40           # Request 40 CPUs for the task
#SBATCH --gres=gpu:4                 # Request 4 GPUs (using generic resource syntax)
#SBATCH --time=48:00:00              # Set max runtime to 48 hours
#SBATCH --nodelist=gnode073          # Specify the node list

rm -rf /scratch/saigum/ || echo "Failed to remove /scratch/saigum/"
conda deactivate
mkdir /scratch/saigum && cd /scratch/saigum
git clone "https://github.com/Saigum/MultiGraphFormer.git"
cd MultiGraphFormer
module load u18/python/3.7.4
uv venv pamnet --python 3.7.4 && source pamnet/bin/activate
uv pip install -r requirements.txt

python -u main_qm9.py --dataset 'QM9' --model 'PAMNet' --target=7 --epochs=200 --batch_size=32 --dim=128 --n_layer=6 --lr=1e-4