#!/bin/bash
#SBATCH --job-name=tspec
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/tspec-%j.out
#SBATCH --mail-user=ziming.liu@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p logs

export SLURM_EXPORT_ENV=ALL


module load Anaconda3/2024.02-1
source activate /mnt/parscratch/users/$USER/envs/tspec
module load cuDNN/8.9.2.26-CUDA-12.1.1
echo "module load completed successfully!"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BLOCK_SIZE=2000000
export BATCH_SIZE=1024
echo "export completed successfully!"
# 数据集根目录
export SPEC_ROOT=/mnt/parscratch/users/$USER/TSpec-LLM/3GPP-clean

# 启动作业
echo "Job Start!"
srun python src/tspec_metrics_2.py --wandb-project Tspec
# 运行完毕后，输出完成消息  
echo "Job completed successfully!"
