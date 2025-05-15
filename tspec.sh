#!/bin/bash
#SBATCH --job-name=tspec
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/tspec-%j.out
#SBATCH --mail-user=ziming.liu@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

# 确保日志目录存在
mkdir -p logs

# 将 SLURM 环境变量导出给子进程
export SLURM_EXPORT_ENV=ALL

# 加载 Conda 和 GPU 运行时
module load Anaconda3/2024.02-1
source activate /mnt/parscratch/users/$USER/envs/tspec
module load cuDNN/8.9.2.26-CUDA-12.1.1

# OpenMP 线程数
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BLOCK_SIZE=200000
export BATCH_SIZE=256

# 数据集根目录
export SPEC_ROOT=/mnt/parscratch/users/$USER/TSpec-LLM/3GPP-clean

# 启动作业
srun python tspec_metrics.py
# 运行完毕后，输出完成消息  
echo "Job completed successfully!"