#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -p gpu  # Partition
#SBATCH --mem=64G
#SBATCH -G 1  # Number of GPUs

#SBATCH --constraint=a40|l40s|l4|a100
#SBATCH -t 2-00:00:00  # Job time limit

#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH -q long


module load conda/latest
conda activate ~/envs/llm

python zero_shot_profile.py   --dataset rel-f1   --task driver-position   --model-size 3b \
   --history-sampling-strategy recent_min_overlap \
   --use-dfs   --print-log \
   --llm-batch-size 2   --dfs-batch-size 64 --pipeline-batch-size 32  \
   --context-workers 4   --overlap-prep-llm   --prep-queue-size 4   \
   --debug --max-items 50 \
   --context-components dfs_table \
   --self-example-max-count 10 \
   --other-neighbor-entity-count 3 \
   --other-neighbor-history-count 3 \
   --pred-only

# python zero_shot_profile.py --dataset rel-f1 --task driver-position \
#   --model-size 1b \
#   --use-dfs --debug --max-items 200 \
#   --llm-batch-size 1 --pipeline-batch-size 8 --context-batch-size 8 --dfs-batch-size 64\
#   --overlap-prep-llm --bulk-history-query --prep-queue-size 8 \
#   --profile-slowest-items 10 --enable-torch-profiler

# python zero_shot_profile.py \
#   --dataset rel-f1 \
#   --task driver-position \
#   --model-size 3b \
#   --use-dfs \
#   --pred-only \
#   --llm-batch-size 16 \
#   --dfs-batch-size 64 \
#   --pipeline-batch-size 32 \
#   --context-workers 4 \
#   --overlap-prep-llm \
#   --prep-queue-size 4 \
#   --debug --max-items 256


python zero_shot_profile.py \
  --dataset rel-f1 \
  --task driver-position \
  --model-size 1b \
  --use-dfs \
  --llm-batch-size 16 \
  --dfs-batch-size 64 \
  --pipeline-batch-size 32 \
  --context-workers 4 \
  --overlap-prep-llm \
  --prep-queue-size 4 \
  --debug --max-items 2


# python zero_shot_profile.py   --dataset rel-f1   --task driver-position   --model-size 1b   --use-dfs   --print-log   --llm-batch-size 1   --dfs-batch-size 64   --pipeline-batch-size 32   --context-workers 4   --overlap-prep-llm   --prep-queue-size 4   --debug --max-items 10 --pred-only


# python zero_shot.py \
#   --dataset rel-f1 \
#   --task driver-position \
#   --model-size 3b \
#   --use-dfs \
#   --pred-only \
#   --llm-batch-size 16 \
#   --dfs-batch-size 64 \
#   --pipeline-batch-size 32 \
#   --context-workers 4 \
#   --overlap-prep-llm \
#   --prep-queue-size 4


# python zero_shot.py --dataset rel-f1 --task driver-position \
#   --model-size 8b \
#   --use-dfs \
#   --pipeline-batch-size 8 \
# #   --print-log --pred-only \

# # python zero_shot.py --dataset rel-f1 --task driver-position \
# #   --model-size 3b \
# #   --print-log --pred-only \



# # python zero_shot.py --dataset rel-f1 --task results-position \
# #   --model-size 1b \
# #   --print-log --pred-only \

# # python zero_shot.py --dataset rel-f1 --task results-position \
# #   --model-size 3b \
# #   --print-log --pred-only \


# # python zero_shot.py --dataset rel-f1 --task qualifying-position \
# #   --model-size 3b \
# #   --print-log --pred-only \

# # python zero_shot.py --dataset rel-amazon --task user-ltv \
# #   --model-size 3b --num-hops 1 \
# #   --print-log --pred-only \
# #1hop 1b 54963920 3b 54963924

# # python zero_shot.py --dataset rel-amazon --task item-ltv \
# #   --model-size 3b  --num-hops 1 \
# #   --print-log --pred-only \
# #1hop 1b 54963932 3b 54963935

# #1b-54962558

# # python zero_shot.py --dataset rel-amazon --task review-rating \
# #   --model-size 3b \
# #   --print-log --pred-only \



# # python zero_shot.py --dataset rel-f1 --task driver-position --model-size 1b --print-log --debug --use-dfs
# # python zero_shot.py --dataset rel-amazon --task user-ltv --model-size 8b --print-log --debug --use-dfs  --pred-only --max-items 200

# # python zero_shot.py --dataset rel-f1 --task driver-position --model-size 8b --use-dfs  