#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -p gpu  # Partition
#SBATCH --mem=64G
#SBATCH -G 1  # Number of GPUs

#SBATCH --constraint=a40|l40s|a100
#SBATCH -t 2-00:00:00  # Job time limit

#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH -q long


module load conda/latest
conda activate ~/envs/llm







# #phase 2
# python v2/phase2/phase2_pipeline.py \
#   --dataset rel-trial \
#   --task study-adverse \
#   --phase1-artifacts-dir v2/phase1/artifacts \
#   --model-size 8b \
#   --dfs-train-vector-db-prune-schema \
#   --task-output-dir v2/phase2/artifacts
##phase 3
# python v2/phase3/phase3_pipeline.py   --dataset rel-trial   --task study-adverse \
#     --phase2-artifacts-dir v2/phase2/artifacts   --output-dir v2/phase3/artifacts \
#     --other-example-search-mode train_vector_db   --model-size 8b   --llm-batch-size 4 \
#     --history-length 10   --preprocess-batch-size 128 \
#     --enable-cpu-gpu-pipeline   --pipeline-prompt-queue-size 2 \
#     --print-predictions 



# #phase 2
# python v2/phase2/phase2_pipeline.py \
#   --dataset rel-f1 \
#   --task driver-position \
#   --phase1-artifacts-dir v2/phase1/artifacts \
#   --model-size 8b \
#   --dfs-train-vector-db-prune-schema \
#   --task-output-dir v2/phase2/artifacts
# # #phase 3
python v2/phase3/phase3_pipeline.py   --dataset rel-f1   --task driver-position   \
    --phase2-artifacts-dir v2/phase2/artifacts   --output-dir v2/phase3/artifacts       \
    --other-example-search-mode train_vector_db   --model-size 8b   --llm-batch-size 2       \
    --history-length 10   --preprocess-batch-size 64       \
    --enable-cpu-gpu-pipeline   --pipeline-prompt-queue-size 2     \
    --print-predictions 


python v2/phase3/phase3_pipeline.py   --dataset rel-trial   --task study-adverse   \
    --phase2-artifacts-dir v2/phase2/artifacts   --output-dir v2/phase3/artifacts       \
    --other-example-search-mode train_vector_db   --model-size 8b   --llm-batch-size 8       \
    --history-length 10   --dfs-batch-size 128 --preprocess-batch-size 128       \
    --enable-cpu-gpu-pipeline   --pipeline-prompt-queue-size 2     \
    --print-predictions 



# #phase 2 profile
# python v2/phase2/phase2_pipeline.py \
#   --dataset rel-f1 \
#   --task driver-position \
#   --phase1-artifacts-dir v2/phase1/artifacts \
#   --model-size 8b \
#   --max-rounds 1 \
#   --profile-dfs-train-vector-db \
#   --profile-dfs-train-vector-db-chunks 2 \
#   --profile-dfs-train-vector-db-early-stop \
#   --dfs-train-vector-db-prune-schema \
#   --task-output-dir v2/phase2/profile_runs/driver-position-pruned2



python v2/phase1/phase1_pipeline.py \
  --dataset rel-trial \
  --output-dir v2/phase1/artifacts \
  --max-path-depth 8 \
  --sampling-row-threshold 50000 \
  --sample-size 50000

