python phase3/phase3_pipeline.py \
  --dataset rel-trail \
  --task study-adverse \
  --phase2-artifacts-dir phase2/artifacts \
  --output-dir phase3/artifacts \
  --model-size 1b \
  --llm-batch-size 1 \
  --history-length 10


python RFM/v2/phase3/phase3_pipeline.py \
  --dataset rel-trial \
  --task study-adverse \
  --phase2-artifacts-dir RFM/v2/phase2/artifacts \
  --output-dir RFM/v2/phase3/artifacts \
  --model-size 1b \
  --llm-batch-size 1 \
  --history-length 10


python RFM/v2/phase3/phase3_pipeline.py \
  --dataset rel-f1 \
  --task driver-position \
  --phase2-artifacts-dir RFM/v2/phase2/artifacts \
  --output-dir RFM/v2/phase3/artifacts \
  --model-size 3b \
  --llm-batch-size 1 \
  --history-length 10