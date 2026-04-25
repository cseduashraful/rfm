python RFM/v2/phase1/phase1_pipeline.py \
  --dataset rel-amazon \
  --output-dir RFM/v2/phase1/artifacts \
  --max-path-depth 8 \
  --sampling-row-threshold 50000 \
  --sample-size 50000



python v2/phase1/phase1_pipeline.py \
  --dataset rel-trial \
  --output-dir v2/phase1/artifacts \
  --max-path-depth 8 \
  --sampling-row-threshold 50000 \
  --sample-size 50000