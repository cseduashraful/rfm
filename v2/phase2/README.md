# Phase 2 Pipeline

This folder contains Phase 2 implementation for task-specific meta-path planning and ranking.

## Run

From repo root (with `~/envs/llm` activated):

```bash
python RFM/v2/phase2/phase2_pipeline.py \
  --dataset rel-f1 \
  --task driver-position \
  --phase1-artifacts-dir RFM/v2/phase1/artifacts \
  --output-dir RFM/v2/phase2/artifacts \
  --model-size 8b \
  --max-rounds 5
```

Optional LLM path rescoring (recommended):

```bash
python RFM/v2/phase2/phase2_pipeline.py \
  --dataset rel-f1 \
  --task driver-position \
  --phase1-artifacts-dir RFM/v2/phase1/artifacts \
  --output-dir RFM/v2/phase2/artifacts \
  --model-size 8b \
  --max-rounds 5 \
  --llm-path-score-max-candidates 220 \
  --llm-path-score-batch-size 20
```

## Inputs

Required Phase 1 artifacts under:

- `<phase1-artifacts-dir>/<dataset>/schema.json`
- `<phase1-artifacts-dir>/<dataset>/stats.json`
- `<phase1-artifacts-dir>/<dataset>/semantics.json`
- `<phase1-artifacts-dir>/<dataset>/semantic_context_graph.json`
- `<phase1-artifacts-dir>/<dataset>/path_catalog.json`
- `<phase1-artifacts-dir>/<dataset>/safety_rules.json`

## Outputs

For each dataset/task, the pipeline writes under:

- `<output-dir>/<dataset>/<task>/`

Primary outputs include:

- `task_spec.json`: final selected meta-paths, relevance scores, and attribute aggregation guidance.
- `round_trace.json`: solver/critic per-round trace and issue progression.
- `attribute_importance.json`: global ranking of important attributes across selected paths.

## Notes

- Phase 2 is task-aware and dataset-aware.
- Uses a solver-critic loop to iteratively improve selected meta-paths.
- Prints top selected task-specific meta-paths by relevance score in the console.
- Uses Phase 1 semantic context graph during candidate enumeration.

## Key Flags

- `--max-rounds`: maximum solver-critic iterations.
- `--model-size`: local LLM size selector used by the shared loader.
- `--llm-path-score-max-candidates`: cap for candidate paths sent to LLM rescoring.
- `--llm-path-score-batch-size`: batch size for path-rescoring prompts.

## Code Structure

- `phase2_pipeline.py`: orchestration (candidate build, solver-critic loop, finalization).
- `phase2_policy.py`: path sanitization/finalization and attribute aggregation planning.
- `phase2_rescoring.py`: LLM-based candidate rescoring and robust parse/fallback handling.
- `phase2_prompts.py`: solver and critic prompt builders.
- `phase2_models.py`: Pydantic models for solver/critic structured outputs.
- `phase2_validation.py`: structural validation helpers.
- `phase2_llm.py`: local model loading and generation wrappers.
- `phase2_io.py`: JSON IO and artifact write/read helpers.

## Technical Report

- `technical_report.tex` describes Phase 2 design, scoring policy, solver-critic flow, and extension points.
