# S3FM Implementation Workspace

This workspace contains the S3FM research/code project plus shared notes and
reference papers.

## Project

### S3FM

Path:

```text
s3fm/
```

Research direction:

```text
informative-source guided Flow Matching for sparse-sensor spatiotemporal
reconstruction
```

Main reading docs:

```text
S3FM_项目总览.md
S3FM_AI_IMPLEMENTATION_SPEC.md
S3FM_论文级研究说明.md
S3FM_论文草稿.md
s3fm/PAPER_EXPERIMENTS.md
s3fm/SERVER_RUN.md
```

Server entry points:

```bash
cd s3fm
bash scripts/train_paper_models.sh
bash scripts/run_paper_suite.sh
```

## Reference Papers

The PDFs in this folder are local references and are ignored by Git:

```text
Learning spatiotemporal dynamics with a .pdf
On the Guidance of Flow Matching.pdf
```

## Generated Outputs

Experiment outputs and checkpoints should stay out of version control:

```text
s3fm/experiments/
s3fm/results/
```

Keep `.gitkeep` placeholders, but do not commit generated checkpoints, plots,
CSV files, or smoke outputs unless there is a specific reason.

## Current Recommended Workflow

1. Use `s3fm/` for the informative-source S3FM research direction.
2. Run local smoke tests before uploading the project to the server.
3. Treat server outputs as generated artifacts first; summarize the results in
   Markdown after checking the aggregate CSV files.
