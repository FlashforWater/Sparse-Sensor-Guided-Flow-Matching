# S3FM Server Suite

This directory was produced by `python -m s3fm.run_server_suite`.

## Configuration

- observed fractions: `0.15`
- mask seeds: `0`
- source seeds: `0`
- num windows: `8`
- NFE: `10`
- S3FM lambda: `2.0`
- PF lambda sweep: `0,0.1,0.25`

## Aggregate

- completed cases: `1`
- M4b learned-source pass cases: `1/1`
- val-tuned PF attribution pass cases: `1/1`

Detailed rows are in `aggregate_by_mask.csv`.

## Directory Layout

- `obs_<fraction>/mask_<seed>/m4b_learned/`
- `obs_<fraction>/mask_<seed>/pf_val_sweep/`
- `obs_<fraction>/mask_<seed>/pf_test_val_tuned/`
