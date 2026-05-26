# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project does not yet follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Cluster rejection-reasons hook.** Users can declare a module-level
  `rejection_reasons` counter (a `Counter` or `defaultdict(int)`) in their
  custom constraints file and increment it from inside `is_valid_cluster()`
  before each `return False`. Name Match picks the counter up after clustering
  and renders a sorted (reason, count, %) table in the matching report.
  Opt-in: constraints modules without the counter behave exactly as before.
  See `docs/source/match_setup.rst` ("Tracking rejection reasons") and
  `examples/clue_constraints.py` for a worked example.
- **NMSLIB filtering-efficiency logging in `Block`.** After the candidate-query
  loop, one summary block reports total raw neighbors from NMSLIB, candidates
  before and after dedup, acceptance rate, avg neighbors-per-query, and avg
  final-candidates-per-query. The first batch of the main index also writes
  raw `near_neighbors` to `details/near_neighbors_raw.parquet` for one-time
  debugging.
- **`max_demographic_values` parameter** in `evaluate_predictions()`. Caps the
  number of unique values per categorical demographic variable (default 10),
  skipping high-cardinality variables (e.g., zip code) at universe
  construction. Also gates per-demographic-subgroup INFO logs so run logs show
  clean base-universe metrics; stats are still saved for the matching report.
- **`examples/clue_constraints.py`** tracked as a reference implementation of
  the `rejection_reasons` hook, with four cluster-level constraints (size,
  dob uniqueness, age range, name+dob combos) and a `print_rejection_summary`
  helper.
- **`match_type_thresholds` parameter** in `fit_model` and `predict`. Optional
  per-match-type decision thresholds: when set to a dict with keys `exact_all`
  and/or `inexact_any`, candidate pairs are bucketed by whether every variable
  in `exact_match_variables` agrees exactly, and each bucket is gated against
  its own threshold (one for exact-on-everything pairs, another for pairs with
  at least one inexact field). When `optimize_threshold: True` is also set,
  `find_best_threshold` runs per bucket on the heldout evaluation set and the
  optima are written back into the same dict, so the prediction step
  transparently picks up the optimized values. The matching report renders the
  final per-bucket values and the threshold used for each universe in the
  pair-type performance table. Default `null` preserves the single-threshold
  behavior. See `docs/source/match_setup.rst`.

### Changed

- **`FitModel` uses chunked Parquet reads.** `get_train_eval_data()` now reads
  data rows via `pyarrow.ParquetFile.iter_batches(batch_size=10M)` with
  per-batch filter/sample/concat, replacing two full-frame
  `load_parquet_list()` calls. Avoids OOM on large datasets (validated at 7M
  defendant records, 11 GB data_rows parquet). No behavior change beyond
  memory profile.

### Fixed

- **`blocking_scheme` cosine-distance key typo.** Truncation of >2 cosine
  variables previously reassigned the wrong key (`'variable'` instead of
  `'variables'`), leaving the over-2 list intact and adding a stray key. The
  warning still fires; the slice now actually takes effect.
- **`reformat_dict()` handles nested dicts and numpy scalar types.** Previously
  only the top level of the dict was walked, so nested stats trees containing
  `np.integer` / `np.floating` values failed yaml dump in some environments.
