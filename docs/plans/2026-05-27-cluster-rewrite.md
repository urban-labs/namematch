# namematch Cluster Rewrite — Aggregated Constraint State + JIT

**Date:** 2026-05-27
**Status:** Draft (pending current threshold-sweep completion)
**Target branch:** `feat/cluster-aggregated-state` off `enhanced-matching-report`
**Upstream target:** PR into `urban-labs/namematch`

## Problem

The clustering step in `namematch.cluster.cluster_potential_edges()` processes potential edges sequentially via greedy union-find. For each edge that would merge two clusters of combined size > 2, the loop builds a candidate-merge DataFrame and runs the user's `is_valid_cluster(df, phat)` constraint function on it. Per-edge cost is dominated by pandas overhead:

| Operation | Where | ~Cost per call |
|---|---|---|
| `cluster_info.iloc[[ci_ix_map[r] for r in new_cluster]].copy()` | cluster.py:670 | 100–300 µs |
| `cluster_info.index.isin(cluster_1)` | cluster.py:672 | 50–100 µs |
| `cluster_info.index.isin([rid1, rid2])` | cluster.py:675 | 50–100 µs |
| 4 pandas ops in `clue_constraints.is_valid_cluster` (`nunique`, `astype(int).max() - .min()`, `drop_duplicates()`, `shape[0]`) | clue_constraints.py:34 | 200–500 µs |

At 43M edges (our `inexact_any=0.20` run) with ~30% taking the slow path, this is ≈2–3 hours of pure pandas overhead before any actual clustering work. The constraint logic itself (size cap, dob set, age range, name+dob combo cap) is set operations and min/max — work that should take ~1 µs per merge in numpy, not ~500 µs in pandas.

## Design

Replace the per-edge pandas DataFrame with **per-cluster aggregated state** that namematch maintains as plain Python/numpy data structures, and let constraints query the aggregated state directly. Constraints become deterministic, allocation-light, and JIT-compilable.

### New constraint API (opt-in)

User's constraints module declares the aggregations it needs:

```python
# clue_constraints.py — new API
required_aggregations = {
    'dob':         'set_str',           # union of distinct dob strings
    'age':         'int_min_max',       # cluster-wide min/max
    'fn_ln_dob':   'set_tuple',         # (first_name, last_name, dob) combos
    # Aggregations are computed from these source columns:
    # 'fn_ln_dob' aggregation source = ['first_name', 'last_name', 'dob']
}

aggregation_sources = {
    'fn_ln_dob': ('first_name', 'last_name', 'dob'),
}

def is_valid_cluster_fast(summary, phat):
    """summary is a dict, not a DataFrame.

    Keys (always present):
      summary['size']        : int — number of records
      summary['uid']         : set[str] — for auto uid uniqueness check

    Keys (per required_aggregations):
      summary['dob']         : set[str]
      summary['age_min']     : int
      summary['age_max']     : int
      summary['fn_ln_dob']   : set[tuple]
    """
    if summary['size'] > 300:
        rejection_reasons['too_many_records_>300'] += 1
        return False
    if len(summary['dob']) > 5:
        rejection_reasons['too_many_unique_dobs_>5'] += 1
        return False
    if summary['age_max'] - summary['age_min'] > 3:
        rejection_reasons['age_range_>3_years'] += 1
        return False
    if len(summary['fn_ln_dob']) > 6:
        rejection_reasons['too_many_name_dob_combos_>6'] += 1
        return False
    return True
```

### Aggregation primitives

Namematch supports a fixed vocabulary of merge-friendly aggregations, all of which can be combined in O(1) or O(k) where k = size of smaller cluster:

| Aggregation | Merge op | Storage |
|---|---|---|
| `set_str` | union | `frozenset[str]` or `set[str]` |
| `set_tuple` | union | `set[tuple]` |
| `set_int` | union | `set[int]` |
| `int_min_max` | `min(a_min, b_min)`, `max(a_max, b_max)` | two ints |
| `int_sum` | `a + b` | one int |
| `float_min_max` | as above | two floats |
| `int_count_distinct` | not directly mergeable; user must use `set_int` and `len()` | — |

`uid` is always aggregated as `set_str` (used by the built-in `auto_is_valid_cluster`).

### Internal state representation

Replace `cluster_info: pd.DataFrame` + `ci_ix_map: dict` with a single `ClusterStateStore`:

```python
class ClusterStateStore:
    """Per-cluster aggregated state, maintained incrementally during clustering."""

    def __init__(self, all_names_df, aggregations: dict, sources: dict):
        # Initialize per-record aggregations from all_names, then seed
        # one cluster per record_id.
        self.size: dict[int, int] = {}
        self.uid: dict[int, set] = {}
        self.aggs: dict[str, dict] = {}  # agg_name -> {cluster_id -> value}

    def merge(self, cid1: int, cid2: int) -> dict:
        """Compute (but do not apply) the merged state. Returns the summary
        dict that gets passed to is_valid_cluster_fast."""
        ...

    def apply(self, cid_keep: int, cid_drop: int, merged_state: dict):
        """Commit a merge: drop cid_drop's state, install merged_state for cid_keep."""
        ...
```

### JIT path

For users who only use the built-in aggregation types and don't define a custom `is_valid_cluster_fast`, namematch ships a Numba-jitted default loop. Constraints expressed as simple numeric thresholds (size, set-cardinality caps, range caps) compile cleanly.

User-defined `is_valid_cluster_fast` runs in pure Python with the pre-aggregated summary dict — still 50–100× faster than today because zero pandas, but not JITted.

### Backward compatibility

The legacy `is_valid_cluster(df, phat)` API stays supported. Detection at module-load time:

```python
if hasattr(constraints_module, 'is_valid_cluster_fast'):
    # use new fast path
elif hasattr(constraints_module, 'is_valid_cluster'):
    logger.warning("Using legacy DataFrame-based is_valid_cluster API; consider "
                   "migrating to is_valid_cluster_fast for ~50x speedup. "
                   "See docs/cluster_constraints_migration.rst")
    # use existing slow path unchanged
```

Existing users see no behavior change. New users (and migrating users) opt in by writing `is_valid_cluster_fast` + `required_aggregations`.

### Edge constraints (`is_valid_link`)

Out of scope for this rewrite — `is_valid_link` already operates on a batch DataFrame at the *edge* level (vectorized over multiple edges at once), so it's not in the hot per-edge loop. Keep as-is.

### Auto-validity check

`auto_is_valid_cluster` does uid/eid uniqueness. With `uid` and `eid` as `set_str` aggregations, the check becomes:

```python
def auto_is_valid_cluster_fast(uid_set_1, uid_set_2, eid_set_1, eid_set_2,
                                allow_multiple_uids):
    if not allow_multiple_uids and len(uid_set_1 | uid_set_2) > 1:
        return False
    if len(eid_set_1 | eid_set_2) > 1:
        return False
    return True
```

Trivial to JIT. Single dictionary lookups per side.

## Implementation phases

Each phase is independently reviewable and shippable as a separate commit.

**Phase 1 — `ClusterStateStore` + aggregation primitives.**
New module `namematch/cluster_state.py`. Implements the aggregation vocabulary and the merge/apply API. Unit-tested in isolation. No integration with main loop yet.

**Phase 2 — Constraint module loader supports both APIs.**
Detect `is_valid_cluster_fast` vs legacy. Add `required_aggregations` declaration parsing. Pass through to clustering. If declared, namematch initializes `ClusterStateStore` from `all_names`. No behavior change yet (legacy users hit the original path).

**Phase 3 — Fast path in `cluster_potential_edges`.**
Branch on whether the user's constraints module declares `is_valid_cluster_fast`. If yes: run the new loop that calls `store.merge()`, passes `summary` to the user's function, and calls `store.apply()` on accept. Legacy users still hit the existing loop.

**Phase 4 — Numba JIT of the default constraint path.**  *Skipped after phase 3 measurements.*

Original plan: optional default `is_valid_cluster_fast` exists in `default_constraints.py` that's always JITted.

**Why skipped:** Numba doesn't support arbitrary Python set operations. The hot loop's per-edge work is dominated by `set | set` unions on `set_str` / `set_tuple` aggregations, plus dict allocation for the summary - none of which JIT cleanly.

Profile on a realistic 200K-edge workload after phase 3: **6 µs/edge** (vs ~500 µs in the legacy pandas-slicing loop = ~80× speedup). The biggest remaining levers are:
- Passing precomputed merged state from `merge_preview()` to `commit_merge()` to avoid recomputing aggregations on accept: ~10% gain
- Skipping summary unpacking when the constraint accesses raw state: ~4%
- Declarative size-cap on the constraints module for early-reject before allocating summary: ~15% (when many edges are size-rejected)

None changes the picture meaningfully once the 3h Cluster step is at ~4 minutes. Decision: don't chase the long tail; route the effort into phase 5 validation instead. If a future workload shows the fast path is still a bottleneck, revisit with profile data from that workload first.

**Phase 5 — CLUE migration.**
Port `clue_constraints.py` to the new API. Validate cluster outputs match the legacy run within tolerance (we'd expect bit-identical results since the constraints are deterministic, but verify).

**Phase 6 — Docs + upstream PR.**
- `docs/source/match_setup.rst`: add a "Fast constraints API" subsection under "Creating user defined constraint functions" with the migration recipe.
- `CHANGELOG.md`: opt-in fast path in unreleased.
- Open PR to `urban-labs/namematch`.

## Risks and edge cases

**Aggregation correctness on merge.** Set unions are commutative and associative, so merge order doesn't matter. Min/max same. Need to write merge-property tests.

**`set_tuple` memory.** A 300-record cluster with all-distinct (fn, ln, dob) combos holds 300 tuples in the set. Across 1.86M clusters, this is order-of magnitude similar to the current DataFrame storage — should not be a regression.

**User constraint that needs a column not in `required_aggregations`.** Loud error at constraint-load time, not at runtime. Validate the declaration covers every column the constraint function dereferences (AST scan, conservative).

**Constraint that needs the full record set (not just an aggregation).** Some users may want to do, e.g., "if any two records have edit distance > X on name, reject." Can't express that as a finite aggregation. Workaround: fall back to legacy DataFrame API for that constraint module. Document this limitation clearly. Most production constraints are aggregation-expressible.

**`new_edge` column in legacy API.** Some legacy constraints use the `new_edge` boolean column (records that are endpoints of the current edge). The fast API would need an equivalent — either pass `(rid1, rid2)` as a separate kwarg to `is_valid_cluster_fast`, or skip this entirely (most constraints don't use it). Our CLUE constraints don't use it, so we can defer.

**Tie-breaking in merge order.** Current greedy uses `min(cluster_id_1, cluster_id_2)` for the new cluster id. Preserve this — it's a contract callers depend on for cluster id stability across reruns.

**JIT cache invalidation.** Numba JIT caches compiled code in `__pycache__`-style storage. If a user changes their constraints, the cache must be invalidated. Standard Numba behavior handles this via source hash; document it.

## Effort estimate

| Phase | Estimate |
|---|---|
| 1. `ClusterStateStore` + primitives | 2 days |
| 2. Loader + declaration parsing | 1 day |
| 3. Fast path in main loop | 2 days |
| 4. Numba JIT | 1 day |
| 5. CLUE migration + validation | 1 day |
| 6. Docs + upstream PR | 1 day |
| **Total** | **~1.5 weeks** of focused work |

## Expected impact

Today, on the CLUE v3 dataset:
- 43M edges at `inexact_any=0.20` → ~6–9h clustering
- 12M edges at `inexact_any=0.43` → ~2h47m clustering (measured)

After rewrite (estimated):
- 43M edges → 10–20 min
- 12M edges → 3–5 min

This converts threshold-sensitivity sweeps (currently overnight runs) into per-iteration ~5 min, which makes proper appendix-style robustness analysis tractable.

## Decision points before starting

1. **Branch base.** `enhanced-matching-report` (lots of recent infra changes, including the OOM fix) vs `main` (cleaner, but missing the chunked-FitModel work). Recommend `enhanced-matching-report`.
2. **Backward-compat strictness.** Default to "legacy path stays unchanged, new API is opt-in." Open to a louder deprecation if upstream maintainers want it.
3. **Numba vs Cython.** Numba is lighter (no compilation pipeline). Recommend Numba.
4. **Should `is_valid_link` get the same treatment?** Currently it's batch-vectorized at the pair level, so it's not a hot spot. Defer.

## Out of scope

- Parallel clustering (connected-components based). Bigger algorithmic change, separate proposal.
- Edge pre-filtering (cheap dead-end detection before greedy loop). Separate proposal — could be done first if more urgent.
- GPU-accelerated clustering. Way out of scope.
