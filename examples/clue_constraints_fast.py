"""Fast-API port of clue_constraints.py.

Same cluster-level rules as the legacy version, expressed via
declarative aggregations + a per-edge function that queries a small
summary dict. No pandas DataFrames in the hot path.

How to use:
    nm = NameMatcher(constraints='clue_constraints_fast.py', ...)

Namematch detects `is_valid_cluster_fast` + `required_aggregations`
at constraints-load time and routes the clustering hot loop through
the new fast path. See namematch/cluster_state.py and
docs/plans/2026-05-27-cluster-rewrite.md.

Legacy clue_constraints.py is unchanged - both files can sit
side-by-side and you choose by which one you pass to NameMatcher.
"""

from collections import defaultdict


# --- Legacy callables (unchanged) ------------------------------------------
# Namematch still calls these for edge constraints / link priority /
# get_columns_used; the only swap is at the cluster-validity step.

def is_valid_link(predicted_links_df):
    """No-op edge constraint - same as the legacy file."""
    df = predicted_links_df
    df['valid'] = True
    return df.valid


def is_valid_cluster(cluster, phat):
    """Legacy DataFrame-based check - kept for backward compatibility.

    When the fast API is in use (because this module also defines
    is_valid_cluster_fast + required_aggregations), namematch picks
    the fast version and this function is not called. We keep it here
    so the same module can also serve as a drop-in if you ever want
    to A/B by deleting the fast attributes.
    """
    if cluster.shape[0] > 300:
        rejection_reasons['too_many_records_>300'] += 1
        return False
    if cluster.dob.nunique() > 5:
        rejection_reasons['too_many_unique_dobs_>5'] += 1
        return False
    if cluster.age.astype(int).max() - cluster.age.astype(int).min() > 3:
        rejection_reasons['age_range_>3_years'] += 1
        return False
    if cluster[['first_name', 'last_name', 'dob']].drop_duplicates().shape[0] > 6:
        rejection_reasons['too_many_name_dob_combos_>6'] += 1
        return False
    return True


def apply_link_priority(valid_links_df):
    """Sort by descending phat, ties broken by original_order."""
    return valid_links_df.sort_values(
        by=['phat', 'original_order'], ascending=[False, True])


def get_columns_used():
    """Legacy column declaration. Namematch additionally auto-includes any
    columns referenced by required_aggregations / aggregation_sources, so
    the same set ends up loaded either way."""
    return "all"


# --- Fast-API declarations -------------------------------------------------

# Aggregations namematch must maintain per cluster. Keys here become keys
# in the summary dict passed to is_valid_cluster_fast. Each value names
# an aggregation type from namematch.cluster_state.AGGREGATIONS.
required_aggregations = {
    'dob':         'set_str',       # distinct dob strings in the cluster
    'age':         'int_min_max',   # cluster age range (min, max)
    'fn_ln_dob':   'set_tuple',     # distinct (first_name, last_name, dob) combos
}

# Multi-column aggregations need their source columns declared. Single-
# column aggs (dob, age) default to (agg_name,) so don't need an entry.
aggregation_sources = {
    'fn_ln_dob': ('first_name', 'last_name', 'dob'),
}

# Rejection counter shared with the legacy is_valid_cluster - whichever
# function gets called for a given merge writes to the same dict, so the
# matching report renders identical rejection-reason tables either way.
rejection_reasons = defaultdict(int)


def is_valid_cluster_fast(summary, phat):
    """Same four rules as is_valid_cluster, expressed against the
    pre-aggregated summary dict instead of a pandas slice.

    summary keys (all always present):
        size          : int                - record count in the cluster
        uid           : set[str]           - distinct uids (used by namematch's
                                              auto uid-uniqueness check, not
                                              ours; declared as set_str by
                                              namematch automatically)
        dob           : set[str]           - per `required_aggregations`
        age_min       : int or None        - int_min_max unpacks to two keys
        age_max       : int or None
        fn_ln_dob     : set[tuple]
    """
    if summary['size'] > 300:
        rejection_reasons['too_many_records_>300'] += 1
        return False

    if len(summary['dob']) > 5:
        rejection_reasons['too_many_unique_dobs_>5'] += 1
        return False

    # age_min / age_max are None when every record in the cluster has a
    # missing age. In that case the range check is meaningless; allow.
    # Mirrors the legacy intent (where .astype(int) would have errored
    # on all-NA, which the surrounding code prevented in practice).
    if summary['age_min'] is not None and summary['age_max'] is not None:
        if summary['age_max'] - summary['age_min'] > 3:
            rejection_reasons['age_range_>3_years'] += 1
            return False

    if len(summary['fn_ln_dob']) > 6:
        rejection_reasons['too_many_name_dob_combos_>6'] += 1
        return False

    return True


def print_rejection_summary():
    """Same helper as in the legacy module."""
    print("\n" + "=" * 80)
    print("CLUSTER REJECTION SUMMARY")
    print("=" * 80)

    if not rejection_reasons:
        print("No clusters were rejected.")
    else:
        total = sum(rejection_reasons.values())
        print(f"Total rejected clusters: {total:,}\n")
        for reason, count in sorted(
                rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            pct = 100.0 * count / total if total > 0 else 0
            print(f"  {reason:.<45} {count:>8,} ({pct:>5.1f}%)")

    print("=" * 80 + "\n")
