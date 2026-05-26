import math
import numpy as np
import pandas as pd
import editdistance
from collections import defaultdict

# Global counter to track cluster rejection reasons
rejection_reasons = defaultdict(int)


def is_valid_link(predicted_links_df):
    '''Check if two records would form a valid link. For this default version, 
    it simply returns True.

    Args: 
        predicted_links_df (pd.DataFrame): info on predicted links (record pairs)
            ================      =======================================================
            record_id_1           unique record identifier (for first in pair)
            record_id_2           unique record identifier (for second in pair)
            phat                  predicted probability of a record pair being a match
            original_order        original ordering 1-N (useful so gt is always on top of phat=1 cases)
            <other_cols>          columns from all-names that are required for constraint checking (will have _1 and _2 versions)
            ================      =======================================================

    Returns: 
        boolean or boolean pd.Series: True for default version
    '''
    df = predicted_links_df
    df['valid'] = True
        
    return df.valid


def is_valid_cluster(cluster, phat):
    '''Check if a proposed cluster is valid. For this default version,
    it simply returns True. The information you'll have access to for the
    cluster's records is determined by the column(s) specified in `get_columns_used`.

    Args:
        cluster (pd.DataFrame): all-names file (relevant columns only) records for the proposed cluster
        phat (float): phat value of the proposed link

    Returns:
        bool: True for default version
    '''

    # This prevents clusters that are larger than 300 records
    # Change this according to your expectations
    if cluster.shape[0] > 300:
        rejection_reasons['too_many_records_>300'] += 1
        return False

    # This is saying that there can only be 5 unique birthdates in a cluster
    # Probably want to change this or get rid of it for yours
    if cluster.dob.nunique()>5:
        rejection_reasons['too_many_unique_dobs_>5'] += 1
        return False

    # This is saying that the maximum difference in age in a cluster is 3
    if cluster.age.astype(int).max() - cluster.age.astype(int).min() > 3:
        rejection_reasons['age_range_>3_years'] += 1
        return False

    # Limit clusters to a maximum of 6 unique combinations of first_name, last_name, and dob
    # This prevents over-merging of distinct individuals
    if cluster[['first_name', 'last_name', 'dob']].drop_duplicates().shape[0] > 6:
        rejection_reasons['too_many_name_dob_combos_>6'] += 1
        return False

    return True
   

def apply_link_priority(valid_links_df):
    '''Adjust the order in which valid link will be considered for clustering. For the vast majority of 
    runs, the default behavior -- sorting by descending phat, or P(match) -- is best. 

    Args:
        valid_links_df (pd.DataFrame): info on valid predicted links (record pairs)
            ================      =======================================================
            record_id_1           unique record identifier (for first in pair)
            record_id_2           unique record identifier (for second in pair)
            phat                  predicted probability of a record pair being a match
            original_order        original ordering 1-N (useful so gt is always on top of phat=1 cases)
            <other_cols>          columns from all-names that are required for constraint checking (will have _1 and _2 versions)
            ================      =======================================================

    Returns:
        pd.DataFrame: same as input, with phat columns potentially adjusted
    '''

    valid_links_df = valid_links_df.sort_values(by=['phat', 'original_order'], ascending=[False, True])

    return valid_links_df


def get_columns_used():
    '''The `get_columns_used()` function returns either the string "all" (default) or a
    dictionary mapping the names of data fields needed for constraint checking to the data
    type they should be read in as. This can be useful to limit memory consumption during the
    clustering step (by only reading in the data fields only needed in cluster logic).

    Special note: if you want something to be treated as a date, use
    dtype "date" (will be converted to a valid pandas dtype internally).

    Returns:
        dict: empty for default version
    '''

    return "all"


def print_rejection_summary():
    '''Print a summary of cluster rejection reasons.

    Call this after clustering completes to see breakdown of why clusters were rejected.
    '''
    print("\n" + "="*80)
    print("CLUSTER REJECTION SUMMARY")
    print("="*80)

    if not rejection_reasons:
        print("No clusters were rejected.")
    else:
        total = sum(rejection_reasons.values())
        print(f"Total rejected clusters: {total:,}\n")

        # Sort by count descending
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            pct = 100.0 * count / total if total > 0 else 0
            print(f"  {reason:.<45} {count:>8,} ({pct:>5.1f}%)")

    print("="*80 + "\n")
