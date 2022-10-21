import numpy as np
import pandas as pd


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

    return True


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
