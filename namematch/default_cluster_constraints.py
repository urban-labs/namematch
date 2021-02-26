import numpy as np
import pandas as pd

import logging
logger = logging.getLogger()

try:
    profile
except:
    from line_profiler import LineProfiler
    profile = LineProfiler()


def get_columns_used():
    '''The `get_columns_used()` function returns a dictionary mapping data fields needed for 
    constraint checking to the datatype they should be read in as. For this defaultversion,
    it simply returns an empty dictionary.
    
    Special note: if you want something to be treated as a date, use 
    dtype "date" (will be converted to a valid pandas dtype internally).

    Returns: 
        dict: empty for default version
    '''
	
    return {}


def is_valid_edge(record1, record2, phat):
    '''Check if two records would form a valid link, or edge. For this default version, 
    it simply returns True.

    Special note: for fields read in as dtype object (above), NAs have been filled with ""

    Args: 
        record1 (pd.Series): one row of the all-names file (relevant columns only)
        record2 (pd.Series): one row of the all-names file (relevant columns only)
        phat (float): phat value of the proposed edge

    Returns: 
        bool: True for default version
    '''

    return True


def is_valid_cluster(cluster, phat):
    '''Check if a proposed cluster is valid. For this default version, 
    it simply returns True. The information you'll have access to for the 
    cluster's records is determined by the column(s) specified in `get_columns_used`.

    Args: 
        cluster (pd.DataFrame): all-names file (relevant columns only) records for the proposed cluster
        phat (float): phat value of the proposed edge

    Returns: 
        bool: True for default version
    '''

    return True


def apply_edge_priority(edges_df, records_df):
    '''Adjust the order in which potential edges will be considered by altering 
    the phat (predicted probability of a record pair being a match). Edges will be 
    considered in decreasing order of phat. For the vast majority of runs, the default 
    behaviour of doing nothing is best. 

    Args: 
        edges_df (pd.DataFrame): potential edges information
            ================      =======================================================
            record_id_1           unique record identifier (for first in pair)
            record_id_2           unique record identifier (for second in pair)
            phat                     predicted probability of a record pair being a match
            ================      =======================================================
        records_df (pd.DataFrame): all-names file (relevant columns only)

    Returns: 
        pd.DataFrame: same as input, with phat columns potentially adjusted
    '''

    edges_df = edges_df.sort_values(by=['phat', 'original_order'], ascending=[False, True])

    return edges_df
