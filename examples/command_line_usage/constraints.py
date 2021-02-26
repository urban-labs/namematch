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

    # NOTE: This is left as the default
	
    return {}


def is_valid_edge(record1, record2, phat):

    # Add a constraint specific to our matching task
    if (record1['dataset'] == 'potential_candidates') and (record2['dataset'] == 'potential_candidates'):
        return False

    return True


def is_valid_cluster(cluster, phat):
    
    # Add a constraint specific to our matching task
    if (cluster['dataset'] == 'potential_candidates').sum() > 1:
        return False

    return True


def apply_edge_priority(edges_df, records_df):
    
    # NOTE: This is left as the default

    edges_df = edges_df.sort_values(by=['phat', 'original_order'], ascending=[False, True])

    return edges_df


