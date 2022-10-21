import numpy as np
import pandas as pd

def get_columns_used():

    # NOTE: This is left as the default
	
    return {}


def is_valid_link(predicted_links_df):

    predicted_links_df['valid'] = True

    # Add a constraint specific to our matching task
    predicted_links_df.loc[
        (predicted_links_df.dataset_1 == 'potential_candidates') & 
        (predicted_links_df.dataset_2 == 'potential_candidates'), 
        'valid'] = False
    
    return predicted_links_df['valid']


def is_valid_cluster(cluster, phat):
    
    # Add a constraint specific to our matching task
    if (cluster['dataset'] == 'potential_candidates').sum() > 1:
        return False

    return True


def apply_link_priority(valid_links_df):
    
    # NOTE: This is left as the default

    valid_links_df = valid_links_df.sort_values(by=['phat', 'original_order'], ascending=[False, True])

    return valid_links_df
