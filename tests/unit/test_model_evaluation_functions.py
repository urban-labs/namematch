import pandas as pd
import numpy as np
import tempfile

from namematch.model_evaluation_functions import get_accuracy


def test_get_accuracy():

    thresh = 0.5

    df = pd.DataFrame({
        'phat':[0, .4, .5, .6, .9],
        'outcome':[0, 0, 1, 1, 1],
        'weight':[1, 1, 1, 1, 1]
    })

    assert get_accuracy(df, thresh) == 1

    one_wrong_no_weights_df = pd.DataFrame({
        'phat':[0, .9, .5, .6, .9],
        'outcome':[0, 0, 1, 1, 1],
        'weight':[1, 1, 1, 1, 1]
    })
    
    one_wrong_weights_df = pd.DataFrame({
        'phat':[0, .9, .5, .6, .9],
        'outcome':[0, 0, 1, 1, 1],
        'weight':[1, 2, 1, 1, 1]
    })

    assert get_accuracy(one_wrong_no_weights_df, thresh) > get_accuracy(one_wrong_weights_df, thresh)

