import json
from unittest.mock import patch

import pandas as pd

from namematch.generate_data_rows import GenerateDataRows


def test_generate_name_probabilities_object(params_and_schema, an_df, logger_for_testing):
    params, schema = params_and_schema
    with patch('namematch.generate_data_rows.logger', logger_for_testing) as mock_debug:
        generate_data_rows = GenerateDataRows(
            params,
            schema,
            all_names_file=an_df,
            candidate_pairs_file=None,
            output_dir=None,
        )

        prob_obj = generate_data_rows.generate_name_probabilities_object(
            an=an_df,
            fn_col=None,
            ln_col="column_name"
        )
        assert prob_obj is None

        prob_obj = generate_data_rows.generate_name_probabilities_object(
            an=an_df,
            fn_col='first_name',
            ln_col='last_name',
        )
        assert prob_obj is not None


def test_find_valid_training_records():
    pass


def test_generate_actual_data_rows():
    pass


def test_generate_data_row_files():
    pass
