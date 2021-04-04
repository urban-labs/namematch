import numpy as np
import tempfile

from namematch.fit_model import define_necessary_models


def test_get_feature_info():
    pass


def test_fit_model():
    pass


def test_fit_models():
    pass


def test_define_necessary_models(data_rows_parquet_file):

    # fake data
    dr_file_list = [data_rows_parquet_file]
    missing_field = "first_name"

    with tempfile.TemporaryDirectory() as temp_output_dir:
        # missing
        model_info = define_necessary_models(dr_file_list, temp_output_dir, missing_field)
        assert 2 == len(model_info)
        assert "no_{}".format(missing_field) in model_info

        # not missing
        model_info = define_necessary_models(dr_file_list, temp_output_dir, None)
        assert 1 == len(model_info)


def test_get_match_train_eligible_flag():
    pass


def test_load_model_data():
    pass


def test_get_flipped0_potential_edges():
    pass


