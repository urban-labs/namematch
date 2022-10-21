import pandas as pd
from unittest.mock import patch

from namematch.data_structures.variable import Variable
from namematch.process_input_data import (
    ProcessInputData,
    process_set_missing,
    process_drop
)


def test_process_geo_column(logger_for_testing, raw_data_df, params_and_schema):
    # load fake data
    v = Variable({"name": "lat/lon"})
    raw_data_df["lat"] = raw_data_df["lat"].astype(str)
    raw_data_df["lon"] = raw_data_df["lon"].astype(str)

    with patch('namematch.process_input_data.logger', logger_for_testing) as mock_debug:
        # test two cols
        params, schema = params_and_schema
        processed_geo = ProcessInputData(params, schema).process_geo_column(raw_data_df[["lat", "lon"]], v)
        assert 1 == processed_geo.shape[1]
        assert raw_data_df.shape[0] == processed_geo.shape[0]

        # test one col
        processed_geo = ProcessInputData(params, schema).process_geo_column(raw_data_df[["lat/lon"]], v)
        assert 1 == processed_geo.shape[1]
        assert raw_data_df.shape[0] == processed_geo.shape[0]


def test_parse_address(params_and_schema):
    test_cases = [
        ("123 Main St.", "123", "main", "street"),
        ("123 Broadway", "123", "broadway", ""),
        ("", "", "", ""),
        ("123 MAIN STREET", "123", "main", "street"),
        ("123 MAIN ST", "123", "main", "street"),
        # ("666 Lincoln Terrace", "666", "lincoln", "terrace"), see TODO in generate_data_files
        ("Twelve Forbes Ave", "12", "forbes", "avenue")]
    params, schema = params_and_schema
    process_input_data = ProcessInputData(params, schema)
    for (address, expected_num, expected_name, expected_type) in test_cases:
        parsed_address = process_input_data.parse_address(address)
        assert len(parsed_address) == 3
        (num, name, typ) = parsed_address
        assert expected_num == num
        assert expected_name == name
        assert expected_type == typ


def test_process_address_column(raw_data_df, params_and_schema):
    params, schema = params_and_schema
    # process
    processed_address = ProcessInputData(params, schema).process_address_column(raw_data_df[["address"]])

    # test
    assert raw_data_df.shape[0] == processed_address.shape[0]
    assert 3 == processed_address.shape[1]


def test_process_check_name_items(logger_for_testing, params_and_schema):
    with patch('namematch.process_input_data.logger', logger_for_testing) as mock_debug:
        # test that the series name gets set and
        # items are stripped and upper case
        name = "test_name"
        v = Variable({"name": name, "compare_type": "", "check": ""})
        s = pd.Series([" test1", "test2", " test3 "])
        params, schema = params_and_schema
        processed_s = ProcessInputData(params, schema).process_check(s, v)
        expected_s = pd.Series(["TEST1", "TEST2", "TEST3"])
        expected_s.name = name
        pd.testing.assert_series_equal(expected_s, processed_s)


def test_process_check_numeric_strings(logger_for_testing, params_and_schema):
    with patch('namematch.process_input_data.logger', logger_for_testing) as mock_debug:
        # test that numeric strings are checked
        name = "test_name"
        v = Variable({"name": name, "compare_type": "", "check": "Numeric"})
        s = pd.Series(["123", "5555", "1997", "notnumeric", "alsonot"])
        params, schema = params_and_schema
        processed_s = ProcessInputData(params, schema).process_check(s, v)
        expected_s = pd.Series(["123", "5555", "1997", "", ""])
        expected_s.name = name
        pd.testing.assert_series_equal(expected_s, processed_s)


def test_process_check_dates(logger_for_testing, params_and_schema):
    with patch('namematch.process_input_data.logger', logger_for_testing) as mock_debug:
        # test that dates are checked
        name = "test_name"
        v = Variable({"name": name, "compare_type": "", "check": "Date - %y-%m-%d"})
        s = pd.Series(["97-6-5", "85-10-2", "", "0000"])
        params, schema = params_and_schema
        processed_s = ProcessInputData(params, schema).process_check(s, v)
        expected_s = pd.Series(["1997-06-05", "1985-10-02", '', ''])
        expected_s.name = name
        pd.testing.assert_series_equal(expected_s, processed_s)


def test_process_check_categorical_variables(logger_for_testing, params_and_schema):
    with patch('namematch.process_input_data.logger', logger_for_testing) as mock_debug:
        # test that categorical variables are checked
        name = "test_name"
        v = Variable({"name": name, "compare_type": "Categorical", "check": "f,m,other"})
        s = pd.Series(["m", "f", "other", "", "mm", "m", "mf", "other"])
        params, schema = params_and_schema
        processed_s = ProcessInputData(params, schema).process_check(s, v)
        expected_s = pd.Series(["m", "f", "other", "", "", "m", "", "other"])
        expected_s.name = name
        pd.testing.assert_series_equal(expected_s, processed_s)

def test_process_set_missing(logger_for_testing):
    with patch('namematch.process_input_data.logger', logger_for_testing) as mock_debug:
        input_s = pd.Series(["hello", "test1", "NA", "0000", "NA", "test2", "0000"])
        expected_s = pd.Series(["hello", "test1", ""  ,  ""   , ""  , "test2", ""])
        processed_s = process_set_missing(input_s, ["NA", "0000"])
        pd.testing.assert_series_equal(expected_s, processed_s)

def test_process_drop(logger_for_testing):
    with patch('namematch.process_input_data.logger', logger_for_testing) as mock_debug:
        s = pd.Series(["hello", "test1", "NA", "0000", "NA", "test2", "0000"])
        expected_drops = [2, 3, 4, 6]
        processed_drops = process_drop(s, ["NA", "0000"])
        assert expected_drops == processed_drops


def test_process_auto_drops():
    pass


def test_process_data():
    pass


def test_split_last_names():
    pass


