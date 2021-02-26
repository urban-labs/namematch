import json
import os
import pandas as pd
import unittest

from namematch.data_structures.data_file import *
from namematch.process_input_data import *
from namematch.data_structures.variable import *

logging_config = yaml.load(open('tests/logging_config.yaml', 'r'), Loader=yaml.FullLoader)
setup_logging(logging_config, None)
logger = logging.getLogger()
logging.disable(logging.CRITICAL)


class TestProcessInputData(unittest.TestCase):

    PATH = "tests/unit/data/"

    def test_process_geo_column(self):

        # load fake data
        v = Variable({"name": "lat/lon"})
        df = pd.read_csv(self.PATH + "raw_data.csv")
        df["lat"] = df["lat"].astype(str)
        df["lon"] = df["lon"].astype(str)

        # test two cols
        processed_geo = process_geo_column(df[["lat", "lon"]], v)
        self.assertEqual(1, processed_geo.shape[1])
        self.assertEqual(df.shape[0], processed_geo.shape[0])

        # test one col
        processed_geo = process_geo_column(df[["lat/lon"]], v)
        self.assertEqual(1, processed_geo.shape[1])
        self.assertEqual(df.shape[0], processed_geo.shape[0])


    def test_parse_address(self):

        test_cases = [
            ("123 Main St.", "123", "main", "street"),
            ("123 Broadway", "123", "broadway", ""),
            ("", "", "", ""),
            ("123 MAIN STREET", "123", "main", "street"),
            ("123 MAIN ST", "123", "main", "street"),
            # ("666 Lincoln Terrace", "666", "lincoln", "terrace"), see TODO in generate_data_files
            ("Twelve Forbes Ave", "12", "forbes", "avenue")]

        for (address, expected_num, expected_name, expected_type) in test_cases:
            parsed_address = parse_address(address)
            self.assertEqual(len(parsed_address), 3)
            (num, name, typ) = parsed_address
            self.assertEqual(expected_num, num)
            self.assertEqual(expected_name, name)
            self.assertEqual(expected_type, typ)


    def test_process_address_column(self):

        # load fake data
        df =  pd.read_csv(self.PATH + "raw_data.csv")

        # process
        processed_address = process_address_column(df[["address"]])

        # test
        self.assertEqual(df.shape[0], processed_address.shape[0])
        self.assertEqual(3, processed_address.shape[1])


    def test_process_check(self):

        # test that the series name gets set and
        # items are stripped and upper case
        name = "test_name"
        v = Variable({"name": name, "compare_type": "", "check": ""})
        s = pd.Series([" test1", "test2", " test3 "])
        processed_s = process_check(s, v)
        expected_s = pd.Series(["TEST1", "TEST2", "TEST3"])
        expected_s.name = name
        pd.testing.assert_series_equal(expected_s, processed_s)

        # test that numeric strings are checked
        v = Variable({"name": name, "compare_type": "", "check": "Numeric"})
        s = pd.Series(["123", "5555", "1997", "notnumeric", "alsonot"])
        processed_s = process_check(s, v)
        expected_s = pd.Series(["123", "5555", "1997", "", ""])
        expected_s.name = name
        pd.testing.assert_series_equal(expected_s, processed_s)

        # test that dates are checked
        v = Variable({"name": name, "compare_type": "", "check": "Date - %y-%m-%d"})
        s = pd.Series(["97-6-5", "85-10-2", "", "0000"])
        processed_s = process_check(s, v)
        expected_s = pd.Series(["1997-06-05", "1985-10-02", "", ""])
        expected_s.name = name
        pd.testing.assert_series_equal(expected_s, processed_s)

        # test that categorical variables are checked
        v = Variable({"name": name, "compare_type": "Categorical", "check": "f,m,other"})
        s = pd.Series(["m", "f", "other", "", "mm", "m", "mf", "other"])
        processed_s = process_check(s, v)
        expected_s = pd.Series(["m", "f", "other", "", "", "m", "", "other"])
        expected_s.name = name
        pd.testing.assert_series_equal(expected_s, processed_s)


    def test_process_set_missing(self):

        input_s = pd.Series(["hello", "test1", "NA", "0000", "NA", "test2", "0000"])
        expected_s = pd.Series(["hello", "test1", ""  ,  ""   , ""  , "test2", ""])
        processed_s = process_set_missing(input_s, ["NA", "0000"])
        pd.testing.assert_series_equal(expected_s, processed_s)


    def test_process_drop(self):

        s = pd.Series(["hello", "test1", "NA", "0000", "NA", "test2", "0000"])
        expected_drops = [2, 3, 4, 6]
        processed_drops = process_drop(s, ["NA", "0000"])
        self.assertEqual(expected_drops, processed_drops)


    def test_process_auto_drops(self):
        pass


    def test_process_data(self):
        pass


    def test_split_last_names(self):
        pass


