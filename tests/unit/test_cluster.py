import pandas as pd
import unittest

from unittest.mock import Mock

from namematch.cluster import *
from namematch.utils.utils import *

logging_config = yaml.load(open('tests/logging_config.yaml', 'r'),Loader=yaml.FullLoader)
setup_logging(logging_config, None)
logger = logging.getLogger()
logging.disable(logging.CRITICAL)


class TestClustering(unittest.TestCase):

    PATH = "tests/unit/data/"

    def test_get_initial_clusters(self):

        # test from scratch

        an = pd.read_csv(self.PATH + "all_names_clustering.csv")
        ml = pd.read_csv(self.PATH + "gt_edges_clustering.csv")
        eid_col = None

        clusters, cluster_assignments, original_cluster_ids = \
                get_initial_clusters(ml, an, eid_col)
        self.assertIsNone(original_cluster_ids)
        self.assertEqual(3, len(clusters))
        self.assertEqual({"A", "B", "C", "D", "E", "F", "G", "H"}, set(cluster_assignments.keys()))

        # test incremental

        # TODO


    def test_auto_is_valid_edge(self):
        pass


    def test_auto_is_valid_cluster(self):
        pass


    def test_get_potential_edges(self):
        pass


    def test_load_cluster_info(self):
        pass


    def test_cluster_potential_edges(self):
        pass


    def test_pandas_sorting_default(self):

        df = pd.DataFrame(data={
            'a':[1, 2, 3, 4, 5],
            'b':[1, .99, .98, .97, .8]
        })
        df_dict_list = df.to_dict('records')

        a_vec = [d['a'] for d in df_dict_list]
        b_vec = [d['b'] for d in df_dict_list]

        self.assertEqual(df.a.tolist(), a_vec)
        self.assertEqual(df.b.tolist(), b_vec)


