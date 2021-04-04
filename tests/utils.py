import tempfile

from contextlib import contextmanager

import pandas as pd


@contextmanager
def make_temp_parquet_file(file_path):
    df = pd.read_csv(file_path)
    tf = tempfile.NamedTemporaryFile(delete=False)
    df.to_parquet(tf.name)
    yield tf.name
    tf.close()

