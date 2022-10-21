import os
import logging
import tempfile
import papermill as pm
from nbconvert import HTMLExporter
from namematch.base import NamematchBase
from namematch.data_structures.parameters import Parameters
from namematch.data_structures.schema import Schema

class IgnoreBlackWarning(logging.Filter):
    def filter(self, record):
        return 'Black is not installed' not in record.msg


logging.getLogger("papermill.translators").addFilter(IgnoreBlackWarning())
logger = logging.getLogger()


class GenerateReport(NamematchBase):
    '''
    params (Parameters object): contains parameter values
    schema (Schema object): contains match schema info (files to match, variables to use, etc.)
    report_file (str): full path of the report html file
    '''
    def __init__(self, params, schema, report_file, *args, **kwargs):
        super(GenerateReport, self).__init__(params, schema, *args, **kwargs)
        self.report_file = report_file

    @property
    def output_files(self):
        return [self.report_file]


    def main(self, **kw):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output_notebook = os.path.join(temp_dir, 'matching_report_output.ipynb')
            # Execute notebook as a temp notebook
            pm.execute_notebook(
                os.path.join(os.path.dirname(__file__), 'matching_report.ipynb'),
                temp_output_notebook,
                parameters=dict(nm_info_path=self.nm_info_file),
                kernel_name='python3',
                progress_bar=False
            )
            # Export to html
            logger.info("Exporting to html")
            html_exporter = HTMLExporter()
            html_exporter.exclude_input = True
            html_exporter.exclude_input_prompt = True
            html_exporter.exclude_output_prompt = True
            html_data, resources = html_exporter.from_filename(temp_output_notebook)
            with open(self.report_file, 'w') as f:
                f.write(html_data)

            logger.info(f"The matching report is created at {self.report_file}")
