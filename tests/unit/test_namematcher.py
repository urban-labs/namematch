import tempfile
import os

from namematch.namematcher import NameMatcher

def test_namematcher(config_dict):
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dict['num_workers'] = 1
        nm = NameMatcher(config=config_dict, output_dir=os.path.join(temp_dir, 'nm_output'))
        nm.process_input_data.run()
        nm.generate_must_links.run()
        nm.block.run()
        nm.generate_data_rows.run()
        nm.fit_model.run()
        nm.predict.run()
        nm.cluster.run()
        nm.generate_output.run()
        nm.generate_report.run()
        assert os.path.exists(os.path.join(temp_dir, 'nm_output', 'details', 'nm_info.yaml'))
        assert os.path.exists(os.path.join(temp_dir, 'nm_output', 'matching_report.html'))
