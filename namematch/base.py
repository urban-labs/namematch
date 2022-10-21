import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
import ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

from ruamel.yaml.comments import CommentedMap

from namematch.utils.utils import reformat_dict, setup_logging, camel_to_snake
from namematch.data_structures.parameters import Parameters
from namematch.data_structures.schema import Schema

logger = logging.getLogger()

class NamematchBase(ABC):
    '''The base class for namematch steps

    All the inherited classes should have other input/output file paths for each step.

    Args:
        params (object): namematch's Parameter object
        schema (object): namematch's Schema object
        output_file (str): output file path

    '''
    def __init__(
        self,
        params: Parameters,
        schema: Schema,
        output_dir: str = None,
        nm_info_file=None,
        nm_metadata=None,
        profile_dir: str = None,
        next=None,
        enable_lprof=False,
    ):

        self.params = params
        self.schema = schema
        self.output_dir = output_dir
        self.stats_dict = CommentedMap()
        self.nm_info_file = nm_info_file
        self.nm_metadata = nm_metadata
        self.profile_dir = profile_dir
        self.next = next
        self.enable_lprof=enable_lprof

    @abstractmethod
    def main(self):
        '''Main method for each task class which is called in namematcher.NameMatcher'''
        raise NotImplementedError

    @property
    def output_files(self):
        return []

    @property
    def common_path(self):
        return Path(self.nm_info_file).parents[1]

    @property
    def yaml_ready_stats_dict(self):
        '''Change yaml format or style to create the yaml ready stats_dict'''
        stats_dict = reformat_dict(self.stats_dict)
        return {self.__class__.__name__: stats_dict}

    def check_output(self):
        all_output_files_existed = True
        if len(self.output_files) > 0:
            all_output_files_existed = []
            for output_file in self.output_files:
                if os.path.exists(output_file):
                    logger.info(f"Output '{os.path.relpath(output_file, self.common_path)}' exists!")
                    all_output_files_existed.append(True)
                else:
                    logger.debug(f"Output '{os.path.relpath(output_file, self.common_path)}' doesn't exist!")
                    all_output_files_existed.append(False)
            all_output_files_existed = all(all_output_files_existed)

        return all_output_files_existed

    def remove_output(self):
        if len(self.output_files) > 0:
            for f in self.output_files:
                if os.path.exists(f):
                    os.remove(f)
                    logger.info(f"Removing {os.path.relpath(f, self.common_path)} from {self.__class__.__name__}")

    def remove_downstream_output_and_stats(self):
        next_task = self.next
        nm_info = self.get_nm_info_with_synced_metadata()
        nm_stats = nm_info['stats']
        while next_task:
            # remove next task's output
            next_task.remove_output()
            task_name = next_task.__class__.__name__
            # remove next task's stats in stats file
            if nm_stats.get(task_name, None):
                del nm_stats[task_name]
            # empty next task's stats_dict
            next_task.stats_dict = CommentedMap()
            next_task = next_task.next

        with open(self.nm_info_file, 'w') as f:
            for key, value in nm_info.items():
                nm_info[key] = reformat_dict(value)
            yaml.dump(nm_info, f)

    def remove_current_task_from_stats_file(self):
        nm_info = self.get_nm_info_with_synced_metadata()
        nm_stats = nm_info['stats']
        if nm_stats.get(self.__class__.__name__, None):
            del nm_stats[self.__class__.__name__]

        nm_info['stats'] = nm_stats

        with open(self.nm_info_file, 'w') as f:
            for key, value in nm_info.items():
                nm_info[key] = reformat_dict(value)
            yaml.dump(nm_info, f)

    def run(self, force=False, write_stats_file=True):
        '''Run the Namematch Task

        Args:
            force (bool): whether to force the task to rerun even if the output exists

        '''
        try:

            if force:
                # remove current task's output
                self.remove_output()
                # remove current stats in stats file
                self.remove_current_task_from_stats_file()
                # empty current task's stats_dict
                self.stats_dict = CommentedMap()
                # remove all the downstream tasks' outputs and stats
                self.remove_downstream_output_and_stats()

            if not self.check_output():
                # remove current tasks' output
                self.remove_output()
                # remove current stats in stats file
                self.remove_current_task_from_stats_file()
                # remove current task's stats_dict
                self.stats_dict = CommentedMap()
                # if nm_stats file is not empty, remove downstream outputs and stats
                if self.get_nm_info_with_synced_metadata():
                    self.remove_downstream_output_and_stats()
                else:
                    if self.__class__.__name__ != "ProcessInputData":
                        raise Exception("Fatal Error. Please clean the output and rerun NameMatch.")
                logger.info(f"Running task: {self.__class__.__name__}")
                self.main()

                if write_stats_file:
                    logger.info(f"Writing stats for task: {self.__class__.__name__}")
                    try:
                        self.write_nm_info_to_yaml()
                    except Exception as e:
                        raise e

            else:
                logger.info(f"{self.__class__.__name__} does not run.")
                logger.info(f"The output of {self.__class__.__name__} exists. To force run, set `force` to True.")


        except KeyboardInterrupt as e:
            # remove current tasks' output
            self.remove_output()
            # remove current stats in stats file
            self.remove_current_task_from_stats_file()
            # remove current task's stats_dict
            self.stats_dict = CommentedMap()
            logger.critical(f"Keyboard Interrupt")
            logger.critical(f"Rolling back...")
            logger.critical(f"Next run will start from {self.__class__.__name__}")

            raise

    def get_nm_info_with_synced_metadata(self):
        '''Get the nm_info to sync metadata from the nm_info_file with the metadata from the namematch object.

        This is to update the metadata information from nm_info_file if one changes param/schem from the namematcher object.

        '''
        if os.path.exists(self.nm_info_file):
            with open(self.nm_info_file, 'r') as f:
                nm_info = yaml.load(f)
                nm_info['metadata'] = self.nm_metadata
        else:
            nm_info = CommentedMap()
            nm_info['metadata'] = self.nm_metadata
        return nm_info

    @property
    def nm_stats(self):
        return self.get_nm_info_with_synced_metadata()['stats']

    def write_nm_info_to_yaml(self):
        '''Write stats_dict to yaml file'''
        nm_metadata = self.get_nm_info_with_synced_metadata()

        nm_metadata['stats'].update(self.yaml_ready_stats_dict)
        with open(self.nm_info_file, 'w') as f:
            yaml.dump(nm_metadata, f)

    def write_line_profile_stats(self, profiler, output_unit=1e-6, stripzeros=False):
        current_lprof_file = camel_to_snake(self.__class__.__name__) + ".lprof"

        with open(os.path.join(self.profile_dir, current_lprof_file), 'w+') as f:
            profiler.print_stats(
                stream=f,
                output_unit=output_unit,
                stripzeros=stripzeros,
            )
            # clean up profiler
            profiler.functions = []
            profiler.code_map = {}


