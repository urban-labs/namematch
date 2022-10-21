import argparse
import logging
import os
import yaml

from argcmdr import RootCommand, Command, main

logging.basicConfig(
    format="%(asctime)s - %(levelname)-8s %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO
)


class NameMatch(RootCommand):
    """manage namematch"""

    def __init__(self, parser):
        parser.add_argument(
            "-c",
            "--config-file",
            help="configuration yaml file "
        )
        parser.add_argument(
            "-i",
            "--info-file",
            help="namematch info yaml file"
        )
        parser.add_argument(
            "--output-dir",
            default="output",
            help="output folder path (default: output)"
        )
        parser.add_argument(
            "--output-temp-dir",
            help="output temp folder path (default: <output_dir>/details)"
        )
        parser.add_argument(
            "--constraints-file",
            default=None,
            help="constraints file (optional)"
        )
        parser.add_argument(
            "-f",
            "--force",
            dest='force',
            action='store_true',
            help="force match to run even if outputs exist (default: False)"
        )
        parser.add_argument(
            "--trained-model-info-file",
            default='None',
            help="path to trained model from initial non-incremental run (required, only for incremental runs)"
        )
        parser.add_argument(
            "--existing-blocking-index-file",
            dest='og_blocking_index_file',
            default='None',
            help="path to existing blocking index from previous run (optional, only for incremental runs)"
        )
        parser.add_argument(
            "--enable-lprof",
            action='store_true',
            dest="enable_lprof",
            help="generate the line_profiler files for certain methods"
        )
        parser.add_argument(
            "--log-level",
            dest='log_level',
            default='INFO',
            help="logging level"
        )

    def get_namematcher(self):
        if bool(self.args.config_file) == bool(self.args.info_file):
            raise ValueError("Please only provide either config_file or info_file")


        from namematch.namematcher import NameMatcher

        if self.args.config_file:
            config = yaml.load(open(self.args.config_file, 'r'), Loader=yaml.FullLoader)

            if self.args.output_temp_dir:
                output_temp_dir = self.args.output_temp_dir
            else:
                output_temp_dir = os.path.join(self.args.output_dir, 'details')

            nm = NameMatcher(
                config=config,
                output_dir=self.args.output_dir,
                output_temp_dir=output_temp_dir,
                constraints=self.args.constraints_file,
                trained_model_info_file=self.args.trained_model_info_file,
                og_blocking_index_file=self.args.og_blocking_index_file,
                enable_lprof=self.args.enable_lprof,
                logging_level=self.args.log_level,
            )

        if self.args.info_file:
            logging.info("Loading from the stats file: {self.args.info_file}")
            nm = NameMatcher.load_namematcher(self.args.info_file)

        return nm


@NameMatch.register
class Run(Command):
    """Run all namematch steps"""
    def __call__(self, args):
        nm = self.root.get_namematcher()
        nm.run(force=self.root.args.force)


@NameMatch.register
class ProcessInputData(Command):
    """Process input data"""

    class Run(Command):
        def __init__(self, parser):
            parser.add_argument(
                '-f', '--force',
                dest='force',
                action='store_true',
                default=False,
                help="Process input data",
            )

        def __call__(self, args):
            nm = self.root.get_namematcher()
            nm.process_input_data.run(force=args.force)


@NameMatch.register
class GenerateMustLinks(Command):
    """Generate must links"""

    class Run(Command):
        def __init__(self, parser):
            parser.add_argument(
                '-f', '--force',
                dest='force',
                action='store_true',
                default=False,
                help="Genearte must links"
            )

        def __call__(self, args):
            nm = self.root.get_namematcher()
            nm.generate_must_links.run(force=args.force)


@NameMatch.register
class Block(Command):
    """Block"""

    class Run(Command):
        def __init__(self, parser):
            parser.add_argument(
                '-f', '--force',
                dest='force',
                action='store_true',
                default=False,
                help="Block"
            )

        def __call__(self, args):
            nm = self.root.get_namematcher()
            nm.block.run(force=args.force)


@NameMatch.register
class GenerateDataRows(Command):
    """Generate data rows"""

    class Run(Command):
        def __init__(self, parser):
            parser.add_argument(
                '-f', '--force',
                dest='force',
                action='store_true',
                default=False,
                help="Generate data rows"
            )

        def __call__(self, args):
            nm = self.root.get_namematcher()
            nm.generate_data_rows.run(force=args.force)


@NameMatch.register
class FitModel(Command):
    """Fit model"""

    class Run(Command):
        def __init__(self, parser):
            parser.add_argument(
                '-f', '--force',
                dest='force',
                action='store_true',
                default=False,
                help="Fit model"
            )

        def __call__(self, args):
            nm = self.root.get_namematcher()
            nm.fit_model.run(force=args.force)


@NameMatch.register
class Predict(Command):
    """Predict"""

    class Run(Command):
        def __init__(self, parser):
            parser.add_argument(
                '-f', '--force',
                dest='force',
                action='store_true',
                default=False,
                help="Predict"
            )

        def __call__(self, args):
            nm = self.root.get_namematcher()
            nm.predict.run(force=args.force)


@NameMatch.register
class Cluster(Command):
    """Cluster"""

    class Run(Command):
        def __init__(self, parser):
            parser.add_argument(
                '-f', '--force',
                dest='force',
                action='store_true',
                default=False,
                help="cluster"
            )

        def __call__(self, args):
            nm = self.root.get_namematcher()
            nm.cluster.run(force=args.force)


@NameMatch.register
class GenerateOutput(Command):
    """Generate output"""

    class Run(Command):
        def __init__(self, parser):
            parser.add_argument(
                '-f', '--force',
                dest='force',
                action='store_true',
                default=False,
                help="generate output"
            )

        def __call__(self, args):
            nm = self.root.get_namematcher()
            nm.generate_output.run(force=args.force)


@NameMatch.register
class GenerateReport(Command):
    """Generate report"""

    class Run(Command):
        def __init__(self, parser):
            parser.add_argument(
                '-f', '--force',
                dest='force',
                action='store_true',
                default=False,
                help="generate report"
            )

        def __call__(self, args):
            nm = self.root.get_namematcher()
            nm.generate_report.run(force=args.force)


def execute():
    main(NameMatch)


if __name__ == "__main__":
    main(NameMatch)
