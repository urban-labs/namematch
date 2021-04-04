import argparse
import logging
import os
import yaml

from argcmdr import RootCommand, Command, main, cmdmethod
from namematch.namematcher import NameMatcher

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
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
            "--output-dir",
            default="output",
            help="output folder path"
        )
        parser.add_argument(
            "--output-temp-dir",
            help="output temp folder path"
        )

    def __call__(self, args):
        config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

        if args.output_temp_dir:
            output_temp_dir = args.output_temp_dir
        else:
            output_temp_dir = os.path.join(args.output_dir, 'details')

        nm = NameMatcher(
            config=config,
            output_dir=args.output_dir,
            output_temp_dir=output_temp_dir
        )
        nm.run()


def execute():
    main(NameMatch)


if __name__ == "__main__":
    main(NameMatch)
