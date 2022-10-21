# Command line tool usage

## Run namematch the first time

You need to provide a config file, output directory path, and (optional) constraint file to run namematch. The config file should have information about the data files being linked -- such as where to find them and which data fields should be used -- and the desired values for different Name Match parameters. Please check out `examples/command_line_usage/config.yaml` to see an example.

```
cd examples/command_line_usage/
namematch --config-file=config.yaml --output-dir=nm_ouptut --cluster-constraints-file=constraints.py run
```

## Start where you left with `--info-file` flag

Every run of namematch will create a `nm_info.yaml` where all the metadata and stats are stored. There are times that nameamtch is killed and stopped accidentally or on purpose. With `nm_info.yaml` you can start where you left last time when namematch got killed.

```
namematch -i nm_output/details/nm_info.yaml run
```

## Use `namematch --help` to see all the options

```shell
$ namematch --help
usage: cli.py [-h] [--tb] [-c CONFIG_FILE] [-i INFO_FILE]
              [--output-dir OUTPUT_DIR] [--output-temp-dir OUTPUT_TEMP_DIR]
              [--constraints-file CONSTRAINTS_FILE] [-f]
              [--trained-model-info-file TRAINED_MODEL_INFO_FILE]
              [--existing-blocking-index-file OG_BLOCKING_INDEX_FILE]
              [--enable-lprof]
              {run} ...

manage namematch

optional arguments:
  -h, --help            show this help message and exit
  --tb, --traceback     print error tracebacks
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        configuration yaml file
  -i INFO_FILE, --info-file INFO_FILE
                        namematch info yaml file
  --output-dir OUTPUT_DIR
                        output folder path (default: output)
  --output-temp-dir OUTPUT_TEMP_DIR
                        output temp folder path (default:
                        <output_dir>/details)
  --onstraints-file CONSTRAINTS_FILE
                        constraints file (optional)
  -f, --force           force match to run even if outputs exist (default:
                        False)
  --trained-model-info-file TRAINED_MODEL_INFO_FILE
                        path to trained model from initial non-incremental run
                        (required, only for incremental runs)
  --existing-blocking-index-file OG_BLOCKING_INDEX_FILE
                        path to existing blocking index from previous run
                        (optional, only for incremental runs)
  --enable-lprof        generate the line_profiler files for certain methods

namematch commands:
  {run}                 available commands
    run                 Run all namematch steps
```

