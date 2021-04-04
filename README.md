# Name Match

## About the Project

Tool for probabilistically linking the records of individual entities (e.g. people) within and across datasets.

The code is optimized for linking people in criminal justice datasets (arrests, victimizations, city programs, etc.) using at least first name, last name, and date of birth (some dob-missingness is tolerated). Other data fields, such as race, gender, address, and zipcode, can be included to strengthen the match as available, especially when date of birth information is missing.

Project Link: https://github.com/urban-labs/namematch

## Getting Started

### Installation

```
pip install namematch
```

### Reference

* [Examples](examples/)
* [End-to-end tutorial](examples/end_to_end_tutorial.ipynb)
* [Detailed usage docs](docs/source/match_setup.rst)
* [Algorithm docs](docs/source/algorithm.rst)
* [Developer docs](docs/source/api.rst)

## Usage

### Package usage

```python
config = {
    
    'data_files': {
        'dataset1': {
            'filepath' : '../preprocessed_tutorial_data/dataset1.csv',
            'record_id_col' : 'record_id'
        },
        'dataset2': {
            'filepath' : '../preprocessed_tutorial_data/dataset2.csv',
            'record_id_col' : 'record_num'
        }        
    },
    
    'variables': [
        {
            'name' : 'first_name',
            'compare_type' : 'String',
            'dataset1' : 'first_name',
            'dataset2' : 'fname',
        }, {
            'name' : 'last_name',
            'compare_type' : 'String',
            'dataset1' : 'last_name',
            'dataset2' : 'lname',
        }, {
            'name' : 'dob',
            'compare_type' : 'Date',
            'dataset1' : 'date_of_birth',
            'dataset2' : 'dob',
        }, {
            'name' : 'social_security_number',
            'compare_type' : 'UniqueID', 
            'dataset1' : '', 
            'dataset2' : 'ssn'
        }
    ]
}

nm  = NameMatcher(config=config)
nm.run()
```

See `examples/end_to_end_tutorial.ipynb` or `examples/python_usage/link_data.py` for a full runnable example.


### Command line tool usage

```
namematch --config-file=examples/command_line_usage/config.yaml --output-dir=output 
```

## Roadmap

See the [open issues](https://github.com/urban-labs/namematch/issues) for a list of proposed features (and known issues).

## Contributing

All contributions -- to code, documentation, tests, examples, etc. -- are greatly appreciated. For more detailed information, see CONTRIBUTING.md.
1. Fork the project
2. Create your feature branch (git checkout -b some-feature)
3. Commit your changes (git commit -m 'Add some amazing feature')
4. Push to the branch (git push origin some-feature)
5. Open a pull request

## License

Distributed under the XXXXXX license. See LICENSE for more information.

## Team

Melissa McNeill, UChicago Crime and Education Labs

Eddie Tzu-Yun Lin, UChicago Crime and Education Labs

Zubin Jelveh, University of Maryland

## Citation

If you use Name Match in an academic work, please give this citation:

Zubin Jelveh and Melissa McNeill. 2021. Name Match. https://github.com/urban-labs/namematch.

