# Name Match

## About the Project

Tool for probabilistically linking the records of individual entities (e.g. people) within and across datasets.

The code was originally developed for linking records in criminal justice-related datasets (arrests, victimizations, city programs, school records, etc.) using at least first name, last name, date of birth, and age (some missingness in DOB and age is tolerated). If available, other data fields like middle initial, race, gender, address, and zipcode can be included to strengthen the quality of the match.

Project Link: https://urban-labs.github.io/namematch/

## Getting Started

### Installation
```
pip install namematch 
```

Name Match has been tested using Python 3.7 and 3.8, on both linux and Windows systems. Note, Name Match will not currently work using Python 3.9 on Windows because of the dependency on NMSLIB.


### Reference

* [Examples](https://github.com/urban-labs/namematch/tree/main/examples)
* [End-to-end tutorial](https://github.com/urban-labs/namematch/blob/main/examples/end_to_end_tutorial.ipynb)
* [Overview + usage docs](https://urban-labs.github.io/namematch/about.html)
* [Algorithm docs](https://urban-labs.github.io/namematch/algorithm.html)
* [Developer docs](https://urban-labs.github.io/namematch/api.html)


### Requirements of the input data

Name Match links records by learning a supervised machine learning model that is then used to predict the likelihood that two records "match" (refer to the same person or entity). To build this model the algorithm needs training data with ground-truth "match" or "non-match" labels. In other words, it needs a way of generating a set of record pairs where it knows whether or not the records should be linked. Fortunately, if a subset of the records being input into Name Match already have a unique identifier like Social Securuity Number (SSN) or Fingerprint ID, Name Match is able to generate the training data it needs. 

To see an example of this, say you are linking two datasets: dataset A and dataset B. People in dataset A can show up multiple times and can be uniquely identified via SSN. People in dataset B cannot be uniquely identified by any existing data field (hence the reason for using Name Match). If John (SSN 123) has two records in dataset A, we have found an example of two records that we know are a match. If Jane (SSN 456) also has a record in dataset A, we have found an example of two records that we know are NOT a match (Jane's record and either of John's records). Already we are on our way to building a training dataset for the Name Match model to learn from.

To facilitate the above process and make using Name Match possible, **a portion of the input data must meet the following criteria**: 
* Already have a unique person or entity identifier that can be used to link records (e.g. SSN or Fingerprint ID)
* Be granular enough that some people or entities appear multiple times (e.g. the same person being arrested two or three times)
* Contain inconsistencies in identifying fields like name and date of birth (e.g. arrested once as John Browne and once as Jonathan Brown)


## Usage

### Package usage

```python
config = {
    
    'data_files': {
        'datasetA': {
            'filepath' : '../preprocessed_data/datasetA.csv',
            'record_id_col' : 'record_id'
        },
        'datasetB': {
            'filepath' : '../preprocessed_data/datasetB.csv',
            'record_id_col' : 'record_num'
        }        
    },
    
    'variables': [
        {
            'name' : 'first_name',
            'compare_type' : 'String',
            'datasetA' : 'first_name',
            'datasetB' : 'fname',
        }, {
            'name' : 'last_name',
            'compare_type' : 'String',
            'datasetA' : 'last_name',
            'datasetB' : 'lname',
        }, {
            'name' : 'dob',
            'compare_type' : 'Date',
            'datasetA' : 'date_of_birth',
            'datasetB' : 'dob',
        }, {
            'name' : 'social_security_number',
            'compare_type' : 'UniqueID', 
            'datasetA' : 'ssn',
            'datasetB' : ''
        }
    ]
}

nm  = NameMatcher(config=config)
nm.run()
```

See `examples/end_to_end_tutorial.ipynb` or `examples/python_usage/link_data.py` for a full runnable example.


### Command line tool usage

```
cd examples/command_line_usage/
namematch --config-file=config.yaml --output-dir=nm_output --cluster-constraints-file=constraints.py run
```

For more details, please checkout [`examples/command_line_usage/README.md`](examples/command_line_usage/README.md).


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

Distributed under the GNU Affero General Public License v3.0 license. See LICENSE for more information.

## Team

Melissa McNeill, UChicago Crime and Education Labs

Eddie Tzu-Yun Lin, UChicago Crime and Education Labs

Zubin Jelveh, University of Maryland

## Citation

If you use Name Match in an academic work, please give this citation:

Zubin Jelveh, Melissa McNeill, and Tzu-Yun Lin. 2022. Name Match. https://github.com/urban-labs/namematch.
