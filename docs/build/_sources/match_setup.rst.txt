.. _setting-up-a-match:

Setting Up a Match
==================

Identifying which fields to use
###############################

Naturally, the dataset(s) you are using Name Match to link or deduplicate contains information about people. But exactly what information do they contain? What fields should Name Match rely on to compare two records and decide if they refer to the same person or not? Answering this question is the first step for setting up a match. 

Certain fields are required:
	* First name
	* Last name
	* Date-of-birth
	* Age
	* An existing person identifier that links people across a portion of your input records. This field -- called the **"Unique ID"** -- is special, and very important for Name Match to work properly. See :ref:`requirements-for-using-name-match` to understand why this is the case and where you might find this field. 

Other fields are optional, but can be helpful when comparing records to each other. Here are examples of fields that you could consider using in your match (but feel free to try others): 
	* Middle name or initial
	* Race
	* Gender
	* Address
	* Phone number

Some fields might be useful for defining constraints, or logic about what is definitely NOT a match. A few examples:
	* Date of record generation - used to invalidate links that don't make sense based on their chronological order
	* School the person currently attends - used to invalidate links that wouldn't make sense in the real world


Creating the configuration file
###############################

The config contains instructions for how Name Match will read the input data and complete the match. You can see more samples in the repo's ``config_examples`` directory, but it generally looks like this.

.. code-block:: yaml

    data_files :

        'victim' : 
            'filepath' : '/path/to/input/data/file1.csv'
            'record_id_col' : 'row_id'
            'delim' : ','
            'cluster_type' : 'victim'
            'output_file_stem' : 'victim'
    
        'arrests' :  
            'filepath' : '/path/to/input/data/file2.csv'
            'record_id_col' : 'arrest_id'
            'delim' : ','
            'cluster_type' : 'offender'
            'output_file_stem' : 'arrests' 

    variables :

        - 'name' : 'first_name'
          'compare_type' : 'String'
          'arrests_col' : 'FIRST_NME'
          'victim_col' : 'VFIRST'
          'drop' : [''," ",'REFUSED','FIRM','BUSINESS']

        - 'name' : 'last_name'
          'compare_type' : 'String'
          'arrests_col' : 'LAST_NME'
          'victim_col' : 'VLAST'
          'drop' : ['',' ','REFUSED','FIRM','BUSINESS']

        - 'name' : 'dob'
          'compare_type' : 'Date'
          'arrests_col' : 'BIRTH_DATE'
          'victim_col' : 'VDOB'
          'check' : 'Date - %Y-%m-%d'

        - 'name' : 'age'
          'compare_type' : null
          'arrests_col' : 'AGE_IN_2025'
          'victim_col' : 'age_2025'

        - 'name' : 'gender'
          'compare_type' : 'Categorical'
          'check' : 'M,F'
          'arrests_col' : 'SEX_CODE_CD'
          'victim_col' : 'VSEX'

        - 'name' : 'arrestee_id'
          'compare_type' : 'UniqueID'
          'arrests_col' : 'ARRESTEE_ID'
          'victim_col' : ''
          'set_missing' : ['',' ','NA','nan','REDACTED','SEALED']

    # general parameters
    num_workers : 6
    allow_clusters_w_multiple_unique_ids : False
    pct_train : 0.9


Below is an overview of each section of the config, and the parameters that can be used throughout.

Input data files
****************

Each input CSV should have a short descriptive nickname, like ``victim`` or ``arrests``. These nicknames serves as the keys in a dictionary of information about each input file. The following information is needed for each input data file:

* ``filepath``: *(str)* Path to where the CSV is stored
* ``record_id_col``: *(str)* Column that uniquely identifies a row (cannot contain nulls)
* ``delim``: *(str, default=",")* The character that separates each field
* ``cluster_type``: *(str, default="cluster")* String that describes the type of entity, such as victim, offender, student
* ``output_file_stem``: *(str, default=<file-nickname>)* String used to name the file output by name match, ``<output_file_stem>_with_clusterid.csv``

Variable structure
******************

Name Match needs to know which fields in the input data files contain information that is relevant to matching. In this section, you must define the fields you want Name Match to have access to, how to use them, and what they're called in each of the input data files.

* ``name``: *(str)* Short descriptive name describing the field, e.g. first_name.

* ``compare_type``: *(str)* The data type of the field. This :ref:`determines what similarily metrics will be used<compare-type-to-metric-mapping>` to compare the values of this field between two records. (One of ``String``, ``Date``, ``Categorical``, ``Address``, or ``UniqueID``)

* ``<dataset_nickname>_col``: *(str)* These fields tells Name Match where to find this variable in the input CSVs, using the short, descriptive nickname established in the ``input data files`` section. If we run Name Match with two input CSVs, nicknamed ``arrests`` and ``victim``, then each variable listed in the ``variable structure`` section will need to have a field called ``arrests_col`` and one called ``victim_col``, which tell Name Match the column names to read in from the input CSVs. If one of the CSVs doesn't have this information, assign an empty string to this field in the config. 

* ``drop``: *(list, default:[])* A list of values. If a record has one of these values in this field, that record will be ignored by Name Match.

* ``set_missing``: *(list, default:[])* If a field has one of these values in this column, it will be replaced with a ``NA``.

* ``check``: *(str, default=None)* For a ``Date``, this should be in the format ``Date - <format_string>`` (ex, ``Date - %Y-%m-%d``), and records that don't have the date formatted in this way will be replaced with ``NA``. For a ``Categorical`` variable, this should be a comma-separated string list of the values that are allowed (ex, ``M,F``), and records with something other than one of these values in this field will be replaced with ``NA``.

**The default version of Name Match requires at least the following three variables:** ``first_name``, ``last_name``, ``dob``, and ``age``. Some missingness is tolerated for the dob and age fields.

**One of the variables defined in the config must have compare_type: "UniqueID".** The ability to handle multiple UniqueIDs is an advanced Name Match feature, with *very little* testing. In almost all use cases, only one UniqueID variable should be specified.

Note, first and last name fields will be cleaned automatically (e.g. non-letter characters removed).  All string comparisons are case-insensitive, so it does not matter if input data is all caps, all lowercase, or a mix.

General parameters
******************

There are a number of different parameters that can be set to configure exactly how Name Match runs. The full list of parameters -- and their default values --  can be found `here <https://github.com/urban-labs/namematch/-/blob/master/namematch/default_parameters.yaml>`_. There are only a few parameters, however, that are somewhat common to adjust from the default values -- these parameters are defined below. To change a parameter from the default value, include the parameter and its desired value as a key-value pair in the config. 

* ``num_workers``: *(int, default=1)* The number of workers that should be used to parallelize various Name Match steps

* ``allow_clusters_w_multiple_unique_ids``: *(bool, default=False)* Determines if the final groupings of records into people, or clusters, can contain more than one unique value for a given ``UniqueID`` field. If you trust that your Unique ID data is highly accurate, you'd set this to False. However, if you think it's possible for some people to have multiple values (for example, a student moves schools and is accidentally assigned a new student_id), then you can set this to True.

* ``leven_thresh``: *(int, default=None)* Sometimes the Unique ID used to label data will contain typos. To minimize the issues this might cause, this parameter causes the ground truth label to be ignored for record pairs that are labeled "not a match" if their ``UniqueID`` values have an edit distance that is less than or equal to ``leven_thresh``. This essentially moves record pairs from the training data to the unlabeled set that requires prediction.

* ``pct_train``: *(float, default:=0.9)* The percentage of labeled data that should be used to train the model. Setting this below 1 allows us to evaluate performance on a held-out labeled test set. 

* ``missingness_model``: *(str, default='dob')* Variable that is so critical for matching that Name Match should train a separate model on the records with missing values for the variable if needed. By default, for example, if there are missing values in dob a "dob missingness model" will be built and used to estimate P(match) for record pairs with missing dob information.

* ``optimize_threshold``: *(bool, default=True)* Should the probability threshold for distinguishing predicted links from predicted non-links be determined programatically (to optimize F Score)? If False, the `default_threshold` will be used.

* ``default_threshold``: *(float, default=0.7)* If ``optimize_threshold=False`` or an error is encountered during optimization, what threshold should be used to distinguish predicted links from predicted non-links?


Pre-processing: making sure your data is ready for Name Match
#############################################################

The following important steps need to be taken before inputting CSVs to Name Match:

	* Ensure that categorical variables are defined identically across CSVs. For example, if one dataset encodes female as ``F`` and another encodes it as ``fem``, Name Match will not recognize that these fields are the same.

	- Standardize date formats across files. If DOB is represented in one dataset as ``%Y-%m-%d`` and ``%d%B%y`` in another, select one format and convert the other prior to running Name Match.

	* Drop any place holders that occur *across multiple fields*. If you want to drop all records where the first name is ``NA``, this can be listed in the config, and Name Match will take care of that automatically (see below). However, if you want to drop all records where the first name is "POLICE" *and* the last name is "OFFICER", this needs to be taken care of before running Name Match. Dropping records that don't quite map to a person (e.g. businesses, municipalities, officers) during pre-processing can greatly improve the runtime and quality of the match. NOTE: John Doe and Jane Doe will automatically be ignored during Name Match.

	- Identify other place holder values and set them to ``NA``. For example, if missing date of birth values are encoded as 1800-01-01 rather than NA, you should convert that value to NA during preprocessing so that two values with 1800-01-01 are not treated as identical, rather than unknown. It is especially important to do this for the field designated as the UniqueID, so that two records with values ``UNKNWN`` or `9999999` are not considered ground truth matches. If you prefer, you can also solve this problem by including placeholder values in the relevant "set_missing" list in the config.

	* Create an 'age' column that uses a single reference date (i.e. "age" is really "age as of 2025-01-01"). This is necessary because it is likely that the records associated wth a given person were not all generated on the same day. For example, if a person is 18 in an arrest that happened in 2010 and 26 in victimization record from 2016, we don't want the algorithm to see 18 and 26 and assume it's not the same person.


Creating user defined constraint functions
##########################################

Because Name Match is an imperfect and probabilistic tool, there may be times when two records that do not refer to the same person get linked together and placed in the same cluster. Sometimes these "false positive" links are very easy for a human to identify because they defy some real-world logic or violate rules that the user wishes to enforce based on their knowledge of the dataset or domain.

For example, say you are linking school enrollment records from a given year to identify students who have transfered -- and one of the data fields available is number of days the student attended classes at that school. If a link between two records implies that a student attended school for more than 365 days in a given year, a human would immediately know that that link should not be allowed.

To solve this problem, we provide the ability for the user to write custom, problem-specific constraints (logic that determines if a final link/cluster is valid or invalid). The user defines these constraints by writing python functions called``is_valid_link()`` and ``is_valid_cluster()``.

There are two options for how the user can pass these custom constraint functions to the NameMatcher object's ``cluster_constraints`` argument. Option 1: Pass the path to a standalone python script that contains these two functions. Option 2: Pass a ``ClusterConstraints`` object that defines these functions (see the very end of the `tutorial notebook <https://github.com/urban-labs/namematch/blob/master/examples/end_to_end_tutorial.ipynb>`_ for an example of this option).

Defining and using custom cluster constraints is *optional*. By default, the NameMatcher's ``cluster_constraints`` argument is left empty and `default constraints <https://github.com/urban-labs/namematch/blob/master/namematch/default_constraints.py>`_ that classify all links and clusters as valid are applied. 

``is_valid_link()``
*******************

This function takes in a ``pandas DataFrame`` of predicted links, or pairs of records that are predicted to match. It returns a boolean ``pandas Series`` indicating whether the predicted link is valid. The input dataframe has the following columns:

	* Information about both records, for example: ``first_name_1``, ``last_name_1``, ``dataset_1``, ..., ``first_name_2``, ``last_name_2``, ``dataset_2``, ...
	* ``phat`` (float):  the prediction from the model, or the probability that the two records belong to the same person

**Example:** Let's say that one of the datasets you are inputting to Name Match is a program roster that you know for a fact has at most one record per person. You want to add a constraint that says any link between two records in that dataset is invalid. You could enforce this constraint using the ``is_valid_link()`` function like so:
::

    def is_valid_link(predicted_links_df):

        predicted_links_df['valid] = True

        predicted_links_df.loc[
            (predicted_links_df.dataset_1 == 'program_roster') & 
            (predicted_links_df.dataset_2 == 'program_roster'), 
            'valid'] = False

        return predicted_links_df['valid']


If there is no special logic you wish to encode, this function should simply return True.


``is_valid_cluster()``
**********************

This function takes in a ``pandas DataFrame`` with information about a single potential cluster, i.e. a group of records that all refer to the same person. It returns ``True`` if this cluster is valid and ``False`` otherwise. This function is run near the very end of the record linkage process during the clustering step, which determines which records will end up with the same person identifier in the final output. If a cluster is deemed valid, the records in that cluster will all have the same person id.

**Example:** Let's again pretend that you are linking school enrollment records for a given year to identify transfer students. Equipped with domain knowledge about the school district you are linking, you may want to enforce a constraint that no student can attend more than 5 different schools in a single school year. You can use ``is_valid_cluster`` to prevent this from happening like so:
::

    def is_valid_cluster(cluster, phat=None):

        # count the number of unique schools in the cluster
        n_unique_schools = cluster["school_id"].nunique()
        is_valid = n_unique_schools <= 5

        return is_valid

If there is no special logic you wish to encode, this function should simply return ``True``.

Notice the optional ``phat`` parameter being passed into ``is_valid_cluster()``. This float is the prediction from the model, or the probability that the two records belong to the same person. This information might be useful if, for example, you want to apply looser constraints to links the model is more confident in.

**A quick note about missing values:** In the dataframes passed to ``is_valid_link()`` and ``is_valid_cluster()``, missing values will be encoded as ``np.NaN`` for ID columns, numeric columns, and date columns. Missing values will be encoded as empty strings ("") for string/object columns. For control over the data type each column is represetned as, see the section on the ``get_columns_used`` function below. 

**A quick note about variable names:** Notice how we reference a column called ``school_id`` in the ``is_valid_cluster`` example above. This is made possible via the Name Match config file where we defined a variable called `school_id`, like so:
::

    - 'name' : 'school_id'
      'compare_type' : null 
      'dataset_1_col' : 'SchoolID'
      'dataset_2_col' : 'SCHOOL_IDENTIFIER'

This then allows us to access the ``school_id`` field in ``is_valid_link`` and ``is_valid_cluster``. Also note that by specifiying a ``null`` compare type, we have indicated that we only want the field to be used in constraint-checking (**not in the prediction model**).


Additional user-defined functions
##################################

There are two additional user-defined functions that can be altered to customize how Name Match clusters potential links. 

``apply_link_priority()``
*************************

The ``apply_link_priority()`` function allows the user to change the order in which potential links are considered during the clustering step. By default, potential links are considered in descending order of ``phat`` -- that is, the links we consider first are those that have the greatest likelihood of being a match. In **very rare** cases, however, the user may wish to alter the ``phat`` values for certain potential links before sorting or to sort by something else entirely. 

This function takes in a ``pandas DataFrame`` of valid predicted links, or pairs of records that are predicted to match and passed the ``is_valid_link()`` criteria. During clustering, Name Match iterates through this list and approves or disapproves each link one by one according to the ``is_valid_cluster()`` function.

To see an example of how this function might be used, imagine the user wants to manually deprioritize predicted links that were missing DOB values. That could be done as follows:
::

    def apply_link_priority(valid_links_df):

        missing_dob_phat_penalty = 0.1
        valid_links_df['missing_dob'] = valid_links_df.dob_1.isnull() | valid_links_df.dob_2.isnull()
        valid_links_df.loc[valid_links_df.missing_dob == 1, 'phat'] = valid_links_df.phat - missing_dob_phat_penalty

        valid_links_df = valid_links_df.sort_values(by='phat', ascending=False)

        return valid_links_df


Another possible use case for this function is if the user wishes to consider potential links in chronological order rather than in order of descending confidence (which can be useful when linking data for RCTs or in other research settings).


``get_columns_used()``
**********************

This optional function tells Name Match which data fields are needed by the constraint functions (and what data type to read them in as). By default, all data fiels are read in. This function can be useful for limiting the memory usage of the clustering step. If for example, RAM is an issue and the only data field referenced in the user's custom constraint functions is "dob", the user an define this function to return {"dob":"date"} to limit the fields that are loaded to just dob. 

The ability to control the data type a field is read in as can be useful as the user writes their custom constraint functions. For example, if the ``is_valid_cluster()`` function contains logic related to the dob field, here the user can specify if they want this column to be represented as a string or a date object. By default, all columns are read in as strings. Type options are ``object`` (string), ``int``, ``float``, or ``date``.
::

    def get_columns_used():
        type_dict = {
            "age": 'float',
            "dob": 'date'
        }
        return type_dict

If there is no special logic you wish to encode in ``is_valid_link()`` or ``is_valid_cluster()``, this function should simply return ``{}``.


Special cases
#############

Incremental runs
****************

An "incremental" run of Name Match is where you have some data that has already been matched (assigned person ids, or "clusters") and you want to match more data to these same ids. There are two reasons you may want to do an incremental run of Name Match:
    1. Speed: If you've already linked 1 million records and need to link 1,000 more, it will be faster to incrementally add the 1,000 records to the pre-matched 1 million records than it would be to run Name Match from scrath on 1,001,00 records. 
    2. Cluster consistency: Because Name Match is a probabilistic tool, the links and clusters formed may change slightly between runs. This means if you link two dataset today and re-link them tomorrow, the records associated with Person X may vary between runs and the specific person identifier assigned to Person X will almost certainly differ between runs. Incremental Name Match can be used if you want to treat the original clusters as fixed. During an incremental run of Name Match, new records can only form brand new clusters or get added to existing (fixed) clusters. Clusters that are fixed cannot lose records, and two clusters that are fixed cannot merge to become one larger cluster.

This type of run will very rarely be required the first time you use Name Match. If you do need to do an incremental run, however, then there are a few additional config requirements and parameters to be aware of: 

    * In addition to defining the ``data_files`` to match during the run, you need to define the set of ``existing_data_files`` in your config -- these are the file(s) that already have a person identifier assigned to all rows by a previous run of Name Match. This section of the config is set up almost identically to the ``data_files`` section, however you don't need to define a ``cluster_type`` or ``output_file_stem`` for these files. Keep in mind that these existing files still need to have corresponding ``_col`` definitions in the variables section. If you are using the all-names output of a previous Name Match run as the existing data file for an incremental run, the record_id column likely already contains a prefix such as "arrests__XXXX." If that is the case, we recommend setting the 'use_record_id_as_is' parameter to True.

    - You must add a variable with ``compare_type : ExistingID`` (set up in the same way as all of the other variables) that indicates which column is the person id to match to (e.g. cluster_id). 
    
    * Incremental runs don't take the time to re-learn the match model, so you must provide the path to the "model info" file produced in the original "from scratch" run (this model info file itself contains the path to the trained model) as input.
    
    - (Optional) Incremental runs can take advantage of the already-built blocking index from the original run as well, which can sometimes yeild runtime savings. If you would like to do this, then pass the path to the ``blocking_index.bin`` file produced in the original run to the ``og_blocking_index_file`` parameter.
    
    * Because  no model training happens during incremental runs, it is unnecessary to include the ``pct_train`` parameter in the config. 

Special note on using cluster constraints during an incremental Name Match run: It is important to realize that people (clusters of records) in the ``existing_data_files`` files are fixed -- they can acquire more records during subsequent incremantal Name Match runs, but they cannot lose any of their original records. And two people from the existing data files cannot merge to become one person during incremental runs, even if you change the cluster constraints to be more lax during the incremental run. Imposing new cluster constraints during incremental runs that are stricter than the original constraints can prohibit new records from getting added to existing clusters if not coded carefully. 
