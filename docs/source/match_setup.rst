Setting Up a Match
==================

Preparing the data for Name Match
#################################

The following important steps need to be taken before inputting CSVs to Name Match:

.. rst-class:: spaced-list

    * Ensure that categorical variables across CSVs are the same. If one CSV labels a female person as ``F``, and a second CSV labels her as ``fem``, then Name Match will not recognize these fields as a match. 
    * Standardize date formats across files. If dob is represented in one dataset as ``%Y-%m-%d`` and ``%d%B%y`` in another, select one and convert the other prior to running Name Match.
    * Drop any place holders that occur *across multiple fields*. If you want to drop all records where the first name is ``NA``, this can be listed in the config, and Name Match will take care of that automatically (see below). However, if you want to drop all records where the first name is "POLICE" *and* the last name is "OFFICER", this needs to be taken care of before running Name Match. Dropping records that don't quite map to a person (e.g. businesses, municipalities, officers) during pre-processing can greatly improve the runtime and quality of the match. NOTE: John Doe and Jane Doe will be automatically handled by the code.
    * Identify other place holder values and either set them to ``NA`` prior to Name Match or include them in that variable's ``set_missing`` list in the config (see below). A missing date of birth is commonly encoded as, for example, '1800-01-01' rather than ``NA``, which will lower Name Match's performance. Missing unique identifiers are sometimes encoded as ``UNKNWN`` or ``99999999`` which would result in a lot of incorrectly labeled training data.
    * Create an 'age' column that uses a single reference date (i.e. "age" is really "age as of 2025-01-01"). If a person is 18 in an arrest that happened in 2010 and 26 in victimization record from 2016, we don't want the algorithm to see 18 and 26 and assume it's not the same person. 


Creating the configuration file
###############################

The config contains instructions for how Name Match will read the data. You can see samples in the ``config_examples`` directory. Below is an overview of each of the fields.

Input data files
****************

Each input CSV should have a short descriptive nickname, like ``victim`` or ``arrests``. These nicknames serves as the keys in a dictionary of information about each input file. The following information is needed for each input data file:

* ``filepath``: Path to where the CSV is stored
* ``record_id_col``: Column that uniquely identifies a row (cannot contain nulls)
* ``delim``: The character that separates each field
* ``cluster_type``: String that describes the type of entity, such as victim, offender, student (default: "cluster")
* ``output_file_stem``: String used to name the file output by name match, ``<output_file_stem>_with_clusterid.csv`` (default: file nickname)

Variable structure
******************

Name Match needs to know which fields in the input data files contain information that is relevant to matching. In this section, you must define the fields you want Name Match to have access to, how to use them (i.e. ``compare_type``), and what they're called in each of the input data files.

* ``name``: The name of the column in the single file that Name Match constructs.
* ``compare_type``: This determines what distance-metric will be used to compare this field in two records. (One of ``String``, ``Date``, ``Categorical``, ``Address``, or ``UniqueID``)
* ``<nickname>_col``: These fields tells Name Match where to find this variable in the input CSVs, using the short, descriptive nickname established in the ``input data files`` section. If we run Name Match with two input CSVs, nicknamed ``arrests`` and ``victim``, then each variable listed in the ``variable structure`` section will need to have a field called ``arrests_col`` and one called ``victim_col``, which tell Name Match the column names to read in from the input CSVs. If one of the CSVs doesn't have this information (e.g. usually when the ``compare_type`` is ``UniqueID``), assign an empty string to this field in the config. 
* ``drop``: A list of values. If a record has one of these values in this column, that record will be dropped and ignored by Name Match
* ``set_missing``: A list of values. If a field has one of these values in this column, it will be replaced with a ``NA``
* ``check``: For a ``Date``, this will be in the format ``Date - format_string``, and records that don't have the date formatted in this way will be replaced with ``NA``. For a ``Categorical`` variable, this should be a list of the values that are allowed (ex, ``M,F``), and records with something other than one of these values in this field will be replaced with ``NA``.

**The default version of Name Match requires at least the following four variables:** ``first_name``, ``last_name``, ``dob``, and ``age``. Some missingness is tolerated for the dob and age fields.

Note, first and last name fields will be cleaned automatically (e.g. non-letter characters removed).  

UniqueID variables
******************

Variables specified with ``compare_type : UniqueID`` are used to create the labeled dataset for the prediction model. So the UniqueID variable should be a **person-identifier**, and is often only available in a subset of the data (otherwise you woulnd't need Name Match). For example, say you're matching two dataset where one is linked by SSN and the other has no person-identifier. SSN should be the UniqueID for this match. 

**Multiple UniqueIDs:** *In almost all use cases, only one UniqueID variable should be specified. The ability to handle multiple UniqueIDs is an advanced Name Match feature, with very litte testing.* If you specify multiple UniqueIDs, then links will be made between records if *either* of their UniqueIDs match. For example, if a record has SSN 123 and Fingerprint No. 456 and another record has SSN 123 but Fingerprint No. 052, a link will be generated. In cases with conflicts like this, the first UniqueID variable will be prioritized.


General params
**************

* ``verbose``: Number controlling how frequently progress is logged (if verbose is set to 50000, then a message will print every 50000 records/pairs processed)
* ``num_threads``: The number of threads that should be used to parallelize processes
* ``allow_clusters_w_multiple_unique_ids``: Flag indicating if the final groupings of records into people, or clusters, can contain more than one unique value for a given ``UniqueID`` field. If you trust that your unique id data is highly accurate, you'd set this to False. However, if you think it's possible for some people to have multiple values (for example, a student moves schools and is accidentally assigned a new student_id), then you can set this to True. 
* ``leven_thresh``: Sometimes the unique id used to label data will be filled with typos. To minimize the issues this might cause, this parameter flips 0 labels (not a match) to "unknown" labels if two ``UniqueID`` values have an edit distance that is less than or equal to ``leven_thresh``. This essentially moves record pairs from the training data to the unlabeled test set. 
* ``pct_train``: The percentage of labeled data that should be used to train the model. Setting this below 1 allows us to evaluate performance on a labeled test set. 


Creating user defined clustering functions
##########################################

Aspects of the Name Match algorithm are specific to the input data, in a way that is difficult to encode abstractly. To make Name Match more flexible, we provide the ability for the user to write domain-specific constraints (logic that determines if a final cluster is valid or invalid). For example, matches that link person-level data (e.g. program participants) to event-level data should encode an "at most one program participant record per cluster" rule.

There are four functions defined in ``cluster_logic.py`` that can be edited before running Name Match. The types *must* be as described below, or Name Match will either crash or produce incorrect results. The variable names that are used should correspond to the ``name`` field of the config, rather than the ``nickname_col`` fields.

``is_valid_edge()``
*******************

This function takes in two records and returns ``True`` if linking them would be valid (not violate a constraint), ``False`` otherwise. Each record is represented as a ``pandas Series``, through which you can access its values for fields like dob and address. 

Let's say that you want to prohibit links between homicide victimizations and records with later date (e.g. a person cannot get arrested after being a homicide victim). You could enforce this using the is_valid_edge function like so:
::

    def is_valid_edge(record1, record2, phat=None):

        homicide_date = None
        if record1['dataset'] == 'HOMICIDE_VICTIMS': 
            homicide_date = record1['date']
            other_date = record2['date']
        if record2['dataset'] == 'HOMICIDE_VICTIMS': 
            homicide_date = record2['date']
            other_date = record1['date']
        if homicide_date: 
            if other_date > homicide_date: 
                return False

        return True


If there is no special logic you wish to encode, this function should simply return ``True``.

You'll notice that the ``phat``, or probability that the two records belong to the same person, is also passed to this function. This information might be useful if, for example, you want to apply looser constraints to edges with a very high phat than those with a lower phat.

``is_valid_cluster()``
**********************

This function takes in a ``pandas DataFrame`` with information about a potential cluster and returns ``True`` if this cluster is valid, ``False`` otherwise. A common situation that occurs is a cluster will end up with multiple unique IDs. Consider three records: record A has unique id of 5, record B has a unique ID of 8, and record C is from the unlabeled data and has no unique id. Our model predicts that A and C are almost certainly a match, and B and C are almost certainly a match. Now we have a cluster with records A, B, and C that has more than one unique identifier (5 and 8). You can use ``is_valid_cluster`` to prevent this from happening like so:

*Note: the input dataframe denotes missing values as the empty string, rather than NaN*
::

    def is_valid_cluster(cluster, phat=None):

        # count the number of unique IDs that are in this cluster
        n_unique_ids = cluster["unique_id"].nunique()

        # only one is allowed (or 0, if it's all missing values)
        return (n_unique_ids <= 1)


If there is no special logic you wish to encode, this function should simply return ``True``.

You'll notice that the optional phat parameter is passed to ``is_valid_cluster`` as well.

``get_columns_used()``
**********************

This function tells Name Match which columns to read in (and as what type) when enforcing cluster constraints. For example, if you reference "dob" in ``is_valid_edge`` or ``is_valid_cluster``, then "dob" needs to be listed in this dictionary. Type options are ``object`` (string), ``float``, or ``date``. Note, ``fillna('')`` will happen automatically for columns that are read in as type ``object``.
::

    def get_columns_used():
        type_dict = {
            "age": 'float',
            "dob": 'date'
        }
        return type_dict

If there is no special logic you wish to encode in ``is_valid_edge`` or ``is_valid_cluster``, this function should simply return ``{}``.

``apply_edge_priority()``
*************************

This function takes in a ``pandas DataFrame`` of potential edges (links between two records). During clustering, Name Match iterates through this list and allows the edges that don't violate a constraint (using is_valid_edge and is_valid_cluster above). The purpose of this function is to allow the user to change the priority of an edge by increasing or decreasing its phat and/or choosing the order in which potential edges are considered. **The vast majority of the time, this function will not need to be changed.** The three columns present in this dataframe are ``record_id_1``, ``record_id_2``, and ``phat``. One example of what you could do in this function is boost the phats for edges that involve program records.
::

    def apply_edge_priority(edges_df, records_df=None):

        edges_df = edges_df.copy()

        boost = 1
        edges_df['involve_program_record'] = (
            edges_df.record_id_1.str.contains('program') | 
            edges_df.record_id_2.str.contains('program')).astype(int)
        edges_df.loc[edges_df.involve_program_record == 1, 'phat'] = edges_df.phat + boost

        edges_df = edges_df.sort_values(by=['phat', 'original_order'], ascending=[False, True])

        return edges_df

If there is no special logic you wish to encode, this function should simply return the input dataframe, ``edges_df``, but sorted in descending order of P(match), or ``phat``. This is the default behavior.

The optional ``records_df`` parameter refers to a dataframe of all the records being linked. It can be merged onto the edges_df if your ideal priority logic requires information from the records (e.g. if you want to manually reduce the P(match) by some percentage for rows with missing DOB).



Running the code
################

After setting up your virtual environment, preparing your data, filling out the config, and establishing the user defined functions, the last step is to update the ``makefile``. The ``config_file`` variable needs to be updated to wherever you stored your config file. If you created custom cluster contraint logic, the  ``cluster_constraints_file`` variable needs to be updated as well. Then, you can run the code by navigating into the repo and typing ``make``. That's it!

After the code finishes, the final output CSVs can be found in the ``output_dir`` directory defined in the makefile. A log and intermediate output files can be found in the ``output_dir``\_temp. 



Special cases
#############

Incremental runs
****************

An "incremental" run of Name Match is where you have some data that has already been matched (assigned person ids, or "clusters") and you want to match more data to these same ids. This type of run will very rarely be required the first time you use Name Match. If you do need to do an incremental run, however, then there are a few additional config requirements and parameters to be aware of: 

.. rst-class:: spaced-list

    * In addition to defining the ``data_files`` to match during the run, you need to define the set of ``existing_data_files`` in your config -- these are the file(s) that already have a person_id assigned by a previous run of Name Match. This section of the config is set up almost identically to the ``data_files`` section, however you don't need to define a ``cluster_type`` or ``output_file_stem`` for these files. Keep in mind that these existing files still need to have corresponding ``_col`` definitions in the variables section. If you are using the all-names output of a previous Name Match run as the existing data file for an incremental run, the record_id column likely already contains a prefix such as "arrests__XXXX." If that is the case, then feel free to set the 'use_record_id_as_is' parameter to True.

    * You must add a variable with ``compare_type : ExistingID`` (set up in the same way as all of the other variables) that indicates which column is the person id to match to (e.g. cluster_id). 
    * Incremental runs don't take the time to re-learn the match model, so you must provide the path to the "model info" file produced in the original "from scratch" run (this model info file itself contains the path to the trained model). This can be accomplished by changing the ``trained_model_info_file`` path in the makefile (to no longer be None).
    * (Optional) Incremental runs can take advantage of the already-built blocking index from the original run as well, which can sometimes yeild runtime savings. If you would like to do this, then change the ``og_blocking_index_file`` variable in the makefile from None to the path to the ``blocking_index.bin`` file produced in the original run. 
    * Because  no model training happens during incremental runs, it is unnecessary to include the ``pct_train`` parameter in the config. 

It is important to realize that people (record clusters) in the ``existing_data_files`` files are fixed -- they can acquire more records during subsequent incremantal Name Match runs, but they cannot lose any of their original records. And two people from the existing data files cannot merge to become one person during incremental runs, even if you change the cluster constraints to be more lax during the incremental run. Note, imposing new cluster constraints during incremental runs that are stricter than the original constraints can prohibit new records from getting added to existing clusters if not coded carefully. 
