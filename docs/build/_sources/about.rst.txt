About Name Match
================

Name Match is a tool for probabilistically linking the records of individual entities (e.g. people) within and across datasets.

The code is optimized for linking people in criminal justice datasets (arrests, victimizations, city programs, etc.) using at least first name, last name, and date of birth (some dob-missingness is tolerated). Other data fields, such as race, gender, address, and zipcode, can be included to strengthen the match as available, especially when date of birth information is missing.


What is Name Match doing?
#########################

The goal of Name Match is to group all records referring to the same person within and across datasets. To do so, it makes a prediction for whether pairs of records refer to the same person. The prediction is based on the similarity of demographic factors across the two records. The similarity of two fields is computed in different ways depending on the type of information in the field (i.e. string, date, categorical variable). Edit distance between first name fields, number of days between dob fields, and whether or not the zipcode fields exactly match are all examples of similarity metrics that might be generated. The prediction is supervised, meaning the model learns what a match looks like based on labeled examples of record pairs that do and do not refer to the same entity. High probability links are then chained together to form *clusters*, or groups of records that all refer to the same entity (e.g. if records A and B match, and records B and C match, then records A, B, and C all refer to the same entity). 

No really, what is Name Match specifically doing, step by step?
***************************************************************

Here's a brief outline of the process:

.. rst-class:: spaced-list

    * **Combine the data**: First, all of the input CSVs are combined into a single file called the "all names" file
    * **Blocking**: If Name Match made comparisons between every record pair, the number of comparisons would run into the billions or trillions. We can drastically cut runtime by eliminating the long list of obvious "non-match" pairs. For example, a record with name "JOHN SMITH" does not need to be fully compared to a record with name "MARISSA LOPEZ." Blocking identifies record pairs that have a reasonable chance of referring to the same entity based on *just* the first name, last name, and date of birth. To do this, it first uses ``nmslib`` to quickly identify a name's "approximate nearest neighbors" based on cosine distance, and then filters out name pairs with very dissimilar dobs. For records with missing dob, this filtering step uses age. Only the pairs with reasonably similar name and dob (or age) move forward to the next step. 
    * **Modeling**: Using the labeled data, we train a random forest model that predicts the probability that two records are a match. Then, we make a prediction for each one of the candidate pairs identified in the blocking step. While blocking just looked at the name and date of birth, this prediction will use every field available.
    * **Clustering**: Now that there is information about pairs of records and their likelihood of belonging to the same entity, we can group all of an entity's records together. If you think of each record as a node in a network, this step identifies connected components (e.g. links A-B, B-C, and D-E produce two clusters: {A,B,C} and {D,E}). Clusters are created one link at a time. As each link is added, the validity of the link itself, and of the resulting cluster, is checked against constraints defined by the user. Once all the edges have been considered, each cluster gets a unique id.
    * **Output**: Lastly, Name Match will write a copy of each of the input files with an additional column indicating the entity to which each record belongs.

**Incremental runs:** There's a special type of Name Match run called an "incremental" run. This is where you have some data that has already been matched (assigned person ids, or "clusters") and you want to match more data to these same ids. See the "Incremental runs" section below if this is the type of match you are doing. However, keep in mind that first time use of Name Match will almost always be "from scratch" as opposed to incremental.


Where do the labeled examples come from?
****************************************

In order to use Name Match, one of the input datasets must have multiple rows per entity, and contain an entity-level unique id (at least for a subset of rows). Within an entity, there also must be some discrepencies in demographic information between records (e.g. typos, nicknames, new addresses, etc.). In the criminal justice context, arrest data typically meets this requirement. Consider the example of linking arrest data to victimization data (crimes). When we compare an arrest record to a different arrest record, we know whether or not the records refer to the same person because each person who has been arrested has a unique "arrestee" id. For a given arrestee id, one record might say MIKE SMITH and another might say MICHAEL SMITH. We use these arrest/arrest pairs to train the model, and then we can predict the probability that a victim/victim pair or a victim/arrest pair is a match. 

Inputs and outputs
##################

Inputs:

* One or more CSV files to be deduplicated and/or linked
* Configuration file specifying which data fields to use (where to find them, what type they are, etc.)
* (Optional) Functions written in Python 3 that determine whether a grouping of records is valid or not 

Outputs:

* Copy of the CSV files used as input, now with an additional "cluster id" column. This id is a unique identifier linking records to entities within and between files.

See "Setting up a match" for specific information about preparing the inputs.