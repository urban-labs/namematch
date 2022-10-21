.. _understanding-results:

Understanding Results
=====================

During the run, Name Match will log information about the matching process: tracking execution, reporting performance metrics, and flagging any issues. In addition to printing in the console, this log is written to file and can be found at ``output/details/name_match.log``. 

The log contains several different metrics indicating how successful the match was, and should be checked after the match is finished to ensure high quality. Below is a breakdown of the terms and metrics you will see -- what they mean and what values are reasonable. 

Blocking
********

**Terms:**

* True pairs: pairs of first_name/last_name/dob values that we know refer to the same entity based on the ``UniqueID``
* Covered pairs: pairs of first_name/last_name/dob values that make it through the blocking stage
* Uncovered pairs: true pairs that don't make it through the blocking stage (the fewer the better)

**Metrics:**

* Pair completeness: share of true pairs that are covered (the bigger the better, max 1)

  * Including equal :ref:`blockstrings <take me to blockstrings>` : 0.90 in our experience with arrest records
  * For non-equal blockstrings: > 0.75 in our experience with arrest records

Modeling
********

**Terms:**

* Threshold: which P(match) -- ``phat`` -- threshold is being used to classify a pair as a "match" vs. "non-match" 

**Metrics:**

* Base rate: what fraction of record pairs with ground truth labels are a match?
* Various typical classifier performance metrics (e.g. precision, recall, f1, auc): out-of-sample metrics reported. These metrics can be computed out-of-sample due to the heldout labeled data available when ``pct_train`` is less than 1. 

Clustering
**********

**Terms:**

* Cluster: a group of records all referring to the same entity (person). Every record in a cluster will get the same person identifier. 
* Invalid links: record pairs that wanted to get clustered together, based on P(match), but couldn't because it would have caused an edge constraint violation
* Invalid clusters: record pairs that wanted to get clustered together, based on P(match), but couldn't because it would have caused a cluster constraint violation
* Singleton clusters: records that did not match any other record (so are now in a cluster by themselves)

**Metrics:**

* Number of invalid predicted links skipped over during record linkage
* Number of times an invalid cluster was prevented during clustering
* Number of merges, or number of predicted links that were valid and produced valid clusters
* Number of singleton clusters in the final set of clusters 
* Number of clusters, or people, discovered across the input dataset(s)
