
Understanding Results
=====================

During the run, Name Match will log information about the matching process: tracking execution, reporting performance metrics, and flagging any issues. In addition to printing in the console, this log is written to the name_match.log file. 

The name match report contains several different metrics indicating how successful the match was, and should be checked after the match is finished to ensure high quality. Below is a breakdown of the terms and metrics you will see -- what they mean and what values are reasonable. 

**Blocking**

* True pairs: first_name/last_name/dob string pairs that we know refer to the same entity based on the ``UniqueID``
* Covered pairs: first_name/last_name/dob string pairs that do make it through the blocking stage
* Uncovered pairs: true pairs that don't make it through the blocking stage (the fewer the better)
* Pair completeness: share of true pairs that are covered (the bigger the better, max 1)

  * Including equal blockstrings: > 0.90 in our experience
  * For non-equal blockstrings: > 0.75 in our experience

**Modeling**

* Threshold: which phat threshold is being used to classify a pair as a "match" vs. "non-match" (expected range: 0.6-0.85)
* Base rate: what fraction of labeled pairs are a known match?
* Various typical classifier performance metrics (e.g. precision, recall, f1, auc): out-of-sample metrics reported (precision and recall should be > 0.9 and auc should be > 0.95)

**Clustering**

* Cluster: a group of records all referring to the same entity (person)
* Invalid edges: record pairs that wanted to get clustered together, based on phat, but couldn't because it would have caused an edge logic violation
* Invalid clusters: record pairs that wanted to get clustered together, based on phat, but couldn't because it would have caused a cluster logic violation
* Merges: record pairs that were allowed to form
* Singleton clusters: records that did not match any other record (so are now in a cluster by themselves)

**Error Rates**

These statistics have to do with the blocking metrics (pair completeness or "covered rate") and are broken down by category (for the variables with ``compare_type: 'Categorical'`` in your config).
