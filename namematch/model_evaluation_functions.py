import numpy as np
import pandas as pd
from sklearn import metrics

import logging
logger = logging.getLogger()


def get_precision(eval_df, thresh):
    '''Compute precision.
 
    Args:
        eval_df (pd.DataFrame): contains three columns needed for performance metrics: phat, outcome, weight
        thresh (float): probability threshold that optimizes fscore

    Returns:
        float: precision
    '''

    predicted_positives = eval_df[eval_df.phat >= thresh].copy()
    if len(predicted_positives) == 0:
        return None
    precision = np.average(predicted_positives.outcome, weights=predicted_positives.weight)
    return precision


def get_recall(eval_df, thresh):
    '''Compute recall.

    Args:
        eval_df (pd.DataFrame): contains three columns needed for performance metrics: phat, outcome, weight
        thresh (float): probability threshold that optimizes fscore

    Returns:
        float: recall
    '''

    actual_positives = eval_df[eval_df.outcome == 1].copy()
    actual_positives['yhat'] = actual_positives.phat >= thresh
    if len(actual_positives) == 0:
        return None
    recall = np.average(actual_positives.yhat, weights=actual_positives.weight)

    return recall


def get_fscore(eval_df, thresh, beta):
    '''Compute fscore.

    Args:
        eval_df (pd.DataFrame): contains three columns needed for performance metrics: phat, outcome, weight
        thresh (float): probability threshold that optimizes fscore
        beta (float): ratio of recall weighting to precision weighting (e.g. 0.5 weights precision double)

    Returns:
        float: fscore
    '''

    precision = get_precision(eval_df, thresh)
    recall = get_recall(eval_df, thresh)
    fscore = None
    if precision and recall:
        fscore = (1 + (beta**2)) * ((precision * recall) / (((beta**2) * precision) + recall))

    return fscore


def get_accuracy(eval_df, thresh):
    '''Compute accuracy.

    Args:
        eval_df (pd.DataFrame): contains three columns needed for performance metrics: phat, outcome, weight
        thresh (float): probability threshold that optimizes fscore

    Returns:
        float: accuracy
    '''

    eval_df['correct'] = eval_df.outcome == (eval_df.phat >= thresh).astype(int)
    if len(eval_df) == 0:
        return None
    accuracy = np.average(eval_df.correct, weights=eval_df.weight)

    return accuracy


def get_fpr(eval_df, thresh):
    '''Compute false postive rate.

    Args:
        eval_df (pd.DataFrame): contains three columns needed for performance metrics: phat, outcome, weight
        thresh (float): probability threshold that optimizes fscore

    Returns:
        float: false positive rate
    '''

    actual_negatives = eval_df[eval_df.outcome == 0].copy()
    actual_negatives['yhat'] = actual_negatives.phat >= thresh
    if len(actual_negatives) == 0:
        return None
    fpr = np.average(actual_negatives.yhat, weights=actual_negatives.weight)

    return fpr


def pairwise_metrics(labeled_preds, threshold, phat_col, outcome, fscore_beta, weight):
    '''Generate evaluation metrics that are standard for classification problems (e.g. precision, auc).

    Args:
        labeled_preds (pd.DataFrame): df with labeled test phats and columns needed for evaluation (limited to a universe)
        threshold (float): probability threshold that optimizes fscore
        phat_col (str): phat column for evaluation
        outcome (str): outcome to evaluate
        fscore_beta (float): ratio of recall weighting to precision weighting (e.g. 0.5 weights precision double)
        weight (bool): should the evaluation metrics utilize the selection model weights

    Return:
        list of floats: base_rate, precision, recall, fpr, fnr, auc, accuracy, fscore
    '''
    labeled_preds = labeled_preds.copy()

    preds = labeled_preds[phat_col]
    labels = labeled_preds[outcome]

    if weight:
        weights = labeled_preds[phat_col.replace('_match_phat', '_weight')]
    else: 
        weights = [1 for i in np.arange(len(labeled_preds))]

    base_rate = np.average(labels, weights=weights)

    # Check if only one class is present to avoid ROC AUC warning
    if len(np.unique(labels)) < 2:
        auc = None
    else:
        auc = metrics.roc_auc_score(labels, preds, sample_weight=weights)

    eval_df = pd.DataFrame(data={
        'phat':preds, 
        'outcome':labels,
        'weight':weights
    })

    precision = get_precision(eval_df, threshold)
    recall = get_recall(eval_df, threshold)
    fpr = get_fpr(eval_df, threshold)
    fnr = 1 - recall
    fscore = get_fscore(eval_df, threshold, fscore_beta)
    accuracy = get_accuracy(eval_df, threshold)

    return base_rate, precision, recall, fpr, fnr, auc, accuracy, fscore


def find_best_threshold(df, beta, weight):
    '''Find the threshold that optimizes fscore.

    Args:
        df (pd.DataFrame): predicted and actual values
            =====================   =======================================================
            outcome                 whether the pair is a match or not
            phat                    predicted probability of match
            weight                  weight to use for evaluation 
            =====================   =======================================================
        beta (float): ratio of recall weighting to precision weighting (e.g. 0.5 weights precision double)
        weight (bool): should we use the selection model weights when picking a threshold?

    Returns:
        float: probability threshold that optimizes fscore
    '''

    df = df.copy()

    if not weight:
        df['weight'] = 1

    logger.info('Finding optimal threshold.')

    best_t = -1
    best_fscore = -1
    for t in range(99, 1, -1):
        try:
            fscore = get_fscore(df[['phat', 'outcome', 'weight']], t/100.0, beta)
        except:
            continue
        if (fscore is not None) and (fscore > best_fscore):
            best_fscore = fscore
            best_t = t / 100.0

    if best_t == -1:
        logger.error('Problem calculating optimal threshold.')
        raise ValueError

    return best_t

def get_cv_metrics(mod): 
    '''Create table of cv scoring metrics between train and test.

    Args: 
        mod (sklearn model): Fit gridsearch object

    Returns: 
        pd.DataFrame: train/test performance for each param set in grid

    '''

    cv_results = pd.DataFrame.from_records(mod.cv_results_['params'])

    cv_results['train_score'] = mod.cv_results_['mean_train_score']
    cv_results['test_score'] = mod.cv_results_['mean_test_score']
    cv_results['train_test_difference'] = \
            cv_results['train_score'] - cv_results['test_score']

    return cv_results


def evaluate_predictions(phat_df, model_type, phat_col, outcome, weight=False,
        default_threshold=0.5, missingness_model_threshold_boost=0.2,
        optimize_threshold=False, fscore_beta=1.0, demographic_variables=None, all_names_df=None,
        max_demographic_values=10, match_type_thresholds=None):
    '''Calculates metrics such as precision, recall, fscore, etc. for the pairwise record
    match predictions. Also, get the threshold that maximizes f score.

    Args:
        phat_df (pd.DataFrame): df with labeled test phats and columns needed for evaluation
        model_type (str): match or selection
        phat_col (str): phat column for evaluation
        outcome (str): outcome to evaluate
        weight (bool): should the evaluation metrics utilize the selection model weights
        default_threshold (float): threshold for match/non-match (use if don't find optimal)
        missingness_model_threshold_boost (float): value to add to default threshold if missingess model
        optimize_threshold (bool): should we find the threshold that optimizes f1
        fscore_beta (float): ratio of recall weighting to precision weighting (e.g. 0.5 weights precision double)
        demographic_variables (list): list of demographic/categorical variable names for subgroup analysis
        all_names_df (pd.DataFrame): all_names dataframe with demographic columns and record_id
        max_demographic_values (int): maximum number of unique values for a demographic variable to report performance by category (default: 10)

    Returns:
        float: probability threshold that optimizes fscore
        dict: keys are universe str, values are dicts with perf metrics
    '''
    phat_df = phat_df.copy()

    all_model_stats = {}

    # hacky for now
    if 'no_' in phat_col and 'basic' not in phat_col:
        default_threshold = default_threshold + missingness_model_threshold_boost

    weight_col = phat_col.replace(f'{model_type}_phat', 'weight')
    if not weight:
        phat_df[weight_col] = 1
    
    if outcome != 'match_train_eligible':
        phat_df = phat_df[phat_df.match_train_eligible == 1]

    if len(phat_df) == 0:
        logger.info(f"Unable to measure performance (no labeled testing data available).")
        return default_threshold, all_model_stats

    logger.info(f'Number of labeled test rows evaluated: {len(phat_df)}')

    # Join demographic data if provided
    demographic_universes = []
    if demographic_variables and all_names_df is not None and len(demographic_variables) > 0:
        try:
            # Select only needed columns from all_names
            demographic_cols = ['record_id'] + demographic_variables
            demographic_cols = [col for col in demographic_cols if col in all_names_df.columns]
            all_names_subset = all_names_df[demographic_cols].copy()

            # Join demographics for both records in the pair
            phat_df = phat_df.merge(
                all_names_subset.rename(columns={col: f'{col}_1' if col != 'record_id' else 'record_id' for col in all_names_subset.columns}),
                left_on='record_id_1', right_on='record_id', how='left'
            ).drop(columns=['record_id'], errors='ignore')

            phat_df = phat_df.merge(
                all_names_subset.rename(columns={col: f'{col}_2' if col != 'record_id' else 'record_id' for col in all_names_subset.columns}),
                left_on='record_id_2', right_on='record_id', how='left'
            ).drop(columns=['record_id'], errors='ignore')

            # Build demographic universes dynamically from data
            for demog_var in demographic_variables:
                col1 = f'{demog_var}_1'
                col2 = f'{demog_var}_2'

                if col1 in phat_df.columns and col2 in phat_df.columns:
                    # Get unique values from both columns
                    values_1 = set(phat_df[col1].dropna().unique())
                    values_2 = set(phat_df[col2].dropna().unique())
                    all_values = values_1 | values_2

                    # Skip demographic variable if it has too many unique values
                    n_unique_values = len(all_values)
                    if n_unique_values > max_demographic_values:
                        logger.info(f"Skipping demographic variable '{demog_var}' ({n_unique_values} unique values > max {max_demographic_values})")
                        continue

                    # Create universe for each value (only if enough samples)
                    for value in all_values:
                        universe_name = f'{demog_var}:{value}'
                        # Count how many pairs have this demographic (either record)
                        mask = (phat_df[col1] == value) | (phat_df[col2] == value)
                        n_pairs = mask.sum()

                        if n_pairs >= 30:  # Minimum sample size threshold
                            demographic_universes.append(universe_name)
                        else:
                            logger.debug(f"Skipping demographic universe '{universe_name}' (only {n_pairs} pairs, minimum 30 required)")

            logger.info(f"Evaluating {len(demographic_universes)} demographic universes")
        except Exception as e:
            logger.warning(f"Could not join demographic data for subgroup analysis: {e}")

    if optimize_threshold:
        try:
            df = phat_df[[phat_col, outcome, weight_col]].copy()
            df.columns = ['phat', 'outcome', 'weight']
            threshold = find_best_threshold(df, fscore_beta, weight)
        except:
            threshold = default_threshold
            logger.warning(f'Could not find optimal threshold, using {threshold}.')
    else:
        threshold = default_threshold
    logger.info(f"Threshold: {threshold}")

    # Per-bucket threshold optimization (when match_type_thresholds is set):
    # find_best_threshold per universe subset and update the dict in-place.
    # Mirrors the per-row gate in predict.py so reported metrics match what
    # production gating will produce.
    if match_type_thresholds is not None:
        bucket_subsets = [('exact_all', phat_df[phat_df.exactmatch == 1]),
                          ('inexact_any', phat_df[phat_df.exactmatch == 0])]
        for bucket_name, subset_df in bucket_subsets:
            if bucket_name not in match_type_thresholds:
                continue
            if optimize_threshold and len(subset_df) > 0:
                try:
                    bucket_df = subset_df[[phat_col, outcome, weight_col]].copy()
                    bucket_df.columns = ['phat', 'outcome', 'weight']
                    optimum = find_best_threshold(bucket_df, fscore_beta, weight)
                    logger.info(
                        f"Per-bucket threshold optimum [{bucket_name}]: "
                        f"{optimum:.3f} (was {match_type_thresholds[bucket_name]:.3f}); "
                        f"updating match_type_thresholds in place"
                    )
                    match_type_thresholds[bucket_name] = optimum
                except Exception as e:
                    logger.warning(
                        f"Could not optimize threshold for bucket "
                        f"[{bucket_name}], keeping configured value "
                        f"{match_type_thresholds[bucket_name]}: {e}"
                    )
            else:
                logger.info(
                    f"Per-bucket threshold [{bucket_name}] manual value: "
                    f"{match_type_thresholds[bucket_name]:.3f} "
                    f"(optimize_threshold={optimize_threshold})"
                )

    # Build list of universes to evaluate
    base_universes = ['all pairs', 'non exactmatch pairs', 'exactmatch pairs']
    all_universes = base_universes + demographic_universes

    for universe in all_universes:

        model_stats = {}

        # Choose the threshold to apply when computing metrics for this
        # universe. Per-bucket thresholds (when configured) align reported
        # metrics with what predict.py will actually gate on.
        universe_threshold = threshold
        if match_type_thresholds is not None:
            if universe == 'exactmatch pairs' and 'exact_all' in match_type_thresholds:
                universe_threshold = match_type_thresholds['exact_all']
            elif universe == 'non exactmatch pairs' and 'inexact_any' in match_type_thresholds:
                universe_threshold = match_type_thresholds['inexact_any']

        phat_univ_df = phat_df.copy()
        if universe == 'non exactmatch pairs':
            phat_univ_df = phat_df[phat_df.exactmatch == 0].copy()
        elif universe == 'exactmatch pairs':
            phat_univ_df = phat_df[phat_df.exactmatch == 1].copy()
        elif ':' in universe:  # Demographic universe (e.g., 'race:BRANCA')
            demog_var, demog_value = universe.split(':', 1)
            col1 = f'{demog_var}_1'
            col2 = f'{demog_var}_2'
            # Include pairs where either record has this demographic value
            phat_univ_df = phat_df[(phat_df[col1] == demog_value) | (phat_df[col2] == demog_value)].copy()

        # log phat distributions
        try:

            one_phats = phat_univ_df[phat_univ_df[outcome] == 1][phat_col]
            one_phat_dist = pd.Series(pd.cut(one_phats, np.arange(0, 1.1, .1))).value_counts(normalize=True, sort=False)
            logger.trace(f'Phat distribution of actual 1s ({universe}): \n{one_phat_dist.to_string()}')
            model_stats['phat_distribution_1s'] = list(one_phat_dist)

            zero_phats = phat_univ_df[phat_univ_df[outcome] == 0][phat_col]
            zero_phat_dist = pd.Series(pd.cut(zero_phats, np.arange(0, 1.1, .1))).value_counts(normalize=True, sort=False)
            logger.trace(f'Phat distribution of actual 0s ({universe}): \n{zero_phat_dist.to_string()}')
            model_stats['phat_distribution_0s'] = list(zero_phat_dist)

        except:
            logger.info(f"Issue calculating phat distributions ({universe}).")

        try:
            if len(phat_univ_df) == 0:
                baserate = precision = recall = fpr = fnr = auc = accuracy = fscore = None
            else:
                baserate, precision, recall, fpr, fnr, auc, accuracy, fscore = pairwise_metrics(
                    phat_univ_df, universe_threshold, phat_col, outcome, fscore_beta, weight)

            model_stats['threshold'] = float(universe_threshold) \
                if universe_threshold is not None else None

            # Only log performance metrics for base universes, not demographic subgroups
            # Demographic stats are still saved to stats_dict for the matching report
            is_demographic_universe = ':' in universe

            if not is_demographic_universe:
                logger.info(f"Base rate ({universe}): {baserate}")
            model_stats['baserate'] = float(baserate) if baserate is not None else None

            if not is_demographic_universe:
                logger.info(f"Precision ({universe}): {precision}")
            model_stats['precision'] = float(precision) if precision is not None else None

            if not is_demographic_universe:
                logger.info(f"Recall ({universe}): {recall}")
            model_stats['recall'] = float(recall) if recall is not None else None

            if not is_demographic_universe:
                logger.info(f"False positive rate ({universe}): {fpr}")
            model_stats['fp_rate'] = float(fpr) if fpr is not None else None

            if not is_demographic_universe:
                logger.info(f"False negative rate ({universe}): {fnr}")
            model_stats['fn_rate'] = float(fnr) if fnr is not None else None

            if not is_demographic_universe:
                logger.info(f"AUC ({universe}): {auc}")
            model_stats['auc'] = float(auc) if auc is not None else None

            if not is_demographic_universe:
                logger.info(f"F-score ({universe}): {fscore}")
            model_stats['fscore'] = float(fscore) if fscore is not None else None

        except:
            message = f"Issue with given threshold -- not all areas of confusion matrix present ({universe})."
            if universe != 'exactmatch pairs':
                logger.warning(message)
            else: 
                logger.debug(message)

        all_model_stats[universe] = model_stats

    return threshold, all_model_stats


def evaluate_models(phats_df, outcome, model_type, weight=False, default_threshold=0.5,
            missingness_model_threshold_boost=0.2, optimize_threshold=False, fscore_beta=1.0,
            stats_dict=None, demographic_variables=None, all_names_df=None, max_demographic_values=10,
            match_type_thresholds=None):
    '''Wrapper for evaluating different models (e.g. basic and no-dob) on different universes.

    Args:
        phat_df (pd.DataFrame): df with labeled test phats and columns needed for evaluation
        outcome (str): outcome to evaluate
        model_type (str): either "selection" or "match"
        weight (bool): should the evaluation metrics utilize the selection model weights
        default_threshold (float): threshold for match/non-match (use if don't find optimal)
        missingness_model_threshold_boost (float): value to add to default threshold if missingess model (use if don't find optimal)
        optimize_threshold (bool): should we find the threshold that optimizes fscore
        fscore_beta (float): ratio of recall weighting to precision weighting (e.g. 0.5 weights precision double)
        demographic_variables (list): list of demographic/categorical variable names for subgroup analysis
        all_names_df (pd.DataFrame): all_names dataframe with demographic columns and record_id
        max_demographic_values (int): maximum number of unique values for a demographic variable to report performance by category (default: 10)

    Return:
        dict: maps model name (e.g. basic, no-dob) to thresholds (no return type)
    '''
    model_names = [col.replace(f'_{model_type}_phat', '')
                   for col in phats_df.columns.tolist()
                   if f'_{model_type}_phat' in col]

    if model_type == 'selection':
        optimize_threshold = False

    thresholds = {}
    model_type_model_stats = {}
    for model_name in model_names:

        phats_to_eval_df = phats_df[phats_df.model_to_use == model_name]
        # NOTE: will cause problems if this universe isn't represented in the labeled data;
        #       use basic poplation as a backup
        if len(phats_to_eval_df) == 0:
            phats_to_eval_df = phats_df[phats_df.model_to_use == 'basic']

        phat_col = f'{model_name}_{model_type}_phat'

        logger.info(f'----- EVALUATING {model_name.upper()} {model_type.upper()} MODEL -----')
        thresholds[model_name], \
        model_type_model_stats[model_name] = evaluate_predictions(
                phats_to_eval_df, model_type, phat_col, outcome, weight,
                default_threshold, missingness_model_threshold_boost, optimize_threshold, fscore_beta,
                demographic_variables, all_names_df, max_demographic_values,
                match_type_thresholds=match_type_thresholds)

    stats_dict[f"model_stats__{model_type}"] = model_type_model_stats
    if model_type == 'match':
        stats_dict[f"model_thresholds__{model_type}"] = thresholds
        # When per-match-type thresholds are configured, surface the (possibly
        # optimized) per-bucket values alongside the single-threshold dict so
        # the matching report and predict.py can both consume them.
        if match_type_thresholds is not None:
            stats_dict[f"match_type_thresholds__{model_type}"] = dict(match_type_thresholds)

    if model_type == 'match':
        return thresholds
