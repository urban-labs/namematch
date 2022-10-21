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
        optimize_threshold=False, fscore_beta=1.0):
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

    for universe in ['all pairs', 'non exactmatch pairs', 'exactmatch pairs']:

        model_stats = {}

        phat_univ_df = phat_df.copy()
        if universe == 'non exactmatch pairs':
            phat_univ_df = phat_df[phat_df.exactmatch == 0].copy()
        elif universe == 'exactmatch pairs':
            phat_univ_df = phat_df[phat_df.exactmatch == 1].copy()

        # log phat distributions
        try:

            one_phats = phat_univ_df[phat_univ_df[outcome] == 1][phat_col]
            one_phat_dist = pd.value_counts(pd.cut(one_phats, np.arange(0, 1.1, .1)), normalize=True, sort=False)
            logger.trace(f'Phat distribution of actual 1s ({universe}): \n{one_phat_dist.to_string()}')
            model_stats['phat_distribution_1s'] = list(one_phat_dist)

            zero_phats = phat_univ_df[phat_univ_df[outcome] == 0][phat_col]
            zero_phat_dist = pd.value_counts(pd.cut(zero_phats, np.arange(0, 1.1, .1)), normalize=True, sort=False)
            logger.trace(f'Phat distribution of actual 0s ({universe}): \n{zero_phat_dist.to_string()}')
            model_stats['phat_distribution_0s'] = list(zero_phat_dist)

        except:
            logger.info(f"Issue calculating phat distributions ({universe}).")

        try:
            if len(phat_univ_df) == 0:
                baserate = precision = recall = fpr = fnr = auc = accuracy = fscore = None
            else:
                baserate, precision, recall, fpr, fnr, auc, accuracy, fscore = pairwise_metrics(
                    phat_univ_df, threshold, phat_col, outcome, fscore_beta, weight)

            logger.info(f"Base rate ({universe}): {baserate}")
            model_stats['baserate'] = float(baserate) if baserate else None

            logger.info(f"Precision ({universe}): {precision}")
            model_stats['precision'] = float(precision) if precision else None

            logger.info(f"Recall ({universe}): {recall}")
            model_stats['recall'] = float(recall) if recall else None

            logger.info(f"False positive rate ({universe}): {fpr}")
            model_stats['fp_rate'] = float(fpr) if fpr else None

            logger.info(f"False negative rate ({universe}): {fnr}")
            model_stats['fn_rate'] = float(fnr) if fnr else None

            logger.info(f"AUC ({universe}): {auc}")
            model_stats['auc'] = float(auc) if auc else None

            logger.info(f"F-score ({universe}): {fscore}")
            model_stats['fscore'] = float(fscore) if fscore else None

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
            stats_dict=None):
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
                default_threshold, missingness_model_threshold_boost, optimize_threshold, fscore_beta)
    
    stats_dict[f"model_stats__{model_type}"] = model_type_model_stats
    if model_type == 'match':
        stats_dict[f"model_thresholds__{model_type}"] = thresholds

    if model_type == 'match':
        return thresholds
