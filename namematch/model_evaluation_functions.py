
import numpy as np
import os
import pandas as pd
from sklearn import metrics
import sys

from namematch.utils.utils import log_runtime_and_memory

import logging
logger = logging.getLogger()

try:
    profile
except:
    from line_profiler import LineProfiler
    profile = LineProfiler()


def get_confusion_matrix(preds, labels, thresh):
    '''Compute confusion matrix. 

    Args: 
        preds (pd.Series): predicted probabilty of match (float), for a set of labeled record pairs
        labels (pd.Series): flag, 1 if the record pairs are actually a match 0 otherwise
        threshold (float): probability threshold that optimizes fscore

    Returns: 
        float: confusion matrix
    '''

    conf_matrix = pd.crosstab(preds >= thresh, labels)
    tp = conf_matrix[1][1]  # true positive
    fp = conf_matrix[0][1]  # false positve
    fn = conf_matrix[1][0]  # false negative
    tn = conf_matrix[0][0]  # true negatives
    return(tp, fp, fn, tn)


def get_precision(preds, labels, thresh):
    '''Compute precision. 

    Args: 
        preds (pd.Series): predicted probabilty of match (float), for a set of labeled record pairs
        labels (pd.Series): flag, 1 if the record pairs are actually a match 0 otherwise
        threshold (float): probability threshold that optimizes fscore

    Returns: 
        float: precision
    '''

    tp, fp, fn, tn = get_confusion_matrix(preds, labels, thresh)
    return(float(tp) / (tp + fp) if (tp + fp) else None)


def get_recall(preds, labels, thresh):
    '''Compute recall. 

    Args: 
        preds (pd.Series): predicted probabilty of match (float), for a set of labeled record pairs
        labels (pd.Series): flag, 1 if the record pairs are actually a match 0 otherwise
        threshold (float): probability threshold that optimizes fscore

    Returns: 
        float: recall
    '''

    tp, fp, fn, tn = get_confusion_matrix(preds, labels, thresh)
    return(float(tp) / (tp + fn) if (tp + fn) else None)


def get_fscore(preds, labels, thresh, beta):
    '''Compute fscore. 

    Args: 
        preds (pd.Series): predicted probabilty of match (float), for a set of labeled record pairs
        labels (pd.Series): flag, 1 if the record pairs are actually a match 0 otherwise
        threshold (float): probability threshold that optimizes fscore
        beta (float): ratio of recall weighting to precision weighting (e.g. 0.5 weights precision double)

    Returns: 
        float: fscore
    '''

    precision = get_precision(preds, labels, thresh)
    recall = get_recall(preds, labels, thresh)
    fscore = None
    if (precision + recall):
        fscore = (1 + (beta**2)) * ((precision * recall) / (((beta**2) * precision) + recall))
    return fscore


def get_accuracy(preds, labels, thresh):
    '''Compute accuracy. 

    Args: 
        preds (pd.Series): predicted probabilty of match (float), for a set of labeled record pairs
        labels (pd.Series): flag, 1 if the record pairs are actually a match 0 otherwise
        threshold (float): probability threshold that optimizes fscore

    Returns: 
        float: accuracy
    '''

    tp, fp, fn, tn = get_confusion_matrix(preds, labels, thresh)
    return(float(tp + tn) / (tp + tn + fn + fp))


def get_fpr(preds, labels, thresh):
    '''Compute false postive rate. 

    Args: 
        preds (pd.Series): predicted probabilty of match (float), for a set of labeled record pairs
        labels (pd.Series): flag, 1 if the record pairs are actually a match 0 otherwise
        threshold (float): probability threshold that optimizes fscore

    Returns: 
        float: false positive rate
    '''

    tp, fp, fn, tn = get_confusion_matrix(preds, labels, thresh)
    return(float(fp) / (fp + tn))


def pairwise_metrics(labeled_preds, threshold, phat_col, outcome, fscore_beta):
    '''Generate evaluation metrics that are standard for classification problems (e.g. precision, auc). 

    Args: 
        labeled_preds (pd.DataFrame): df with labeled test phats and columns needed for evaluation (limited to a universe)
        threshold (float): probability threshold that optimizes fscore
        phat_col: phat column for evaluation
        outcome: outcome to evaluate
        fscore_beta: ratio of recall weighting to precision weighting (e.g. 0.5 weights precision double)
    
    Return: 
        list of floats: base_rate, precision, recall, fpr, fnr, auc, accuracy, fscore
    '''

    labeled_preds = labeled_preds.copy()

    preds = labeled_preds[phat_col]
    labels = labeled_preds[outcome]
    base_rate = labels.mean()
    precision = get_precision(preds, labels, threshold)
    recall = get_recall(preds, labels, threshold)
    fpr = get_fpr(preds, labels, threshold)
    fnr = 1 - recall
    fscore = get_fscore(preds, labels, threshold, fscore_beta)
    accuracy = get_accuracy(preds, labels, threshold)

    fpr_auc, tpr_auc, thresholds_auc = metrics.roc_curve(labeled_preds[outcome], labeled_preds[phat_col])
    auc = metrics.auc(fpr_auc, tpr_auc)

    return base_rate, precision, recall, fpr, fnr, auc, accuracy, fscore


@log_runtime_and_memory
def find_best_threshold(df, beta):
    '''Find the threshold that optimizes fscore.

    Args:
        df (pd.DataFrame): predicted and actual values
            =====================   =======================================================
            label                   whether the pair is a match or not
            <phat_col>              predicted probability of match
            =====================   =======================================================
        beta (float): ratio of recall weighting to precision weighting (e.g. 0.5 weights precision double)

    Returns:
        float: probability threshold that optimizes fscore
    '''

    df = df.copy()

    logger.info('Finding optimal threshold.')

    best_t = -1
    best_fscore = -1
    for t in range(99, 1, -1):
        try:
            fscore = get_fscore(df.pred, df.outcome, t/100.0, beta)
        except:
            continue
        if fscore > best_fscore:
            best_fscore = fscore
            best_t = t / 100.0

    if best_t == -1:
        logger.error('Problem calculating optimal threshold.')
        raise ValueError

    return best_t


@log_runtime_and_memory
def evaluate_predictions(phat_df, phat_col, outcome,
        default_threshold=0.5, missingness_model_threshold_boost=0.2, optimize_threshold=False, fscore_beta=1.0, logger_id=None):
    '''Calculates metrics such as precision, recall, fscore, etc. for the pairwise record
    match predictions. Also, get the threshold that maximizes f score.

    Args:
        phat_df (pd.DataFrame): df with labeled test phats and columns needed for evaluation
        phat_col (str): phat column for evaluation
        outcome (str): outcome to evaluate
        default_threshold (float): threshold for match/non-match (use if don't find optimal)
        missingness_model_threshold_boost (float): value to add to default threshold if missingess model
        optimize_threshold (bool): should we find the threshold that optimizes f1
        fscore_beta (float): ratio of recall weighting to precision weighting (e.g. 0.5 weights precision double)

    Returns:
        float: probability threshold that optimizes fscore
        dict: keys are universe str, values are dicts with perf metrics
    '''

    if logger_id:
        logger = logging.getLogger(f"namematch_{logger_id}")
    else:
        logger = logging.getLogger()

    phat_df = phat_df.copy()

    # hacky for now
    if 'no_' in phat_col and 'basic' not in phat_col:
        default_threshold = default_threshold + missingness_model_threshold_boost

    if outcome != 'match_train_eligible':
        phat_df = phat_df[phat_df.match_train_eligible == 1]

    if len(phat_df) == 0:
        logger.info(f"Unable to measure performance (no labeled testing data available).")
        return default_threshold

    logger.info(f'Number of labeled test rows evaluated: {len(phat_df)}')

    if optimize_threshold:
        try:
            df = phat_df[[phat_col, outcome]].copy()
            df.columns = ['pred', 'outcome']
            threshold = find_best_threshold(df, fscore_beta)
        except:
            threshold = default_threshold
            logger.warning(f'Could not find optimal threshold, using {threshold}.')
    else:
        threshold = default_threshold
    logger.info(f"Threshold: {threshold}")

    all_model_stats = {}

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
            logger.debug(f'Phat distribution of actual 1s ({universe}): \n{one_phat_dist.to_string()}')
            model_stats['phat_distribution_1s'] = list(one_phat_dist)

            zero_phats = phat_univ_df[phat_univ_df[outcome] == 0][phat_col]
            zero_phat_dist = pd.value_counts(pd.cut(zero_phats, np.arange(0, 1.1, .1)), normalize=True, sort=False)
            logger.debug(f'Phat distribution of actual 0s ({universe}): \n{zero_phat_dist.to_string()}')
            model_stats['phat_distribution_0s'] = list(zero_phat_dist)

        except:
            logger.info(f"Issue calculating phat distributions ({universe}).")

        try:
            baserate, precision, recall, fpr, fnr, auc, accuracy, fscore = pairwise_metrics(
                phat_univ_df, threshold, phat_col, outcome, fscore_beta)

            logger.info(f"Base rate ({universe}): {baserate}")
            model_stats['baserate'] = np.float(baserate)

            logger.info(f"Precision ({universe}): {precision}")
            model_stats['precision'] = np.float(precision)

            logger.info(f"Recall ({universe}): {recall}")
            model_stats['recall'] = np.float(recall)

            logger.info(f"False positive rate ({universe}): {fpr}")
            model_stats['fp_rate'] = np.float(fpr)

            logger.info(f"False negative rate ({universe}): {fnr}")
            model_stats['fn_rate'] = np.float(fnr)

            logger.info(f"AUC ({universe}): {auc}")
            model_stats['auc'] = np.float(auc)

            logger.info(f"F-score ({universe}): {fscore}")
            model_stats['fscore'] = np.float(fscore)

        except:
            logger.warning(f"Issue with given threshold -- not all areas of confusion "
                         f"matrix present ({universe}).")

        if universe == "all pairs":

            try:
                up_recall = get_recall(up_phat_df[phat_col], up_phat_df[outcome], threshold)
                logger.info(f"Uncovered pair recall ({universe}): {up_recall}")
                model_stats['up_recall'] = up_recall
            except:
                logger.warning(f"Issue generating uncovered pair recall ({universe}).")

            try:
                # manually set preds to 0 if uncovered
                phat_univ_df.loc[phat_univ_df.covered_pair == 0, phat_col] = 0
                total_recall = get_recall(phat_univ_df[phat_col], phat_univ_df[outcome], threshold)
                logger.info(f"Total recall ({universe}): {total_recall}")
                model_stats['total_recall'] = np.float(total_recall)
            except:
                logger.warning("Issue computing total recall (%s).")

        all_model_stats[universe] = model_stats

    return threshold, all_model_stats


def evaluate_models(phats_df, outcome, model_type, default_threshold=0.5, missingness_model_threshold_boost=0.2,
            optimize_threshold=False, fscore_beta=1.0, logger_id=None):
    '''Wrapper for evaluating different models (e.g. basic and no-dob) on different universes. 

    Args:
        phat_df (pd.DataFrame): df with labeled test phats and columns needed for evaluation
        outcome (str): outcome to evaluate
        model_type (str): either "selection" or "match"
        default_threshold (float): threshold for match/non-match (use if don't find optimal)
        missingness_model_threshold_boost (float): value to add to default threshold if missingess model (use if don't find optimal)
        optimize_threshold (bool): should we find the threshold that optimizes fscore
        fscore_beta (float): ratio of recall weighting to precision weighting (e.g. 0.5 weights precision double)

    Return: 
        dict: maps model name (e.g. basic, no-dob) to thresholds (no return type)
    '''

    if logger_id:
        logger = logging.getLogger(f"namematch_{logger_id}")
    else:
        logger = logging.getLogger()

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
                phats_to_eval_df, phat_col, outcome,
                default_threshold, missingness_model_threshold_boost, optimize_threshold, fscore_beta, logger_id)

    logger.stat_dict({f"model_stats__{model_type}": model_type_model_stats})
    if model_type == 'match':
        logger.stat_dict({f"model_thresholds__{model_type}": thresholds})

    if model_type == 'match':
        return thresholds
