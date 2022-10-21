import editdistance
import fuzzy
import jellyfish._jellyfish as jf
import logging
import numpy as np
import pandas as pd

from datetime import datetime
from math import sqrt
from pyjarowinkler import distance

logger = logging.getLogger()

try:
    profile
except:
    from line_profiler import LineProfiler
    profile = LineProfiler()


@profile
def get_name_probabilities(df, np_object, first_name_col, last_name_col):
    '''Use the name-probabilities package to assesess the likelihood of a pair of names
    being the same person.

    Args:
        df (pd.DataFrame): side-by-side table
        np_object (NameMatcher object): contains info about name probabilities
        first_name_col (str): name of first-name column
        last_name_col (str): name of last-name column

    Returns:
        pd.DataFrame: side-by-side table with a few new name-probability columns
    '''

    if np_object is None:
        return df

    df = df.copy()

    df['fn1'] = df[first_name_col + '_1'].str.replace(' ', '')
    df['fn2'] = df[first_name_col + '_2'].str.replace(' ', '')
    df['ln1'] = df[last_name_col + '_1'].str.replace(' ', '')
    df['ln2'] = df[last_name_col + '_2'].str.replace(' ', '')
    df['fn_ln_1'] = '*' + df.fn1 + ' ' + df.ln1 + '*'
    df['fn_ln_2'] = '*' + df.fn2 + ' ' + df.ln2 + '*'
    df['rev_fn_ln_1'] = '*' + df.ln1 + ' ' + df.fn1 + '*'
    df['rev_fn_ln_2'] = '*' + df.ln2 + ' ' + df.fn2 + '*'

    df['prob_name_1'] = df.fn_ln_1.apply(lambda x: np_object.probName(x))
    df['prob_name_2'] = df.fn_ln_2.apply(lambda x: np_object.probName(x))

    df['prob_rev_name_1'] = df.rev_fn_ln_1.apply(lambda x: np_object.probName(x))
    df['prob_rev_name_2'] = df.rev_fn_ln_2.apply(lambda x: np_object.probName(x))

    df['count_pctl_name_1'] = df.fn_ln_1.map(np_object.n_name_appearances_dict)
    df['count_pctl_name_2'] = df.fn_ln_2.map(np_object.n_name_appearances_dict)

    df['count_pctl_fn_1'] = df.fn1.map(np_object.n_firstname_appearances_dict)
    df['count_pctl_fn_2'] = df.fn2.map(np_object.n_firstname_appearances_dict)

    df['count_pctl_ln_1'] = df.ln1.map(np_object.n_lastname_appearances_dict)
    df['count_pctl_ln_2'] = df.ln2.map(np_object.n_lastname_appearances_dict)

    df['prob_same_name'] = np.vectorize(np_object.probSamePerson, otypes=['float'])(
            df.fn_ln_1.values, df.fn_ln_2.values)
    df['prob_same_name_rev_1'] = np.vectorize(np_object.probSamePerson, otypes=['float'])(
            df.rev_fn_ln_1.values, df.fn_ln_2.values)
    df['prob_same_name_rev_2'] = np.vectorize(np_object.probSamePerson, otypes=['float'])(
            df.fn_ln_1.values, df.rev_fn_ln_2.values)

    df.drop(columns=['fn1', 'fn2', 'ln1', 'ln2', 'fn_ln_1',
            'fn_ln_2', 'rev_fn_ln_1', 'rev_fn_ln_2'], inplace=True)

    return df


@profile
def try_switch_first_last_name(df, first_name_col, last_name_col):
    '''The input df to this function is the output of get_name_probabilities. Using the probabilities
    calculated above, it decides whether the names labelled as FIRST and LAST were swapped. If they
    are, it fixes the swap.

    Args:
        df (pd.DataFrame): side-by-side table with a few new name-probability columns
        first_name_col (str): name of first-name column
        last_name_col (str): name of last-name column

    Returns:
        pd.DataFrame: side-by-side table with a few new name-probability columns (names swapped as needed)
    '''


    df = df.copy()

    df['full_name_1'] = df[first_name_col + '_1'] + df[last_name_col + '_1']
    df['full_name_2'] = df[first_name_col + '_2'] + df[last_name_col + '_2']
    df['reversed_full_name_1'] = df[last_name_col + '_1'] + df[first_name_col + '_1']

    # compare name_1 to name_2, and the reversed name_1 to name_2, to see which are more similar
    df['name_ed'] = np.vectorize(editdistance.eval, otypes=['float'])(
            df.full_name_1.values, df.full_name_2.values)
    df['reversed_name_ed'] = np.vectorize(editdistance.eval, otypes=['float'])(
            df.reversed_full_name_1.values, df.full_name_2.values)

    df['least_likely_name'] = (df.prob_name_2 < df.prob_name_1).astype(int) + 1

    # if the reversed name has a lower edit distance, mark that it is likely switched
    df['switched_name'] = 0
    df.loc[df.reversed_name_ed < df.name_ed, 'switched_name'] = df.least_likely_name

    # we need temp columns to orchestrate the switch
    df['fn1'] = df[first_name_col + '_1']
    df['fn2'] = df[first_name_col + '_2']
    df['ln1'] = df[last_name_col + '_1']
    df['ln2'] = df[last_name_col + '_2']

    # for names marked as switched, fix the switch
    df.loc[df.switched_name == 1, 'fn1'] = df[last_name_col + '_1']
    df.loc[df.switched_name == 1, 'ln1'] = df[first_name_col + '_1']

    df.loc[df.switched_name == 2, 'fn2'] = df[last_name_col + '_2']
    df.loc[df.switched_name == 2, 'ln2'] = df[first_name_col + '_2']

    # clean up all the columns
    df.drop(
        columns=[
            first_name_col + '_1',
            first_name_col + '_2',
            last_name_col + '_1',
            last_name_col + '_2',
            'full_name_1',
            'full_name_2',
            'reversed_full_name_1',
            'name_ed',
            'reversed_name_ed',
            'least_likely_name'],
        inplace=True)
    df.rename(
        columns={
            'fn1': first_name_col + '_1',
            'fn2': first_name_col + '_2',
            'ln1': last_name_col + '_1',
            'ln2': last_name_col + '_2'},
        inplace=True)

    return df


@profile
def compare_strings(df, varname):
    '''Compute string distances between two records for a particular matching field.

    Args:
        df (pd.DataFrame): side-by-side table
        varname (str): name of the all-names column to compute distance measures for

    Returns:
        pd.DataFrame: features associated with the current variable
            ===========================   ============================================================
            <var>_missing                 flag, 1 if either record is missing the value for this field
            <var>_edit_dist               number of character edits between whole strings
            <var>_exact_match             flag, 1 if whole string matches
            <var>_exact_match_1st         flag, 1 if first charcater matches
            <var>_exact_match_1st2nd3rd   flag, 1 if first three characters match
            <var>__soundex                flag, 1 if the soundex codes match
            <var>__nysiis                 flag, 1 if the nysiis codes match
            <var>__jw_dist                jarowinkler distance (float)
            ===========================   ============================================================
    '''

    df = df.copy()
    features_df = pd.DataFrame(index=df.index)
    col1 = varname + '_1'
    col2 = varname + '_2'

    df['msng'] = ((df[col1] == '') | (df[col2] == '')).astype(int)
    df['valid_nysiis'] = ((df.msng == 0) & (df[col1].str.contains(r'\d') == False) & (df[col2].str.contains(r'\d') == False)).astype(int)
    features_df[varname + '_missing'] = df.msng

    features_df.loc[df.msng == 0, varname + '_edit_dist'] = \
            np.vectorize(editdistance.eval, otypes=['float'])(
                df[df.msng == 0][col1].values,
                df[df.msng == 0][col2].values)

    features_df.loc[df.msng == 0, varname + '_exact_match'] = \
            (df[col1] == df[col2]).astype(float)

    features_df.loc[df.msng == 0, varname + '_exact_match_1st'] = \
            (df[col1].str.slice(0, 1) == df[col2].str.slice(0, 1)).astype(float)

    features_df.loc[df.msng == 0, varname + '_exact_match_1st2nd3rd'] = \
            (df[col1].str.slice(0, 3) == df[col2].str.slice(0, 3)).astype(float)

    df.loc[df.msng == 0, 'soundex_col_1'] = np.vectorize(jf.soundex, otypes=['str'])(
            df[df.msng == 0][col1])
    df.loc[df.msng == 0, 'soundex_col_2'] = np.vectorize(jf.soundex, otypes=['str'])(
            df[df.msng == 0][col2])
    features_df[varname + '_soundex'] = (df.soundex_col_1 == df.soundex_col_2).astype(int)

    df.loc[df.valid_nysiis == 1, 'nysiis_col_1'] = np.vectorize(fuzzy.nysiis, otypes=['str'])(
            df[df.valid_nysiis == 1][col1].values)
    df.loc[df.valid_nysiis == 1, 'nysiis_col_2'] = np.vectorize(fuzzy.nysiis, otypes=['str'])(
            df[df.valid_nysiis == 1][col2].values)
    features_df[varname + '_nysiis'] = (df.nysiis_col_1 == df.nysiis_col_2).astype(int)

    features_df.loc[df.msng == 0, varname + '_jw_dist'] = \
            1 - np.vectorize(distance.get_jaro_distance, otypes=['float'])(
                df[df.msng == 0][col1].values,
                df[df.msng == 0][col2].values)

    return features_df


def compare_numbers(df, varname):
    '''Compute numeric distances between two records for a particular matching field.

    Args:
        df (pd.DataFrame): side-by-side table
        varname (str): name of the all-names column to compute distance measures for

    Returns:
        pd.DataFrame: features associated with the current variable
            ====================    =======================================================
            <var>_missing           flag, 1 if either record is missing the value for this field
            <var>_num_diff          numeric difference between two field valies
            ====================    =======================================================
    '''

    df = df.copy()
    features_df = pd.DataFrame(index=df.index)
    col1 = varname + '_1'
    col2 = varname + '_2'

    features_df[varname + '_missing'] = ((df[col1] == '') | (df[col2] == '')).astype(int)

    features_df.loc[features_df[varname + '_missing'] == 0, varname + '_num_diff'] = \
            (pd.to_numeric(df[col1]) - pd.to_numeric(df[col2])).abs().astype(float)

    return features_df


@profile
def compare_categories(df, varname):
    '''Compute categorical distances between two records for a particular matching field.

    Args:
        df (pd.DataFrame): side-by-side table
        varname (str): name of the all-names column to compute distance measures for

    Returns:
        pd.DataFrame: features associated with the current variable
            ====================    =======================================================
            <var>_missing           flag, 1 if either record is missing the value for this field
            <var>_exact_match       flag, 1 if categories are identical
            <var>_partial_match     flag, 1 if any part of the categories (split by " ") match
            ====================    =======================================================
    '''

    df = df.copy()
    features_df = pd.DataFrame(index=df.index)
    col1 = varname + '_1'
    col2 = varname + '_2'

    features_df[varname + '_missing'] = ((df[col1] == '') | (df[col2] == '')).astype(int)

    features_df.loc[features_df[varname + '_missing'] == 0, varname + '_exact_match'] = \
            (df[col1] == df[col2]).astype(float)

    # features_df.loc[features_df[varname + '_missing'] == 0, varname + '_partial_match'] = \
    #         df[[col1, col2]].apply(lambda row: \
    #         int(any(col1_piece in row[col2].split(' ') for col1_piece in row[col1].split(' '))), axis=1)

    return features_df


@profile
def compare_dates(df, varname):
    '''Compute date distances between two records for a particular matching field.
    NOTE: The date MUST be in the format yyyy-mm-dd (handled upstream).

    Args:
        df (pd.DataFrame): side-by-side table
        varname (str): name of the all-names column to compute distance measures for

    Returns:
        pd.DataFrame: features associated with the current variable
            ====================    =======================================================
            <var>_missing           flag, 1 if either record is missing the value for this field
            <var>_edit_dist         number of character edits between two date values
            <var>_day_diff          number of days between two date values
            ====================    =======================================================
    '''

    df = df.copy()
    features_df = pd.DataFrame(index=df.index)
    col1 = varname + '_1'
    col2 = varname + '_2'

    df['date1'] = pd.to_datetime(df[col1], format='%Y-%m-%d', errors='coerce')
    df['date2'] = pd.to_datetime(df[col2], format='%Y-%m-%d', errors='coerce')

    features_df[varname + '_missing'] = (df.date1.isnull() | df.date2.isnull()).astype(int)

    features_df.loc[features_df[varname + '_missing'] == 0, varname + '_edit_dist'] = \
            np.vectorize(editdistance.eval, otypes=['float'])(
                df[features_df[varname + '_missing'] == 0][col1].values,
                df[features_df[varname + '_missing'] == 0][col2].values).astype(float)

    features_df.loc[features_df[varname + '_missing'] == 0, varname + '_day_diff'] = \
            ((df.date1 - df.date2) / np.timedelta64(1, 'D')).abs().astype(float)

    return features_df


def compare_geographies(df, varname):
    '''Compute geographic distance between two records for a particular matching field.

    Args:
        df (pd.DataFrame): side-by-side table
        varname (str): name of the all-names column to compute distance measures for

    Returns:
        pd.DataFrame: features associated with the current variable
            ====================    =======================================================
            <var>_missing           flag, 1 if either record is missing the value for this field
            <var>_geog_dist         euclidian distance between two geographic points
            ====================    =======================================================
    '''

    df = df.copy()
    features_df = pd.DataFrame(index=df.index)
    col1 = varname + '_1'
    col2 = varname + '_2'

    df[['x1', 'y1']] = df[col1].str.split(',', n=1, expand=True)
    df[['x2', 'y2']] = df[col2].str.split(',', n=1, expand=True)
    df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].replace('', np.NaN)
    df['x1_minus_x2_squared'] = np.square(df.x1.astype(float) - df.x2.astype(float))
    df['y1_minus_y2_squared'] = np.square(df.y1.astype(float) - df.y2.astype(float))

    features_df[varname + '_missing'] = ((df[col1] == '') | (df[col2] == '')).astype(int)

    features_df.loc[features_df[varname + '_missing'] == 0, varname + '_geog_dist'] = \
            np.sqrt(df.x1_minus_x2_squared + df.y1_minus_y2_squared).astype(float)

    return features_df


@profile
def generate_label(df, uid_vars, leven_thresh):
    '''Create a column indicating whether the UIDs of a pair of people is a match. The value is 1
    if UIDs are non-missing match, 0 if UIDs are non-missing and don't match, and missing if either
    UID is missing or if (0 < edit distance <= `leven_thresh`).

    The `leven_thresh` will never flip a mismatch to a match, only to a missing.

    Args:
        df (pd.DataFrame): side-by-side table
        uid_vars (list): column name(s) of the UniqueID variable(s)
        leven_thresh (int): n character edits to allow between uids, below which 0 labels will get flipped to ""

    Returns:
        pd.Series: label column
    '''

    df = df.copy()

    label_df = pd.DataFrame(index=df.index)

    for uid in uid_vars:

        col1 = uid + '_1'
        col2 = uid + '_2'

        label_df[uid + '_label'] = ''
        label_df.loc[(df[col1] != '') & (df[col2] != ''), uid + '_label'] = (df[col1] == df[col2]).astype(int)

        # if leven_thresh is in use, set missing all labels that are too close
        if leven_thresh is not None:
            label_df[uid + '_ed'] =  np.vectorize(editdistance.eval, otypes=['float'])(
                    df[col1].values, df[col2].values)
            label_df.loc[(label_df[uid + '_ed'] <= leven_thresh) &
                    (label_df[uid + '_ed'] > 0), uid + '_label'] = ''

    # num_0s is the number of times the IDs actually don't match-- a near-match would be stored as a ""
    label_df['num_1s'] = (label_df.filter(regex='_label') == 1).sum(axis=1)
    label_df['num_0s'] = (label_df.filter(regex='_label') == 0).sum(axis=1)

    label_df['label'] = ''

    label_df.loc[(label_df.num_0s == 0) & (label_df.num_1s > 0), 'label'] = '1'
    label_df.loc[(label_df.num_0s > 0) & (label_df.num_1s == 0), 'label'] = '0'
    # NOTE: if there are multiple unique ids, only use pairs that always agree as 1s;
    #       only use pairs that always disagree as 0s

    return label_df.label
