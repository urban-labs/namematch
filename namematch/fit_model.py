import logging
import numpy as np
import os
import pandas as pd
import pickle
import random

import pyarrow.parquet as pq
from descriptors import cachedproperty

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.feature_selection import VarianceThreshold

from namematch.data_structures.parameters import Parameters
from namematch.model_evaluation_functions import evaluate_models, get_cv_metrics
from namematch.predict import Predict
from namematch.base import NamematchBase
from namematch.utils.utils import (
    load_parquet_list,
    dump_yaml,
    log_runtime_and_memory,
    determine_model_to_use,
    load_models,
)
from namematch.utils.profiler import Profiler

profile = Profiler()
logger = logging.getLogger()


# globals:
MATCH_COL = 1
SELECTION_OUTCOME = 'match_train_eligible'
MATCH_OUTCOME = 'label'


class FitModel(NamematchBase):
    '''
    Args:
        params (Parameters object): contains parameter values
        all_names_file (str): path to the all-names file
        data_rows_dir (str): path to the data-rows dir
        model_info_file (str): path to the model info yaml file
        output_dir (str): path to the model dir
        traiend_model_info_file (str): path to the model info yaml file of a previously trained model
        selection_model_name (str): selection model name
        match_model_name (str): match model name
        flipped0_file (str): flipped0 file path
    '''
    def __init__(
        self,
        params,
        all_names_file,
        data_rows_dir,
        model_info_file,
        output_dir,
        trained_model_info_file='None',
        selection_model_name='basic_selection_model.pkl',
        match_model_name='basic_match_model.pkl',
        flipped0_file=None,
        *args,
        **kwargs
    ):

        super(FitModel, self).__init__(params, None, *args, **kwargs)

        self.all_names_file = all_names_file
        self.data_rows_dir = data_rows_dir
        self.model_info_file = model_info_file
        self.trained_model_info_file = trained_model_info_file
        self.selection_model_name = selection_model_name
        self.match_model_name = match_model_name
        self.output_dir = output_dir
        self.flipped0_file = flipped0_file if flipped0_file else os.path.join(os.path.dirname(self.output_dir), 'flipped0_potential_edges.csv')
        self.model_info = None
        self.match_models = None
        self.selection_models = None

    @property
    def output_files(self):
        output_files = [self.flipped0_file]
        if self.trained_model_info_file == 'None':
            output_files += [self.model_info_file, os.path.join(self.output_dir, self.match_model_name)]

        if self.params.weight_using_selection_model:
            output_files.append(os.path.join(self.output_dir, self.selection_model_name))

        if self.params.missingness_model is not None:
            if self.params.weight_using_selection_model:
                output_files.append(os.path.join(self.output_dir, f'no_{self.params.missingness_model}_selection_model.pkl'))
            output_files.append(os.path.join(self.output_dir, f'no_{self.params.missingness_model}_match_model.pkl'))

        return output_files

    @property
    def dr_file_list(self):
        return [
            os.path.join(self.data_rows_dir, dr_file) for dr_file in os.listdir(self.data_rows_dir)
        ]

    @log_runtime_and_memory
    @profile
    def main(self, **kw):
        '''Train and evaluate random foreset model(s). Depending on the settings, this might involved
        training and evaluating multiple types of models (e.g. selection and match models) and/or models
        for different data-row types (e.g. basic and no-dob).
        '''

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        table = pq.read_table(self.all_names_file)
        an = table.to_pandas()
        an = an[an.drop_from_nm == 0]
        an = an.drop_duplicates(['record_id'])

        # add "is valid for training" flag
        an['valid_for_match_train'] = self.find_valid_training_records(an,
                self.params.match_train_criteria.get('all_names', {}))
        an_train_eligible_dict = an.set_index('record_id')['valid_for_match_train'].to_dict()

        # remove flipped0 file if it exists (since this only sometimes gets created,
        # don't want it lingering for future runs -- messes up potential_edges)
        if os.path.exists(self.flipped0_file):
            os.remove(self.flipped0_file)

        # if a trained model is provided, load it
        if self.trained_model_info_file != "None":
            self.match_models, self.model_info = load_and_save_trained_model(
                self.trained_model_info_file,
                self.model_info_file
            )
            match_train_df, match_eval_df, prob_match_train = self.get_train_eval_data(
                None,
                self.model_info,
                self.params,
                'match',
                any_train=False
            )

            match_eval_phats = Predict.predict(self.match_models, match_eval_df, 'match')
            flipped0 = get_flipped0_potential_edges(
                match_eval_phats,
                self.model_info,
                self.params.allow_clusters_w_multiple_unique_ids
            )
            flipped0.to_csv(self.flipped0_file, index=False)

            return

        # if a trained model is not provided, need to build it
        else:
            self.model_info = define_necessary_models(
                self.dr_file_list,
                self.output_dir,
                self.params.missingness_model,
                selection_model_name=self.selection_model_name,
                match_model_name=self.match_model_name,
            )
            if self.params.weight_using_selection_model:
                selection_train_df, selection_eval_df, prob_match_train = self.get_train_eval_data(
                    None, self.model_info, self.params, 'selection')

                self.selection_models = self.fit_models(
                        selection_train_df, 'selection', self.model_info)

                selection_eval_phats = Predict.predict(self.selection_models, selection_eval_df,
                        'selection', prob_match_train=prob_match_train)
                self.evaluate_models(selection_eval_phats, SELECTION_OUTCOME, 'selection')

                selection_train_df.to_parquet(os.path.join(self.output_dir, 'selection_train_df.parquet')) # TEMP
                selection_eval_df.to_parquet(os.path.join(self.output_dir, 'selection_eval_df.parquet')) # TEMP

            match_train_df, match_eval_df, prob_match_train = self.get_train_eval_data(
                an_train_eligible_dict,
                self.model_info,
                self.params,
                'match'
            )

            if self.params.weight_using_selection_model:
                # needed for weighting
                # NOTE: some of the weights will be in-sample
                match_train_df = Predict.predict(self.selection_models, match_train_df,
                        'selection', all_cols=True, all_models=True, prob_match_train=prob_match_train)
                match_eval_df = Predict.predict(self.selection_models, match_eval_df,
                        'selection', all_cols=True, all_models=True, prob_match_train=prob_match_train)

            self.match_models = self.fit_models(
                    match_train_df, 'match', self.model_info)

            match_train_phats = Predict.predict(self.match_models, match_train_df, 'match', oob=True)
            match_eval_phats = Predict.predict(self.match_models, match_eval_df, 'match')
            match_train_phats.to_parquet(os.path.join(self.output_dir, 'train_phats.parquet')) # TEMP
            match_eval_phats.to_parquet(os.path.join(self.output_dir, 'eval_phats.parquet')) # TEMP

            match_train_df.filter(regex='record_id|weight').to_parquet(os.path.join(self.output_dir, 'match_train_weights.parquet')) # TEMP
            match_eval_df.filter(regex='record_id|weight').to_parquet(os.path.join(self.output_dir, 'match_eval_weights.parquet')) # TEMP

            thresholds_dict = self.evaluate_models(
                match_eval_phats,
                MATCH_OUTCOME, 'match',
                self.params.weight_using_selection_model,
                self.params.default_threshold,
                self.params.missingness_model_threshold_boost,
                self.params.optimize_threshold,
                self.params.fscore_beta,
            )

            self.model_info = add_threshold_dict(self.model_info, thresholds_dict)

            flipped0 = get_flipped0_potential_edges(
                    pd.concat([match_train_phats, match_eval_phats]),
                    self.model_info,
                    self.params.allow_clusters_w_multiple_unique_ids)

            # save models and info
            dump_yaml(self.model_info, self.model_info_file)
            save_models(self.selection_models, self.match_models, self.model_info)

            flipped0.to_csv(self.flipped0_file, index=False)

            if self.enable_lprof:
                self.write_line_profile_stats(profile.line_profiler)


    @log_runtime_and_memory
    def fit_model(self, df, vars_to_exclude, outcome, weights=None, n_jobs=1, **kw):
        '''Fit random forest model.

        Args:
            df (pd.DataFrame): data rows, subset to training rows
            vars_to_exclude (list): variables to disallow from the model
            outcome (string): name of the column that we're predicting
            weights (list): sample weights to use for training (can be None)
            n_jobs (int): number of jobs to run in parallel

        Returns:
                tuple: tuple containing:

                - **mod** (*sklearn.ensemble.RandomForestClassifier*): trained sklearn random forest model object
                - **feature_info** (*pd.DataFrame*): feature_importance
        '''

        if df[outcome].nunique() == 1:
            logger.error('Something went wrong; all training outcomes are {}'.format(df[outcome].tolist()[0]))
            raise ValueError

        col_names = df.columns.values.tolist()
        col_names = [col for col in col_names if col not in vars_to_exclude]
        numeric_cols = [c for c in col_names if 'var_' in c and '_missing' not in c and 'match' not in c]
        categorical_cols = [c for c in col_names if 'var_' in c  and ('_missing' in c or 'match' in c)]

        numeric_transformer = FeatureUnion([
            ('impute_numeric', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean'))])),
            ('missing_dummies', Pipeline(steps=[
                ('missing', MissingIndicator(error_on_new=False))]))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)],
            sparse_threshold=0)

        base_clf = RandomForestClassifier(
                n_estimators=200,
                oob_score=True,
                n_jobs=n_jobs,
                verbose=0)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('no_variance_filter', VarianceThreshold()),
            ('clf', base_clf)
        ])

        pipeline_params = {
            'clf__min_samples_leaf':[25, 150],
        }

        mod = GridSearchCV(pipeline, pipeline_params,
			cv=3,
			n_jobs=1, verbose=0,
			scoring='f1',
            return_train_score=True
        )

        mod.fit(df, df[outcome],
                clf__sample_weight=weights)

        feature_info = get_feature_info(mod.best_estimator_, numeric_cols, categorical_cols)

        return mod, feature_info

    def fit_models(self, train_df, model_type, model_info):
        '''Fit random forest model.

        Args:
            train_df (pd.DataFrame): data rows, subset to training rows
            model_type (string): either "selection" or "match"
            model_info (dict): dict with information about how to fit the model

        Returns:
            dict: maps model name (e.g. basic or no_dob) to a trained model object
        '''

        models = {}
        feature_infos = {}
        cv_metrics = {}
        for this_model_name, this_model_info in model_info.items():

            outcome = MATCH_OUTCOME
            if model_type == 'selection':
                outcome = SELECTION_OUTCOME

            weights = None
            if model_type == 'match':
                if f'{this_model_name}_weight' in train_df.columns:
                    weights = train_df[f'{this_model_name}_weight']
                    logger.info(f'Using {this_model_name} weights.')

            logger.info(f'Training {this_model_name} model ({model_type}).')

            models[this_model_name], feature_infos[this_model_name] = self.fit_model(
                train_df,
                this_model_info['vars_to_exclude'],
                outcome,
                weights=weights,
                n_jobs=self.params.num_workers)

        logger.stat_dict({f"feature_info__{model_type}": feature_infos})
        self.stats_dict[f'feature_info__{model_type}'] = feature_infos

        return models

    def evaluate_models(self,
            phats_df,
            outcome,
            model_type,
            weight_using_selection_model=False,
            default_threshold=0.5,
            missingness_model_threshold_boost=0.2,
            optimize_threshold=False,
            fscore_beta=1.0,
            **kw
            ):
        return evaluate_models(
                    phats_df,
                    outcome,
                    model_type,
                    weight_using_selection_model,
                    default_threshold,
                    missingness_model_threshold_boost,
                    optimize_threshold,
                    fscore_beta,
                    self.stats_dict,
                    )

    def get_train_eval_data(self, an_train_eligible_dict, model_info, params, model_type, any_train=True):
        '''Load data-rows, filter to rows that are eligible for training a givem model type, and then split
        the data into a training set and a labeled evaulation set.

        Args:
            an_train_eligible_dict (dict): maps record_id to flag indicating record's all-names based training eligibility
            model_info (dict): dict with information about how to fit the model
            params (Parameters object): contains parameter values
            model_type (str): either "selection" or "match"
            any_train (bool): True if you want training data (e.g. not a pre-trained model), False otherwise

        Returns:
            pd.DataFrame: data rows, filtered to training data (excluding labeled eval data)
            pd.DataFrame: data rows, filtered to labeled eval data
            float: share of data rows that are labeled
        '''

        temp_dr = load_parquet_list(
            self.dr_file_list,
            cols=['dr_id', 'covered_pair', 'labeled_data'])
        n_data_rows = len(temp_dr)
        logger.info(f"Number of data rows: {n_data_rows}")
        self.stats_dict['n_data_rows'] = n_data_rows
        n_covered_data_rows = len(temp_dr[temp_dr.covered_pair == 1])
        self.stats_dict['n_covered_data_rows'] = n_covered_data_rows
        del temp_dr

        if model_type == 'selection':
            sample = params.max_selection_train_eval_n / n_data_rows
            if sample > 1:
                sample = 1
            conditions_dict = {'covered_pair': 1}
            outcome = SELECTION_OUTCOME
        elif model_type == 'match':
            sample = 1
            conditions_dict = {'labeled_data':1, 'covered_pair': 1}
            outcome = MATCH_OUTCOME
        else:
            logger.error("Invalid model_type supplied: must be either 'selection' or 'match.'")
            raise

        train_eval_df = load_parquet_list(
                self.dr_file_list,
                conditions_dict=conditions_dict,
                sample=sample)

        # create match_train_eligible flag (1 if labeled and meet data-row and all-names critiera, 0 otherwise)
        dr_train_eligible_conditions_dict = params.match_train_criteria.get('data_rows', {})
        train_eval_df['match_train_eligible'] = \
            get_match_train_eligible_flag(train_eval_df, dr_train_eligible_conditions_dict, an_train_eligible_dict)

        if model_type == 'selection':
            prob_match_train = (train_eval_df.match_train_eligible.sum() / sample) / self.stats_dict['n_covered_data_rows']
            self.stats_dict['prob_match_train'] = float(prob_match_train)
            logger.info(f"P(s): {prob_match_train}")
        else:
            prob_match_train = self.stats_dict.get('prob_match_train', None)

        train_eval_df[outcome] = train_eval_df[outcome].astype(int)

        train_eval_df['model_to_use'] = \
                determine_model_to_use(train_eval_df, model_info, verbose=True)

        if not any_train:
            train_df = None
            eval_df = train_eval_df.copy()

        else: # need train/eval split

            pct_train = params.pct_train
            if (model_type == 'match') and (pct_train * len(train_eval_df) > params.max_match_train_n):
                pct_train = params.max_match_train_n / len(train_eval_df)

            train_eval_df['train'] = [random.random() <= pct_train
                                      for r in range(len(train_eval_df))]
            # TODO should we split at uid level? -- hard since these are pairs?

            if model_type == 'selection':
                train_df = train_eval_df[(train_eval_df.train == 1)].copy()
                eval_df = train_eval_df[(train_eval_df.train == 0)].copy()
            elif model_type == 'match':
                train_df = train_eval_df[
                        (train_eval_df.train == 1) &
                        (train_eval_df.match_train_eligible == 1)].copy()
                eval_df = train_eval_df[
                        (train_eval_df.train == 0) &
                        (train_eval_df.match_train_eligible == 1)].copy()

        return train_df, eval_df, prob_match_train


    @log_runtime_and_memory
    def find_valid_training_records(self, an, an_match_criteria):
        '''Identify records that meet the all-names criteria for training data.

        Args:
            an (pd.DataFrame): all-names table (one row per input record)

                =====================   =======================================================
                record_id               unique record identifier
                file_type               either "new" or "existing"
                <fields for matching>   both for the matching model and for constraint checking
                <raw name fields>       pre-cleaning version of first and last name
                blockstring             concatenated version of blocking columns (sep by ::)
                drop_from_nm            flag, 1 if met any "to drop" criteria 0 otherwise
                =====================   =======================================================

            an_match_criteria (dict): keys are all-names columns, mapped to acceptable values

        Returns:
            pd.Series: flag, 1 if the record is eligible for training set 0 otherwise
        '''

        an['meets_match_criteria'] = 1
        for col, accepted_values in an_match_criteria.items():
            if not isinstance(accepted_values, list):
                accepted_values = [accepted_values]
            accepted_values = [str(av) for av in accepted_values]
            an['meets_match_criteria'] = (an['meets_match_criteria']) & (an[col].isin(accepted_values))

        return an['meets_match_criteria']


def get_feature_info(pipeline, raw_num_cols, raw_cat_cols):
    '''Extract the feature importance information from a sklearn model pipeline.

    Args:
        pipeline (skleran fitted pipeline): trained model
        raw_num_cols (list): numeric columns that went into the model (before pipeline processing)
        raw_cat_cols (list): categorical columns that went into the model (before pipeline processing)

    Returns:
        pd.DataFrame: feature importance information

        ==============   =================================================
        feature          name of the feautre
        importance       relative importance of this feature to the model
        ==============   =================================================
    '''

    raw_num_cols = raw_num_cols[:]
    raw_cat_cols = raw_cat_cols[:]

    preprocessor = pipeline.named_steps['preprocessor']
    num_transf = preprocessor.named_transformers_['num']
    cat_transf = preprocessor.named_transformers_['cat']

    num_cols_imputed = [raw_num_cols[col_ix]
                        for col_ix in range(len(raw_num_cols))
                        if not np.isnan(num_transf.transformer_list[0][1].named_steps['imputer'].statistics_[col_ix])]

    num_cols_missing = [raw_num_cols[col_ix] + '_is_missing'
                        for col_ix in num_transf.transformer_list[1][1].named_steps['missing'].features_]

    cat_feature_names = ['%s_%s' % (raw_cat_cols[col_ix], cat)
                         for col_ix in range(len(raw_cat_cols))
                         for cat in cat_transf.named_steps['onehot'].categories_[col_ix]]

    feature_names_pre_varthresh = num_cols_imputed + num_cols_missing + cat_feature_names

    retained_flags = pipeline.named_steps['no_variance_filter'].get_support()
    feature_names = [feature_names_pre_varthresh[i]
                     for i in range(len(feature_names_pre_varthresh)) if retained_flags[i]]
    dropped_features = [feature_names_pre_varthresh[i]
                        for i in range(len(feature_names_pre_varthresh)) if not retained_flags[i]]

    importances = [float(imp) for imp in list(pipeline.named_steps['clf'].feature_importances_)]

    feature_info = {"retained": feature_names, "imp": importances, "dropped": dropped_features}

    return feature_info


def save_models(selection_models, match_models, model_info):
    '''Save the models to file.

    Args:
        selection_models (dict): maps model name (e.g. basic or no-dob) to a fit match model object
        match_models (dict): maps model name (e.g. basic or no-dob) to a fit selection model object
        model_info (dict): dict with information about how to fit the model
    '''

    for model_name, this_model_info in model_info.items():

        # save selection model
        if selection_models is not None:
            with open(this_model_info['selection_model_path'], 'wb') as mf:
                pickle.dump(selection_models[model_name], mf)

        # save match model
        with open(this_model_info['match_model_path'], 'wb') as mf:
            pickle.dump(match_models[model_name], mf)


def define_necessary_models(
        dr_file_list,
        output_dir,
        missing_mod_field=None,
        selection_model_name='basic_selection_model.pkl',
        match_model_name='basic_match_model.pkl',
        ):
    '''Determine the different models needed (using a sample) and
    define the characteristics of data that determine which
    model should handle it.

    NOTE: Right now, there is an assumption that the training universe
          is the same between all models (i.e. basic and missingness)

    Args:
        dr_file_list (list): list of paths to all data row files
        output_dir (str): model output folder path
        missing_mod_field (str or None): field that could trigger need for separate model

    Returns:
        dict: mapping the name of a model (str) to a dict of the following information:

            - selection_model_name (str)
            - match_model_name (str)
            - type (str): one of "default" or "missingness"
            - actual_phat_universe (dict): maps a variable name to a value(?)
            - vars_to_exclude (str list)
            - match_thresh (float): threshold for match/nonmatch
    '''

    # initialize categories (always need a basic model)
    categories = {
        'basic' : {
            'selection_model_path' :
                os.path.abspath(os.path.join(output_dir, selection_model_name)),
            'match_model_path' :
                os.path.abspath(os.path.join(output_dir, match_model_name)),
            'type' : 'default',
            'actual_phat_universe' : {},
            'vars_to_exclude' : [],
            'match_thresh' : None
        }
    }

    # determine if missingness model is needed
    if missing_mod_field is not None:
        # missingness model is needed; universe is same as "basic" model

        var_regex = f'var_{missing_mod_field}_'
        dr_first_row = pd.read_parquet(dr_file_list[0], engine='pyarrow').head(1)
        related_variables = dr_first_row.filter(regex=var_regex).columns.tolist()

        categories[f'no_{missing_mod_field}'] = {
            'selection_model_path' :
                os.path.abspath(os.path.join(output_dir, f'no_{missing_mod_field}_selection_model.pkl')),
            'match_model_path' :
                os.path.abspath(os.path.join(output_dir, f'no_{missing_mod_field}_match_model.pkl')),
            'type' : 'missingness',
            'actual_phat_universe' : {
                f'var_{missing_mod_field}_missing' : 1
            },
            'vars_to_exclude' : related_variables,
            'match_thresh' : None
        }

    # NOTE: will need more logic here once we have model types other than
    #       basic and missing (e.g. uncovered_pairs)

    return categories


def load_and_save_trained_model(trained_model_info_file, output_file):
    '''Load a set of pre-trained models and copy them to the current run's
    output directory. Typically only used in incremental runs.

    Args:
        trained_model_info_file (str): path to a model yaml file, which has path/threshold/universe info
        output_file (str): path to output the current run's model yaml file (for copying)

    Returns:
        dict: maps model name (e.g. basic or no_dob) to a trained model object
    '''

    try:
        selection_models, \
        match_models, \
        model_info = load_models(trained_model_info_file, selection=True)
        if len(match_models) == 0:
            raise
    except:
        logger.error("Unable to load trained model or its info yaml.")
        raise

    dump_yaml(model_info, output_file)

    # NOTE: not returning selection_models now, but may need to once using
    #       pre-trained model is fully functional (for unbiased evaluation)
    return match_models, model_info


def get_match_train_eligible_flag(df, dr_train_eligible_conditions_dict, an_train_eligible_dict):
    '''Determine if a data-row is eligible for training (for match models), according
    to both all-names eligibility criteria and data-row eligibility criteria.

    Args:
        df (pd.DataFrame): portion of data-rows file, limited to labeled rows
        dr_train_eligible_conditions_dict (dict): contains data-row training eligibility criteria
        an_train_eligible_dict (dict): maps record_id to flag indicating record's all-names based training eligibility

    Returns:
        pd.Series: flag, 1 if data-row is training eligible (for match models)
    '''

    df = df.copy()

    df['match_train_eligible'] = df.labeled_data

    if an_train_eligible_dict is not None:

        df.loc[df.match_train_eligible == 1, 'record_1_elig'] = \
                df[df.match_train_eligible == 1].record_id_1.map(an_train_eligible_dict)
        df.loc[df.match_train_eligible == 1, 'record_2_elig'] = \
                df[df.match_train_eligible == 1].record_id_2.map(an_train_eligible_dict)

        df['match_train_eligible'] = \
                ((df.match_train_eligible) &
                 (df.record_1_elig == 1) &
                 (df.record_2_elig == 1)).astype(int)

    for col, accepted_values in dr_train_eligible_conditions_dict.items():

        if not isinstance(accepted_values, list):
            accepted_values = [accepted_values]

        df['match_train_eligible'] = \
                (df['match_train_eligible'] &
                    df[col].isin(accepted_values))

    return df.match_train_eligible


def add_threshold_dict(model_info, thresholds_dict):
    '''Add threshold information to the model_info dict, once it's been determined.

    Args:
        model_info (dict): dict with information about how to fit the model
        thresholds_dict (dict): keys are model name (e.g. basic, no-dob), values are optimized thresholds

    Return:
        dict: model dict, now with threshold info
    '''

    for model_name, threshold in thresholds_dict.items():
        model_info[model_name]['match_thresh'] = threshold

    return model_info


def get_flipped0_potential_edges(phats_df, model_info, allow_clusters_w_multiple_unique_ids):
    '''If allowed, identify the set of labeled 0s with high phats so they can be treated
    as matches downstream.

    Args:
        phats_df (pd.DataFrame): phat info for record pairs

            =====================   =======================================================
            record_id (_1, _2)      unique record identifiers
            model_to_use            based on pair characteristics, which model to use (e.g. basic or no-dob)
            covered_pair            did the pair make it through blocking
            match_train_eligible    is the pair eligible for training (for match model)
            exactmatch              is the pair an exact match on name/dob
            label                   whether the pair is a match or not
            <phat_col>              predicted probability of match
            =====================   =======================================================

        model_info (dict): dict with information about how to fit the model
        allow_clusters_w_multiple_unique_ids (bool): param controlling if 0s can be flipped to 1

    Returns:
        pd.DataFrame: same as phats_df, just
    '''

    if not allow_clusters_w_multiple_unique_ids:
        return pd.DataFrame(columns=phats_df.columns)

    phats_df['flipped0'] = 0

    for model_name, this_model_info in model_info.items():

        phat_col = f'{model_name}_match_phat'
        phats_df.loc[(phats_df.model_to_use == model_name) &
                     (phats_df.label == 0) &
                     (phats_df[phat_col] >= this_model_info['match_thresh']), 'flipped0'] = 1

    return phats_df[phats_df.flipped0 == 1].drop(columns=['flipped0'])
