import numpy as np
import os
from collections import defaultdict
import pickle
from .counter import _editCounts, _ngramCount, _probName, _condProbName, _probSamePerson
import importlib

class NameProbability():
    def __init__(self, name_list=None, ngram_len=5, smoothing=.001, unique=0,
        edit_count_max=None, last_comma_first=0, name_list_location=None,
        save_location=None, use_SS=0):
        '''
        - edit_count_max is used to limit the number of samples to consider
        when computing edit operation counts
        '''
        self.smoothing = smoothing
        self.ngram_count = defaultdict(int)
        self.edit_count = defaultdict(int)
        self.ngram_len = ngram_len
        self.pop_size = 0
        self.total_edits = 0
        self.unique = unique
        self.last_comma_first = last_comma_first
        self.edit_count_max = edit_count_max
        self.DATA_PATH = 1
        self.memoize = defaultdict(float)
        self.cp_memoize = defaultdict(float)
        self.psp_memoize = defaultdict(float)
        self.name_list_location = name_list_location
        self.save_location = save_location
        self.name_list = name_list

        if not name_list and not name_list_location and not use_SS:
            raise Exception('Need either a name list, location to name list, or to use SS data')
        if name_list and name_list_location:
            raise Exception('Only one of name_list or name_list_location can be provided')
        name_list_provided = True if name_list or name_list_location else False

        if name_list_provided:
            if name_list_location:
                self.loadNameList()

            if not isinstance(name_list, list):
                self.name_list = list(self.name_list)
            if self.unique:
                self.name_list = list(set(self.name_list))
            if self.last_comma_first:
                for i, n in enumerate(self.name_list):
                    rev_name = n.split(', ')
                    rev_name = rev_name[1] + ' ' + rev_name[0]
                    self.name_list[i] = rev_name

            self.pop_size += len(self.name_list)
            self.ngramCount(self.name_list)
            self.editCounts(self.name_list)

        if not name_list_provided and not use_SS:
            raise Exception('No training data provided. Use either a custom name list or the Social Security data')

        if not name_list_provided and use_SS:
            DATA_PATH = importlib.resources.read_binary(__file__, 'ss_daata.pkl')
            with open(DATA_PATH, 'r') as f:
                ss_data = pickle.load(f)
                self.ngram_count = ss_data[0]
                self.edit_count = ss_data[1]
                self.pop_size = 25e6
                self.total_edits = sum(v for v in self.edit_count.itervalues())


    def ngramCount(self, name_list):
        self.ngram_count = _ngramCount(name_list, self.ngram_len)


    def editCounts(self, name_list):
        # to compute probability of edit operations use a subsample of names
        if self.edit_count_max:
            name_list = np.array(name_list)
            name_samp = name_list[np.random.randint(0, len(name_list),
                                                    self.edit_count_max)].tolist()
        else:
            name_samp = name_list
        edit_count, total_edits = _editCounts(name_samp)
        self.total_edits += total_edits
        for k, v in edit_count.items():
            self.edit_count[k] += v


    def probName(self, name):
        # compute the probability of name based on the training data
        if len(name) < self.ngram_len:
            return 0
        if name not in self.memoize:
            self.memoize = _probName(name, self.ngram_len, self.ngram_count,
                                     self.smoothing, self.memoize)
        return self.memoize[name]


    def condProbName(self, name1, name2):
        # computes the conditional probability of arriving at name1
        # by performing a series of operation on name2.
        if (name1, name2) not in self.cp_memoize:
            self.cp_memoize = _condProbName(name1, name2, self.edit_count, self.total_edits,
                                            self.smoothing, self.cp_memoize)
        return self.cp_memoize[(name1, name2)]


    def probSamePerson(self, name1, name2):
        # computes the probability that the two names belong to the same person.
        if len(name1) < self.ngram_len or len(name2) < self.ngram_len:
            print('Both names should be at least', self.ngram_len, ' characters long')
            return 0.0
        if (name1, name2) not in self.psp_memoize:
            self.psp_memoize, self.cp_memoize, self.memoize = _probSamePerson(name1,
                           name2, self.pop_size, self.edit_count,
                           self.total_edits, self.smoothing, self.ngram_len,
                           self.ngram_count, self.memoize, self.cp_memoize,
                           self.psp_memoize)
        return self.psp_memoize[(name1, name2)]


    def probUnique(self, name):
        # compute the probability that a name is unique in the data
        return 1. / ((self.pop_size - 1) * self.probName(name) + 1)

    def surprisal(self, name):
        return -np.log2(self.probUnique(name))

    def saveObject(self):
        with open(self.save_location, 'w') as f:
            pickle.dump(temp, f)

    def loadNameList(self):
        with open(self.name_list_location, 'r') as f2:
            self.name_list = f2.read().split('\n')
