import Levenshtein as edist
import numpy as np
from collections import defaultdict

def _editCounts(name_samp):
    # to compute probability of edit operations use a subsample of names
    edit_count = defaultdict(int)
    p = len(name_samp)
    total_edits = 0
    for i in range(p):
        for j in range(i + 1, p):
            if i < j:
                edits = edist.editops(name_samp[i], name_samp[j])
                p = len(edits)
                lene = p
                total_edits += len(edits)
                for k in range(lene):
                    edit_count[edits[k]] += 1
    return edit_count, total_edits


def _ngramCount(name_list, ngram_len):
    ngram_count = defaultdict(int)
    for i in range(len(name_list)):
        current_name = name_list[i]
        if len(current_name) > ngram_len - 1:
            for start in range(len(current_name) - (ngram_len - 1)):
                ngram_count[current_name[start:(start + ngram_len)]] += 1
                ngram_count[current_name[start:((start + ngram_len)-1)]] += 1
            ngram_count[current_name[-(ngram_len - 1):]] += 1
    return ngram_count


def _probName(name, ngram_len, ngram_count, smoothing, memoize):
    log_prob = 0.0
    for start in range(len(name) - (ngram_len - 1)):
        numer = ngram_count[name[start:(start + ngram_len)]] + smoothing
        denom = ngram_count[name[start:(start + ngram_len)-1]] + smoothing
        if not denom:
            denom += .000001 # avoid div by zero error
        log_prob += np.log(numer / denom)
    memoize[name] = np.exp(log_prob)
    return memoize


def _condProbName(name1, name2, edit_count, total_edits, smoothing, cp_memoize):
    # computes the conditional probability of arriving at name1
    # by performing a series of operation on name2.
    temp_count = defaultdict(float)
    holder = 0.0
    for k, v in edit_count.items():
        temp_count[k] = v / total_edits
    edits = edist.editops(name1, name2)
    for e in edits:
        holder += np.log(temp_count[e] + smoothing)
    log_cnd_prob = np.sum(holder)
    cp_memoize[(name1, name2)] = np.exp(log_cnd_prob)
    return cp_memoize


def _probSamePerson(name1, name2, pop_size, edit_count, total_edits, smoothing,
                    ngram_len, ngram_count, memoize, cp_memoize, psp_memoize):
    # computes the probability that the two names belong to the same person.
    if not memoize[name1]:
        memoize = _probName(name1, ngram_len, ngram_count, smoothing, memoize)
    if not memoize[name2]:
        memoize = _probName(name2, ngram_len, ngram_count, smoothing, memoize)
    if not cp_memoize[(name1, name2)]:
        cp_memoize = _condProbName(name1, name2, edit_count, total_edits, smoothing, cp_memoize)
    p1 = memoize[name1]
    p2 = memoize[name2]
    p2given1 = cp_memoize[(name1, name2)]
    if ((pop_size - 1.0) * p1 * p2 + p1 * p2given1):
        psp_memoize[(name1, name2)] = (p1 * p2given1) / ((pop_size - 1.0) * p1 * p2 + p1 * p2given1)
    else:
        psp_memoize[(name1, name2)] = 0.0
    return [psp_memoize, cp_memoize, memoize]
