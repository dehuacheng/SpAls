#pragma once
class TensorCP_ALS;

#include "pRNG.h"
#include "utils.h"
#include "CPDecomp.h"
#include "TensorData.h"

#include <vector>
#include <omp.h>
#include <memory>

class TensorCP_ALS
{
  public:
    TensorCP_ALS(const TensorData &_data, shared_ptr<CPDecomp> &_cpd);

    void randInit(RNGeng *rng);

    virtual int updateFactor(const unsigned factorId);

    const size_t rank;

  protected:
    const TensorData &data;
    shared_ptr<CPDecomp> &cpd;
    size_t currIter;

    // pre allocated memory
    vector<vector<vector<T>>> gramABpdt;
    vector<vector<vector<T>>> gramABpdtInv;

    void prepareGramInv(const size_t factorId);
    void updateEntry(const unsigned factorId, const vector<size_t> &froms, const size_t i, const vector<vector<T>> &gramABInv, const T weight);
    vector<T> genRow(const unsigned factorId, const vector<size_t> &froms, const vector<size_t> &_loci, const vector<vector<T>> &pinv);
};