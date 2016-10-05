#pragma once
class CPDecomp;

#include "utils.h"
#include "pRNG.h"

#include "TensorData.h"
#include "TensorAls.h"
#include <vector>

class CPDecomp
{
    friend class TensorCP_ALS;
    friend class TensorCP_SPALS;
    friend class TensorCP_SPALSOMP;

  public:
    CPDecomp() : ro_dims(dims), ro_factors(factors), ro_lambdas(lambdas) {}
    CPDecomp(const TensorData &_data, size_t _rank);
    CPDecomp(const vector<size_t> &_dims, size_t _rank);
    ~CPDecomp(){};

    size_t rank;
    const vector<vector<vector<T>>> &ro_factors;
    const vector<T> &ro_lambdas;
    const vector<size_t> &ro_dims;

    T eval(const vector<size_t> &ind) const;

    void randInit(RNGeng *rng);
    void updateGram();
    const vector<vector<T>> &getGramMtx(const unsigned factorId);
    const vector<vector<T>> &getGramMtxInv(const unsigned factorId);
    const vector<vector<vector<T>>> &getAllGramMtx();

  protected:
    vector<vector<vector<T>>> factors;
    vector<T> lambdas;
    vector<size_t> dims;

    vector<bool> isFactorNormalized;
    vector<bool> isGramUpdated;
    vector<bool> isGramInvUpdated;

    void updateGram(const unsigned factorId);
    void updateGramInv(const unsigned factorId);

    vector<vector<vector<T>>> gramMtx;
    vector<vector<vector<T>>> gramMtxInv;

    void normalizeFactor(const unsigned factorId); //
};