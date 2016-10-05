#pragma once
class TensorCP_ALS;

#include "pRNG.h"
#include "utils.h"
#include "CPDecomp.h"
#include "TensorData.h"

#include <vector>
#include <chrono>
#include <omp.h>
#include <memory>

using namespace std;

class TensorCP_ALS
{
  public:
    TensorCP_ALS(const TensorData &_data, shared_ptr<CPDecomp> &_cpd);

    // void randInit(RNGeng *rng);
    virtual int updateFactor(const unsigned factorId);

    void setErrorRecordInterval(const size_t _interval);
    void setVerbose(const int _verbose);
    const size_t rank;

  protected:
    const TensorData &data;
    shared_ptr<CPDecomp> &cpd;
    size_t currIter;
    int verbose;

    size_t RecordError;
    vector<pair<size_t, T>> absErrorIter;
    vector<pair<size_t, T>> rltErrorIter;
    vector<double> secIter;
    vector<size_t> factorIter;

    // pre allocated memory
    vector<vector<vector<T>>> gramABpdt;
    vector<vector<vector<T>>> gramABpdtInv;
    vector<T> _row;

    void logAfterIter(const unsigned factorId, chrono::time_point<chrono::system_clock> &stepStartTime);
    chrono::time_point<chrono::system_clock> logBeforIter(const unsigned factorId);

    void prepareGramInv(const size_t factorId);
    void updateEntry(const unsigned factorId, const vector<size_t> &froms, const size_t i, const vector<vector<T>> &gramABInv, const T weight);
    void genRow(const unsigned factorId, const vector<size_t> &froms, const vector<size_t> &_loci, const vector<vector<T>> &pinv);
};

class TensorCP_SPALS : public TensorCP_ALS
{
  public:
    TensorCP_SPALS(const TensorDataSpAls &_data, shared_ptr<CPDecomp> &_cpd, SpAlsRNGeng &_rngEng);
    virtual int updateFactor(const unsigned factorId, size_t count);
    virtual int updateFactor(const unsigned factorId);
    void setRate(double _rate);

  protected:
    const TensorDataSpAls &dataSpals;
    double rate;
    SpAlsRNGeng &rngEng;
    // cmf for each factor
    vector<vector<T>> lvrgScores;
    vector<vector<T>> factorCmf;

    void getLvrgScr(const unsigned factorId);
};

class TensorCP_SPALSOMP : public TensorCP_SPALS
{
  public:
    TensorCP_SPALSOMP(const TensorDataSpAls &_data, shared_ptr<CPDecomp> &_cpd, SpAlsRNGeng &_rngEng, size_t _nthread);
    virtual int updateFactor(const unsigned factorId, size_t count);
    virtual int updateFactor(const unsigned factorId);
    const size_t nthread;

  protected:
};