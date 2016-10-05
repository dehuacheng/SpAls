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
    void updateEntry(const unsigned factorId, const vector<size_t> &froms, const size_t i, const vector<vector<T>> &gramABInv, const T weight, vector<T> &row);
    void genRow(const unsigned factorId, const vector<size_t> &froms, const vector<size_t> &_loci, const vector<vector<T>> &pinv, vector<T> &row);
};

class TensorCP_SPALS : public TensorCP_ALS
{
  public:
    TensorCP_SPALS(const TensorDataSpAls &_data, shared_ptr<CPDecomp> &_cpd, SpAlsRNGeng *_rngEng);
    virtual int updateFactor(const unsigned factorId, size_t count);
    virtual int updateFactor(const unsigned factorId);
    void setRate(double _rate);

  protected:
    const TensorDataSpAls &dataSpals;
    double rate;
    SpAlsRNGeng *rngEng;
    // cmf for each factor
    vector<vector<T>> lvrgScores;
    vector<vector<T>> factorCmf;

    void getLvrgScr(const unsigned factorId);
};

class TensorCP_SPALSOMP : public TensorCP_SPALS
{
  public:
    TensorCP_SPALSOMP(const TensorDataSpAls &_data, shared_ptr<CPDecomp> &_cpd, SpAlsRNGeng *_rngEng, size_t _nthread);
    virtual int updateFactor(const unsigned factorId, size_t count);
    virtual int updateFactor(const unsigned factorId);
    virtual int updateFactorAls(const unsigned factorId);
    const size_t nthread;

  protected:
    void getBalanceDist();
    // required only when OpenMP
    // without subsampling
    vector<vector<vector<size_t>>> bDistALS;
    // bDistALS[factorId][thread_id] is all the data_id that bDistSPALS[factorId][loc[thread_id][dimension_id]] = thread_id
    // with subsampling
    vector<vector<size_t>> bDistSPALS;
    // bDistSPALS[factorId][dimension_id] = thread_id

    vector<vector<T>> _rows;
};