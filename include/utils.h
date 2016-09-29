#pragma once
typedef double T;
class RNGeng;
class SpAlsRNGeng;
class SpAlsUtils;

#include "pRNG.h"
#include "asa007.h"
#include <vector>

using namespace std;

class RNGeng
{
  public:
    RNGeng() {}
    virtual double nextRNG() = 0;
};

class SpAlsRNGeng : public RNGeng
{
  public:
    SpAlsRNGeng(sitmo::prng_engine _tmpeng) : tmpeng(_tmpeng) {}
    double nextRNG() override;

  protected:
    sitmo::prng_engine tmpeng;
    double rand01();
};

class SpAlsUtils
{
  public:
    static vector<size_t> getFroms(const int notFrom, const int NDIM);
    static void invert(const vector<vector<T>> &A, vector<vector<T>> &goal, const int N_MAX);
    static void reset(vector<vector<T>> &A);
    static void reset(vector<T> &row);
    static void printMatrix(const vector<vector<T>> &A);
};
