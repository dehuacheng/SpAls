#pragma once
typedef double T;

class Linalg;
class RNGeng;
class SpAlsRNGeng;
class SpAlsUtils;

#include "pRNG.h"
#include "asa007.h"
#include <vector>
#include "CPDecomp.h"
#include "TensorData.h"

using namespace std;

class Linalg
{
  public:
    static double Fnorm2(const TensorData &data);
    static double Fnorm2(CPDecomp &cpd);
    static double Fnorm2Diff(const TensorData &data, CPDecomp &cpd);
};

class SpAlsRNGeng
{
  public:
    SpAlsRNGeng(size_t seed = 0)
    {
        tmpeng.seed(seed);
    }
    double nextRNG();
    void seed(size_t seed) { tmpeng.seed(seed); }

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
    static void printVector(const vector<T> &A);
    static unsigned drawFromCmf(const vector<T> &cmf, T val);
    static void pdf2Cmf(const vector<T> &pdf, vector<T> &cmf);
};
