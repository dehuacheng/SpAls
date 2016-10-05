#include "pRNG.h"
#include "utils.h"
#include "TensorData.h"
#include "CPDecomp.h"

#include <omp.h>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

double SpAlsRNGeng::nextRNG()
{
    return rand01();
}

inline double SpAlsRNGeng::rand01()
{
    double x = 0;
    double rng = 1;
    for (int i = 0; i < 6; ++i)
    {
        x = x * double(1 << 10) + double(tmpeng() % (1 << 10));
        rng *= double(1 << 10);
    }
    return double(x) / double(rng);
}

vector<size_t> SpAlsUtils::getFroms(const int notFrom, const int NDIM)
{
    vector<size_t> froms;
    for (int i = 0; i < NDIM; i++)
        if (i != notFrom)
            froms.push_back(i);

    return froms;
}

void SpAlsUtils::reset(vector<vector<T>> &A)
{
    for (auto &row : A)
    {
        fill(row.begin(), row.end(), 0);
    }
}

void SpAlsUtils::reset(vector<T> &row)
{
    fill(row.begin(), row.end(), 0);
}

void SpAlsUtils::printMatrix(const vector<vector<T>> &A)
{
    cout << "print matrix of size " << A.size() << "-by-" << A[0].size() << endl;
    for (auto const &row : A)
    {
        for (auto const &val : row)
        {
            cout << val << " ";
        }
        cout << endl;
    }
}
void SpAlsUtils::invert(const vector<vector<T>> &A, vector<vector<T>> &goal, const int N_MAX)
{

    T *a = new T[(N_MAX * (N_MAX + 1)) / 2];
    T *c = new T[(N_MAX * (N_MAX + 1)) / 2];
    int ifault;
    int nullty;
    T *w = new T[N_MAX];

    int k = 0;
    for (int i = 0; i < N_MAX; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            a[k] = A[i][j];
            if (i == j)
                a[k] += 1e-9;
            k++;
        }
    }
    syminv(a, N_MAX, c, w, &nullty, &ifault);
    k = 0;
    for (int i = 0; i < N_MAX; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            goal[i][j] = c[k];
            goal[j][i] = c[k];
            k++;
        }
    }

    delete[] a;
    delete[] c;
    delete[] w;
}

unsigned SpAlsUtils::drawFromCmf(const vector<double> &cmf, T val)
{
    auto it = lower_bound(cmf.begin(), cmf.end(), val);
    return it - cmf.begin();
}

void SpAlsUtils::pdf2Cmf(const vector<T> &pdf, vector<T> &cmf)
{
    reset(cmf);
    cmf[0] = pdf[0];
    for (size_t i = 1; i < cmf.size(); i++)
    {
        cmf[i] = cmf[i - 1] + pdf[i];
    }
    T total = cmf.back();
    for (auto &v : cmf)
    {
        v /= total;
    }
    return;
}

void SpAlsUtils::printVector(const vector<T> &row)
{
    cout << "print vector of size " << row.size() << endl;
    for (auto const &val : row)
    {
        cout << val << " ";
    }
    cout << endl;
}

double Linalg::Fnorm2(const TensorData &data)
{

    double s = 0;
#pragma omp parallel for reduction(+ : s)
    for (int i = 0; i < data.ro_nnz; i++)
    {
        s += data.ro_val[i] * data.ro_val[i];
    }
    return s;
}

double Linalg::Fnorm2(CPDecomp &cpd)
{
    cpd.updateGram();

    double ret = 0;
    auto &gramMtx = cpd.getAllGramMtx();
    auto Rank = cpd.rank;
    auto NDIM = cpd.ro_dims.size();
#pragma omp parallel for reduction(+ : ret)
    for (int i = 0; i < Rank; i++)
    {
        for (size_t j = 0; j < Rank; j++)
        {
            double v = cpd.ro_lambdas[i] * cpd.ro_lambdas[j];
            for (size_t k = 0; k < NDIM; k++)
            {
                v *= gramMtx[k][i][j];
            }
            ret += v;
        }
    }
    return ret;
}

double Linalg::Fnorm2Diff(const TensorData &data, CPDecomp &cpd)
{
    cpd.updateGram();

    double ret1 = 0;
    auto &gramMtx = cpd.getAllGramMtx();
    auto Rank = cpd.rank;
    auto NDIM = cpd.ro_dims.size();
#pragma omp parallel for reduction(+ : ret1)
    for (int i = 0; i < Rank; i++)
    {
        for (size_t j = 0; j < Rank; j++)
        {
            double v = cpd.ro_lambdas[i] * cpd.ro_lambdas[j];
            for (size_t k = 0; k < NDIM; k++)
            {
                v *= gramMtx[k][i][j];
            }
            ret1 += v;
        }
    }

    double ret2 = 0;
    const auto &loc = data.ro_loc;
    const auto &nnz = data.ro_nnz;
    const auto &val = data.ro_val;
#pragma omp parallel for reduction(+ : ret2)
    for (int i = 0; i < nnz; i++)
    {
        double v = cpd.eval(loc[i]);
        ret2 -= v * v;
        ret2 += (v - val[i]) * (v - val[i]);
        ;
    }
    return ret1 + ret2;
}