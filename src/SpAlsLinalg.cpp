#include "SpAlsLinalg.h"
#include "TensorData.h"
#include "CPDecomp.h"

#include <omp.h>
#include <vector>
#include <iostream>

using namespace std;

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