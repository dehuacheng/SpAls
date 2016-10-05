#include "utils.h"
#include "CPDecomp.h"
#include "pRNG.h"

#include <vector>
#include <omp.h>
#include <cmath>

using namespace std;

CPDecomp::CPDecomp(const TensorData &_data, size_t _rank) : CPDecomp(_data.ro_dims, _rank) {}

CPDecomp::CPDecomp(const vector<size_t> &_dims, size_t _rank) : ro_dims(dims), ro_factors(factors), ro_lambdas(lambdas), rank(_rank), dims(_dims)
{
    auto NDIM = dims.size();
    factors = vector<vector<vector<T>>>(NDIM);
    for (int factorId = 0; factorId < NDIM; factorId++)
    {
        factors[factorId] = vector<vector<T>>(dims[factorId], vector<T>(rank));
    }
    lambdas = vector<T>(rank, 1.0);
    isFactorNormalized = vector<bool>(NDIM, false);

    isGramUpdated = vector<bool>(NDIM, false);
    isGramInvUpdated = vector<bool>(NDIM, false);
    gramMtx = vector<vector<vector<T>>>(NDIM, vector<vector<T>>(rank, vector<T>(rank)));
    gramMtxInv = vector<vector<vector<T>>>(NDIM, vector<vector<T>>(rank, vector<T>(rank)));
}

void CPDecomp::randInit(RNGeng *rng)
{
#pragma omp parallel for
    for (int factorId = 0; factorId < dims.size(); ++factorId)
    {
        int tid = omp_get_thread_num();
        isFactorNormalized[factorId] = false;
        isGramUpdated[factorId] = false;
        isGramInvUpdated[factorId] = false;

        for (size_t i = 0; i < dims[factorId]; ++i)
        {
            for (size_t j = 0; j < rank; ++j)
            {
                factors[factorId][i][j] = rng[tid].nextRNG() - 0.5;
            }
        }
        normalizeFactor(factorId);
        updateGram(factorId);
    }
}

void CPDecomp::normalizeFactor(const unsigned factorId)
{
    if (isFactorNormalized[factorId])
    {
        return;
    }
    else
    {
        isGramUpdated[factorId] = false;
        isGramInvUpdated[factorId] = false;

        vector<T> columnNorm(rank, 0);
        size_t nid = dims[factorId];

#pragma omp parallel for
        for (int j = 0; j < rank; ++j)
        {
            for (size_t i = 0; i < nid; ++i)
            {
                T tmp = factors[factorId][i][j];
                columnNorm[j] += tmp * tmp;
            }
        }

        for (size_t j = 0; j < rank; ++j)
        {
            columnNorm[j] = sqrt(columnNorm[j]);
            lambdas[j] *= columnNorm[j];
        }

#pragma omp parallel for
        for (int j = 0; j < rank; ++j)
        {
            for (size_t i = 0; i < nid; ++i)
            {
                factors[factorId][i][j] /= columnNorm[j];
            }
        }

        isFactorNormalized[factorId] = true;
    }
}

T CPDecomp::eval(const vector<size_t> &ind) const
{
    T res = 0.0;
    for (size_t j = 0; j < rank; j++)
    {
        T tmp = lambdas[j];
        for (size_t di = 0; di < ind.size(); di++)
        {
            tmp *= factors[di][ind[di]][j];
        }
        res += tmp;
    }
    return res;
}

void CPDecomp::updateGram()
{
    for (int i = 0; i < dims.size(); i++)
    {
        updateGram(i);
    }
}

void CPDecomp::updateGramInv(const unsigned factorId)
{
    if (isGramInvUpdated[factorId])
    {
        return;
    }
    else
    {
        updateGram(factorId);
        SpAlsUtils::reset(gramMtxInv[factorId]);
        SpAlsUtils::invert(gramMtx[factorId], gramMtxInv[factorId], rank);
        isGramInvUpdated[factorId] = true;
    }
}

void CPDecomp::updateGram(const unsigned factorId)
{
    if (isGramUpdated[factorId])
    {
        return;
    }
    else
    {
        SpAlsUtils::reset(gramMtx[factorId]);
        size_t nid = dims[factorId];

#pragma omp parallel for
        for (size_t j0 = 0; j0 < rank; j0++)
        {
            for (int i = 0; i < nid; i++)
            {
                for (size_t j1 = 0; j1 < rank; j1++)
                {
                    gramMtx[factorId][j0][j1] += factors[factorId][i][j0] * factors[factorId][i][j1];
                }
            }
        }
        isGramUpdated[factorId] = true;
    }
}

const vector<vector<T>> &CPDecomp::getGramMtx(const unsigned factorId)
{
    if (factorId < 0 || factorId >= dims.size())
    {
        cout << "Dimension Index Out of Bound: " << factorId << endl;
        exit(1);
    }
    updateGram(factorId);
    return gramMtx[factorId];
}

const vector<vector<T>> &CPDecomp::getGramMtxInv(const unsigned factorId)
{
    if (factorId < 0 || factorId >= dims.size())
    {
        cout << "Dimension Index Out of Bound: " << factorId << endl;
        exit(1);
    }
    updateGramInv(factorId);
    return gramMtxInv[factorId];
}

const vector<vector<vector<T>>> &CPDecomp::getAllGramMtx()
{
    updateGram();
    return gramMtx;
}