#include "utils.h"
#include "TensorAls.h"
#include "CPDecomp.h"
#include "TensorData.h"

#include <memory>
#include <vector>
#include <chrono>

using namespace std;

TensorCP_ALS::TensorCP_ALS(const TensorData &_data, shared_ptr<CPDecomp> &_cpd) : data(_data), cpd(_cpd), rank(_cpd->rank)
{
    gramABpdt = vector<vector<vector<T>>>(data.dims.size(), vector<vector<T>>(rank, vector<T>(rank)));
    gramABpdtInv = vector<vector<vector<T>>>(data.dims.size(), vector<vector<T>>(rank, vector<T>(rank)));
}

int TensorCP_ALS::updateFactor(const unsigned factorId)
{

    chrono::duration<double> stepTime(0);
    auto stepStartTime = chrono::system_clock::now();

    currIter++;
    cpd->lambdas = vector<T>(rank, 1.0);
    SpAlsUtils::reset(cpd->factors[factorId]);

    prepareGramInv(factorId);
    vector<size_t> froms = SpAlsUtils::getFroms(factorId, cpd->dims.size());

    for (size_t i = 0; i < data.nnz; i++)
    {
        updateEntry(factorId, froms, i, gramABpdtInv[factorId], 1.0);
    }

    cpd->normalizeFactor(factorId);
    cpd->getGramMtx(factorId);

    // if (!RecordError)
    // {
    //     if (currIter % RecordError == 0)
    //     {
    //         absErrorIter.push_back(diffOnly());
    //         rltErrorIter.push_back(absErrorIter.back() / normT);
    //     }
    // }
    // //record time
    auto stepEndTime = chrono::system_clock::now();
    stepTime += stepEndTime - stepStartTime;
    // secIter.push_back(stepTime.count());

    return 0;
}

void TensorCP_ALS::prepareGramInv(const size_t factorId)
{
    vector<size_t> froms = SpAlsUtils::getFroms(factorId, cpd->dims.size());

    auto &gramAB = gramABpdt[factorId];
    auto &gramABInv = gramABpdtInv[factorId];

    SpAlsUtils::reset(gramAB);
    SpAlsUtils::reset(gramABInv);

#pragma omp parallel for
    for (int i = 0; i < rank; i++)
    {
        for (size_t j = 0; j < rank; j++)
        {
            gramAB[i][j] = 1.0;
            for (auto &factorId : froms)
            {
                auto &gramMtx = cpd->getGramMtx(factorId);
                gramAB[i][j] *= gramMtx[i][j];
            }
        }
    }
    SpAlsUtils::invert(gramAB, gramABInv, rank);
}

void TensorCP_ALS::updateEntry(const unsigned factorId, const vector<size_t> &froms,
                               const size_t i, const vector<vector<T>> &gramABInv, const T weight)
{
    vector<T> _row = genRow(factorId, froms, data.loc[i], gramABInv);
    for (size_t j = 0; j < rank; ++j)
    {
        cpd->factors[factorId][data.loc[i][factorId]][j] += data.val[i] * _row[j] * weight;
    }
}

vector<T> TensorCP_ALS::genRow(const unsigned factorId, const vector<size_t> &froms, const vector<size_t> &_loci, const vector<vector<T>> &pinv)
{
    vector<T> row(rank, 0);
    for (size_t j = 0; j < rank; j++)
    {
        T krpJ = cpd->factors[froms[0]][_loci[froms[0]]][j] * cpd->factors[froms[1]][_loci[froms[1]]][j];
        for (size_t k = 0; k < rank; k++)
        {
            row[k] += pinv[k][j] * krpJ;
        }
    }
    return row;
}